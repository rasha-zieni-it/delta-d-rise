from dataclasses import dataclass
from typing import List, Optional, Tuple

import copy
import numpy as np
import torch
from torch import Tensor
import torchvision.transforms as T
from PIL import Image
import tqdm

from .config import DeltaDRISEConfig
from .interfaces import DetectionRecord, GeneralObjectDetectionModelWrapper
from .results import ExplainedDetection, ExplanationResult


@dataclass
class MaskAffinityRecord:
    """
    Stores one occlusion mask and its per-detection affinity scores.

    DELTA-D-RISE attributes importance to removed pixels, so the stored mask
    is the occlusion mask (1 - keep_mask), weighted by the increase in
    detection loss caused by masking.
    """
    mask: torch.Tensor
    affinity_scores: List[torch.Tensor]

    def get_weighted_masks(self) -> List[torch.Tensor]:
        out = []
        for s in self.affinity_scores:
            out.append(s.unsqueeze(1).unsqueeze(1).unsqueeze(1) * self.mask)
        return out


def compute_intersections(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    n, m = boxes_a.shape[0], boxes_b.shape[0]
    if n == 0 or m == 0:
        return torch.zeros((n, m), device=boxes_a.device)
    a = boxes_a.unsqueeze(1).repeat(1, m, 1)
    b = boxes_b.unsqueeze(0).repeat(n, 1, 1)
    left = torch.max(a[:, :, 0], b[:, :, 0])
    right = torch.min(a[:, :, 2], b[:, :, 2])
    top = torch.max(a[:, :, 1], b[:, :, 1])
    bottom = torch.min(a[:, :, 3], b[:, :, 3])
    zero = torch.zeros_like(left)
    w = torch.max(zero, right - left)
    h = torch.max(zero, bottom - top)
    return w * h


def compute_areas(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.shape[0] == 0:
        return torch.zeros((0,), device=boxes.device)
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def compute_ious(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    inter = compute_intersections(boxes_a, boxes_b)
    union = compute_areas(boxes_a).unsqueeze(1) + compute_areas(boxes_b).unsqueeze(0) - inter
    return inter / (union + 1e-9)


def scores_to_probs(class_scores: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    probs = torch.clamp(class_scores, min=0.0)
    return probs / (probs.sum(dim=1, keepdim=True) + eps)


def generate_keep_mask(
    base_size: Tuple[int, int],
    img_size: Tuple[int, int],
    padding: int,
    device: str,
    p_keep: float,
) -> torch.Tensor:
    base = (torch.rand(base_size, device=device) < p_keep).float()
    mask = base.repeat([3, 1, 1])
    resized = T.Resize((img_size[0] + padding, img_size[1] + padding), interpolation=Image.NEAREST)(mask)
    cropped = T.RandomCrop(img_size)(resized)
    return cropped


def fuse_mask_mean_fill(img_tensor: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    keep = keep_mask.unsqueeze(0)
    mean = img_tensor.mean(dim=(2, 3), keepdim=True)
    return img_tensor * keep + mean * (1.0 - keep)


def fuse_mask_zero_fill(img_tensor: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    return img_tensor * keep_mask.unsqueeze(0)


def saliency_fusion(
    records: List[MaskAffinityRecord],
    device: str,
    normalize: bool = True,
) -> List[List[dict]]:
    if len(records) == 0:
        return []

    avg_scores = copy.deepcopy(records[0].affinity_scores)
    weighted_acc = copy.deepcopy(records[0].get_weighted_masks())
    unweighted_acc = records[0].mask.clone()

    for rec in records[1:]:
        unweighted_acc += rec.mask
        wms = rec.get_weighted_masks()
        for a, w in zip(weighted_acc, wms):
            a += w
        for a_s, s in zip(avg_scores, rec.affinity_scores):
            a_s += s

    n = len(records)
    for s in avg_scores:
        s /= (n + 1e-9)

    if normalize:
        for a_s, wacc in zip(avg_scores, weighted_acc):
            wacc -= a_s.unsqueeze(1).unsqueeze(1).unsqueeze(1) * unweighted_acc

    out: List[List[dict]] = []
    for det_stack in weighted_acc:
        det_list = []
        for m in det_stack:
            m = m.to(device)
            m = m - m.min()
            m = m / (m.max() + 1e-9)
            det_list.append({"detection": m})
        out.append(det_list)
    return out


def compute_delta_detection_loss_scores(
    base_detections: DetectionRecord,
    masked_detections: Optional[DetectionRecord],
    w_cls: float,
    w_loc: float,
    iou_match_threshold: float,
    match_same_class: bool,
    eps: float = 1e-9,
) -> torch.Tensor:
    base_detections.to("cpu")
    if masked_detections is not None:
        masked_detections.to("cpu")

    boxes_base = base_detections.bounding_boxes
    scores_base = base_detections.class_scores
    d_base = boxes_base.shape[0]
    if d_base == 0:
        return torch.zeros(0)

    probs_base = scores_to_probs(scores_base, eps=eps)
    base_labels = torch.argmax(probs_base, dim=1)
    base_p = probs_base[torch.arange(d_base), base_labels]
    l_cls_base = -torch.log(base_p + eps)
    l_loc_base = torch.zeros(d_base)
    l_det_base = w_cls * l_cls_base + w_loc * l_loc_base

    if (masked_detections is None) or (masked_detections.bounding_boxes.shape[0] == 0):
        l_cls_mask = -torch.log(torch.zeros(d_base) + eps)
        l_loc_mask = torch.ones(d_base)
        l_det_mask = w_cls * l_cls_mask + w_loc * l_loc_mask
        return torch.clamp(l_det_mask - l_det_base, min=0.0)

    boxes_m = masked_detections.bounding_boxes
    scores_m = masked_detections.class_scores
    probs_m = scores_to_probs(scores_m, eps=eps)
    m_labels = torch.argmax(probs_m, dim=1)

    iou_mat = compute_ious(boxes_base, boxes_m)

    if match_same_class:
        same = (base_labels.unsqueeze(1) == m_labels.unsqueeze(0))
        iou_mat = torch.where(same, iou_mat, torch.zeros_like(iou_mat))

    best_iou, best_idx = torch.max(iou_mat, dim=1)

    matched = best_iou >= iou_match_threshold
    p_mask = torch.zeros(d_base)
    if boxes_m.shape[0] > 0:
        p_mask[matched] = probs_m[best_idx[matched], base_labels[matched]]

    l_cls_mask = -torch.log(p_mask + eps)
    l_loc_mask = 1.0 - best_iou
    l_det_mask = w_cls * l_cls_mask + w_loc * l_loc_mask

    return torch.clamp(l_det_mask - l_det_base, min=0.0)


class DeltaDRISE:
    """
    DELTA-D-RISE: detection-loss based perturbation explainability method
    for object detection.
    """

    def __init__(self, config: DeltaDRISEConfig):
        self.config = config

    def explain(
        self,
        model: GeneralObjectDetectionModelWrapper,
        image: Tensor,
        device: str = "cpu",
        verbose: bool = False,
    ) -> ExplanationResult:
        image = image.to(device)
        model = model.to(device)

        with torch.no_grad():
            base_list = model.predict(image)

        base = base_list[0]
        img_size = image.shape[-2:]
        padding = int(max(img_size[0] / self.config.mask_res[0], img_size[1] / self.config.mask_res[1]))

        records: List[MaskAffinityRecord] = []
        iterator = tqdm.tqdm(range(self.config.num_masks)) if verbose else range(self.config.num_masks)

        for _ in iterator:
            keep_mask = generate_keep_mask(
                self.config.mask_res,
                img_size,
                padding,
                device,
                p_keep=self.config.p_keep,
            )
            occ_mask = (1.0 - keep_mask).detach().cpu()

            if self.config.mean_fill:
                masked_img = fuse_mask_mean_fill(image, keep_mask)
            else:
                masked_img = fuse_mask_zero_fill(image, keep_mask)

            with torch.no_grad():
                masked_dets_list = model.predict(masked_img)

            aff_list: List[torch.Tensor] = []
            for base_det, masked_det in zip(base_list, masked_dets_list):
                s = compute_delta_detection_loss_scores(
                    base_detections=base_det,
                    masked_detections=masked_det,
                    w_cls=self.config.w_cls,
                    w_loc=self.config.w_loc,
                    iou_match_threshold=self.config.iou_match_threshold,
                    match_same_class=self.config.match_same_class,
                )
                aff_list.append(s.detach().cpu())

            records.append(MaskAffinityRecord(mask=occ_mask, affinity_scores=aff_list))

        saliency = saliency_fusion(records, device=device, normalize=self.config.fusion_normalize)
        image_sals = saliency[0]

        boxes_np = base.bounding_boxes.cpu().numpy()
        cls_scores_np = base.class_scores.cpu().numpy()
        labels_np = np.argmax(cls_scores_np, axis=1)
        confs_np = np.max(cls_scores_np, axis=1)

        explained = []
        for det_idx, det_sal in enumerate(image_sals):
            sal_3 = det_sal["detection"].detach().cpu()
            sal_hw = sal_3.mean(dim=0)
            sal_hw = sal_hw - sal_hw.min()
            sal_hw = sal_hw / (sal_hw.max() + 1e-9)
            sal_np = sal_hw.numpy().astype(np.float32)

            explained.append(
                ExplainedDetection(
                    det_idx=det_idx,
                    box_xyxy=boxes_np[det_idx],
                    class_id=int(labels_np[det_idx]),
                    confidence=float(confs_np[det_idx]),
                    saliency_map=sal_np,
                )
            )

        return ExplanationResult(
            image_shape=(image.shape[-2], image.shape[-1]),
            detections=explained,
        )
