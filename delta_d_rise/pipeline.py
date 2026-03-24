import csv
import os
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torchvision.transforms as T
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from .config import DeltaDRISEConfig
from .core import DeltaDRISE
from .interfaces import GeneralObjectDetectionModelWrapper
from .results import (
    ExplainedDetection,
    DetectionEvaluation,
    SavedArtifacts,
    FullDeltaDRISEResult,
)
from .evaluation import (
    set_all_seeds,
    object_level_stats,
    apply_deletion,
    match_base_to_pred_conf,
    auc_trapz,
    saliency_to_grid_vector,
    spearman_corr,
    topk_overlap,
)
from .visualization import ensure_dir, overlay_heatmap_np, save_base_detections_image


def run_delta_d_rise_full(
    image_path: str,
    model: GeneralObjectDetectionModelWrapper,
    class_names: Dict[int, str],
    config: DeltaDRISEConfig,
    out_dir: str,
    conf_thr: float = 0.25,
    device: str = "cpu",
    verbose: bool = True,
    robustness_runs: int = 5,
    seed0: int = 0,
    grid: Tuple[int, int] = (16, 16),
    topk_frac: float = 0.10,
    invert_for_display: bool = False,
    preset_name: str = "CUSTOM",
) -> FullDeltaDRISEResult:
    ensure_dir(out_dir)

    img_pil = Image.open(image_path).convert("RGB")
    img_rgb = np.array(img_pil).astype(np.uint8)
    img_tensor = T.ToTensor()(img_pil).unsqueeze(0)

    with torch.no_grad():
        base_list = model.predict(img_tensor.to(device))
    base = base_list[0]
    if base.bounding_boxes.shape[0] == 0:
        raise ValueError("No detections found.")

    boxes_np = base.bounding_boxes.cpu().numpy()
    cls_scores_np = base.class_scores.cpu().numpy()
    labels_np = np.argmax(cls_scores_np, axis=1)
    confs_np = np.max(cls_scores_np, axis=1)
    num_dets = len(boxes_np)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    base_det_img_path = os.path.join(out_dir, f"{base_name}_base_detections.png")
    save_base_detections_image(
        img_rgb=img_rgb,
        boxes_np=boxes_np,
        labels_np=labels_np,
        confs_np=confs_np,
        class_names=class_names,
        out_path=base_det_img_path
    )

    explainer = DeltaDRISE(config)
    result = explainer.explain(model=model, image=img_tensor, device=device, verbose=verbose)

    raw_map_paths: List[str] = []
    heatmap_paths: List[str] = []

    for det in result.detections:
        det_idx = det.det_idx
        sal_np = det.saliency_map

        raw_path = os.path.join(out_dir, f"{base_name}_det{det_idx}_delta_d_rise.npy")
        np.save(raw_path, sal_np)
        raw_map_paths.append(raw_path)

        vis_heat = (1.0 - sal_np) if invert_for_display else sal_np
        vis = overlay_heatmap_np(img_rgb, vis_heat, alpha=0.55)

        base_box = boxes_np[det_idx]
        base_cls = int(labels_np[det_idx])
        cls_name = class_names.get(int(base_cls), str(base_cls))

        x1, y1, x2, y2 = base_box.astype(int)
        vis_box = vis.copy()
        cv2.rectangle(vis_box, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(
            vis_box, f"{cls_name} idx={det_idx}",
            (x1, max(10, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
        )

        heat_path = os.path.join(out_dir, f"{base_name}_det{det_idx}_delta_heat.png")
        plt.figure(figsize=(6, 6))
        plt.imshow(vis_box)
        plt.axis("off")
        plt.title(f"{base_name} det{det_idx} ({cls_name}) - Delta D-RISE")
        plt.tight_layout()
        plt.savefig(heat_path, dpi=200)
        plt.close()
        heatmap_paths.append(heat_path)

    gh, gw = grid
    K = gh * gw
    baseline_uniform = 1.0 / K
    topk = int(max(1, round(topk_frac * K)))
    fracs = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

    det_grid_vectors = [[None for _ in range(num_dets)] for _ in range(robustness_runs)]
    det_deletion_auc = np.zeros((robustness_runs, num_dets), dtype=np.float32)
    det_io_ratio = np.zeros((robustness_runs, num_dets), dtype=np.float32)

    for r in range(robustness_runs):
        seed = seed0 + r
        set_all_seeds(seed)

        result_r = explainer.explain(model=model, image=img_tensor, device=device, verbose=False)

        for det in result_r.detections:
            det_idx = det.det_idx
            sal_np = det.saliency_map

            st = object_level_stats(sal_np, boxes_np[det_idx])
            det_io_ratio[r, det_idx] = float(st["io_ratio"])

            base_box = boxes_np[det_idx]
            base_cls = int(labels_np[det_idx])
            deletion_confs = []

            for frac in fracs:
                del_img = apply_deletion(
                    img_tensor_1x3hw=img_tensor.to(device),
                    saliency_hw=sal_np,
                    box_xyxy=base_box,
                    frac=float(frac),
                    inside_box=True,
                    mean_fill=config.mean_fill,
                )
                with torch.no_grad():
                    pred = model.predict(del_img)[0]

                matched_conf = match_base_to_pred_conf(
                    base_box=base_box,
                    base_cls=base_cls,
                    pred=pred,
                    iou_thr=config.iou_match_threshold,
                    match_same_class=config.match_same_class,
                )
                deletion_confs.append(float(matched_conf))

            det_deletion_auc[r, det_idx] = float(auc_trapz([float(x) for x in fracs], deletion_confs))

            vec = saliency_to_grid_vector(sal_np, grid=grid, normalize_sum1=True)
            det_grid_vectors[r][det_idx] = vec

    evaluations: List[DetectionEvaluation] = []

    for det_idx in range(num_dets):
        X = np.stack([det_grid_vectors[r][det_idx] for r in range(robustness_runs)], axis=0)

        p05 = np.percentile(X, 5, axis=0)
        p50 = np.percentile(X, 50, axis=0)
        robust_mask = p05 > baseline_uniform
        robust_regions_count = int(robust_mask.sum())
        robust_regions_frac = float(robust_regions_count / (K + 1e-9))

        median_vec = p50
        spears = [spearman_corr(X[r], median_vec) for r in range(robustness_runs)]
        overlaps = [topk_overlap(X[r], median_vec, k=topk) for r in range(robustness_runs)]
        spearman_p50 = float(np.percentile(spears, 50))
        topk_overlap_p50 = float(np.percentile(overlaps, 50))

        deletion_auc_p50 = float(np.percentile(det_deletion_auc[:, det_idx], 50))
        io_ratio_p50 = float(np.percentile(det_io_ratio[:, det_idx], 50))

        base_box = boxes_np[det_idx]
        base_cls = int(labels_np[det_idx])
        base_conf = float(confs_np[det_idx])
        cls_name = class_names.get(int(base_cls), str(base_cls))

        evaluations.append(
            DetectionEvaluation(
                image=base_name,
                det_idx=int(det_idx),
                class_name=str(cls_name),
                base_conf=float(base_conf),
                x1=float(base_box[0]),
                y1=float(base_box[1]),
                x2=float(base_box[2]),
                y2=float(base_box[3]),
                robust_regions_count=int(robust_regions_count),
                robust_regions_frac=float(robust_regions_frac),
                spearman_p50=float(spearman_p50),
                topk_overlap_p50=float(topk_overlap_p50),
                deletion_auc_p50=float(deletion_auc_p50),
                io_ratio_p50=float(io_ratio_p50),
                robustness_runs=int(robustness_runs),
                grid=str(grid),
                topk=int(topk),
                method="DELTA-D-RISE",
                preset=preset_name,
            )
        )

    csv_path = os.path.join(out_dir, f"{base_name}_delta_d_rise_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(evaluations[0].__dict__.keys()))
        writer.writeheader()
        writer.writerows([e.__dict__ for e in evaluations])

    return FullDeltaDRISEResult(
        image_name=base_name,
        image_shape=(img_tensor.shape[-2], img_tensor.shape[-1]),
        detections=result.detections,
        evaluations=evaluations,
        artifacts=SavedArtifacts(
            base_detections_image=base_det_img_path,
            raw_map_paths=raw_map_paths,
            heatmap_paths=heatmap_paths,
            summary_csv_path=csv_path,
        ),
    )
