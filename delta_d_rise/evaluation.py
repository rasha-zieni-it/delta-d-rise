import random
from typing import Dict, List, Tuple
import numpy as np
import torch

from .interfaces import DetectionRecord


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def object_level_stats(saliency_hw: np.ndarray, box_xyxy: np.ndarray, eps: float = 1e-9) -> Dict[str, float]:
    h, w = saliency_hw.shape
    x1, y1, x2, y2 = box_xyxy.astype(float)
    x1 = int(np.clip(round(x1), 0, w - 1))
    x2 = int(np.clip(round(x2), 0, w))
    y1 = int(np.clip(round(y1), 0, h - 1))
    y2 = int(np.clip(round(y2), 0, h))
    if x2 <= x1 or y2 <= y1:
        return {"obj_mean": float("nan"), "obj_max": float("nan"), "io_ratio": float("nan")}
    inside = saliency_hw[y1:y2, x1:x2]
    return {
        "obj_mean": float(inside.mean()),
        "obj_max": float(inside.max()),
        "io_ratio": float(inside.sum() / (saliency_hw.sum() + eps))
    }


def apply_deletion(
    img_tensor_1x3hw: torch.Tensor,
    saliency_hw: np.ndarray,
    box_xyxy: np.ndarray,
    frac: float,
    inside_box: bool,
    mean_fill: bool
) -> torch.Tensor:
    img = img_tensor_1x3hw.clone()
    _, _, h, w = img.shape

    region = np.ones((h, w), dtype=bool)
    if inside_box:
        x1, y1, x2, y2 = box_xyxy.astype(int)
        x1 = int(np.clip(x1, 0, w - 1))
        x2 = int(np.clip(x2, 0, w))
        y1 = int(np.clip(y1, 0, h - 1))
        y2 = int(np.clip(y2, 0, h))
        region[:] = False
        region[y1:y2, x1:x2] = True

    s = saliency_hw.copy()
    s[~region] = -1.0

    flat = s.reshape(-1)
    valid_idx = np.where(flat >= 0)[0]
    if len(valid_idx) == 0:
        return img

    k = int(round(frac * len(valid_idx)))
    if k <= 0:
        return img

    valid_scores = flat[valid_idx]
    top_order = np.argsort(-valid_scores)
    chosen = valid_idx[top_order[:k]]

    mask_del = np.zeros(h * w, dtype=np.float32)
    mask_del[chosen] = 1.0
    mask_del = mask_del.reshape(h, w)

    mask_del_t = torch.from_numpy(mask_del).to(img.device).float()
    mask_del_t = mask_del_t.unsqueeze(0).unsqueeze(0)
    mask_del_t3 = mask_del_t.repeat(1, 3, 1, 1)

    if mean_fill:
        mean = img.mean(dim=(2, 3), keepdim=True)
        img = img * (1.0 - mask_del_t3) + mean * mask_del_t3
    else:
        img = img * (1.0 - mask_del_t3)

    return img


def match_base_to_pred_conf(
    base_box: np.ndarray,
    base_cls: int,
    pred: DetectionRecord,
    iou_thr: float,
    match_same_class: bool
) -> float:
    if pred.bounding_boxes.shape[0] == 0:
        return 0.0

    boxes = pred.bounding_boxes.cpu().numpy()
    scores = pred.class_scores.cpu().numpy()
    labels = np.argmax(scores, axis=1)

    bb = base_box.astype(np.float32)
    x1, y1, x2, y2 = bb
    best_iou = 0.0
    best_conf = 0.0

    for b, lab, cls_vec in zip(boxes, labels, scores):
        if match_same_class and int(lab) != int(base_cls):
            continue
        xx1 = max(x1, float(b[0]))
        yy1 = max(y1, float(b[1]))
        xx2 = min(x2, float(b[2]))
        yy2 = min(y2, float(b[3]))
        iw = max(0.0, xx2 - xx1)
        ih = max(0.0, yy2 - yy1)
        inter = iw * ih
        area_a = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
        union = area_a + area_b - inter + 1e-9
        iou = inter / union
        if iou >= iou_thr and iou > best_iou:
            best_iou = iou
            best_conf = float(cls_vec[int(base_cls)])

    return best_conf


def auc_trapz(xs: List[float], ys: List[float]) -> float:
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    area = 0.0
    for i in range(1, len(xs)):
        area += float(xs[i] - xs[i - 1]) * float(ys[i] + ys[i - 1]) / 2.0
    return area


def saliency_to_grid_vector(
    saliency_hw: np.ndarray,
    grid: Tuple[int, int] = (16, 16),
    normalize_sum1: bool = True,
    eps: float = 1e-9
) -> np.ndarray:
    h, w = saliency_hw.shape
    gh, gw = grid
    ys = np.linspace(0, h, gh + 1).astype(int)
    xs = np.linspace(0, w, gw + 1).astype(int)

    vec = np.zeros((gh * gw,), dtype=np.float32)
    idx = 0
    for i in range(gh):
        for j in range(gw):
            y1, y2 = ys[i], ys[i + 1]
            x1, x2 = xs[j], xs[j + 1]
            vec[idx] = float(saliency_hw[y1:y2, x1:x2].sum())
            idx += 1

    if normalize_sum1:
        s = vec.sum()
        vec = vec / (s + eps)

    return vec


def spearman_corr(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    def ranks(v):
        order = np.argsort(v)
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(len(v), dtype=np.float64)
        return r

    ra, rb = ranks(a), ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = (np.sqrt((ra ** 2).sum()) * np.sqrt((rb ** 2).sum())) + eps
    return float((ra * rb).sum() / denom)


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    k = int(max(1, min(k, len(a))))
    ia = set(np.argsort(-a)[:k].tolist())
    ib = set(np.argsort(-b)[:k].tolist())
    inter = len(ia & ib)
    union = len(ia | ib) + 1e-9
    return float(inter / union)
