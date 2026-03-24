import os
from typing import Dict
import cv2
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def overlay_heatmap_np(rgb_img: np.ndarray, heat: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    heat = np.clip(heat, 0.0, 1.0)
    heat_255 = (heat * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_255, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    return (alpha * heat_color + (1.0 - alpha) * rgb_img).astype(np.uint8)


def save_base_detections_image(
    img_rgb: np.ndarray,
    boxes_np: np.ndarray,
    labels_np: np.ndarray,
    confs_np: np.ndarray,
    class_names: Dict[int, str],
    out_path: str
):
    vis = img_rgb.copy()

    for det_idx, (box, lab, conf) in enumerate(zip(boxes_np, labels_np, confs_np)):
        x1, y1, x2, y2 = box.astype(int)
        cls_name = class_names.get(int(lab), str(lab))

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"{cls_name} {conf:.3f} idx={det_idx}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    plt.figure(figsize=(8, 8))
    plt.imshow(vis)
    plt.axis("off")
    plt.title("Baseline detections")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
