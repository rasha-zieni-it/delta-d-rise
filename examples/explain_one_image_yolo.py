"""
Minimal example: explain one image using DELTA-D-RISE with a YOLO detector.

Before running:
- update image_path
- update weights_path
"""

import numpy as np
from PIL import Image
import torchvision.transforms as T
from ultralytics import YOLO

from delta_d_rise import DeltaDRISE, DeltaDRISEConfig
from delta_d_rise.wrappers import YOLODetectionWrapper


# ---- User inputs ----
image_path = "path/to/your/image.jpg"
weights_path = "path/to/your/model.pt"


# ---- Load image ----
img_pil = Image.open(image_path).convert("RGB")
img_tensor = T.ToTensor()(img_pil).unsqueeze(0)


# ---- Load detector ----
detector = YOLO(weights_path)
wrapper = YOLODetectionWrapper(detector, conf=0.25, imgsz=768)


# ---- Config ----
config = DeltaDRISEConfig(
    num_masks=1500,
    mask_res=(32, 32),
    p_keep=0.93,
    iou_match_threshold=0.15,
)


# ---- Run explainer ----
explainer = DeltaDRISE(config)
result = explainer.explain(wrapper, img_tensor, device="cpu", verbose=True)


# ---- Print results ----
print("Image shape:", result.image_shape)

for det in result.detections:
    print(
        f"det_idx={det.det_idx}, class_id={det.class_id}, "
        f"conf={det.confidence:.3f}, box={det.box_xyxy}"
    )
