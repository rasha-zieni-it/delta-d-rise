from typing import List
import numpy as np
import torch
from ultralytics import YOLO

from .interfaces import DetectionRecord, GeneralObjectDetectionModelWrapper


def expand_class_scores(scores: torch.Tensor, labels: torch.Tensor, number_of_classes: int) -> torch.Tensor:
    d = scores.shape[0]
    expanded = torch.ones(d, number_of_classes, dtype=torch.float32)
    for i, (s, lab) in enumerate(zip(scores, labels)):
        residual = (1.0 - float(s)) / (number_of_classes - 1 + 1e-9)
        expanded[i, :] *= residual
        expanded[i, int(lab.item())] = float(s)
    return expanded


class YOLODetectionWrapper(GeneralObjectDetectionModelWrapper):
    """
    Example Ultralytics YOLO wrapper for DELTA-D-RISE.
    """

    def __init__(self, yolo_model: YOLO, conf: float = 0.25, imgsz: int = 640):
        super().__init__()
        self.yolo_model = yolo_model
        self.conf = conf
        self.imgsz = imgsz
        self.num_classes = len(self.yolo_model.names)

    def predict(self, x: torch.Tensor) -> List[DetectionRecord]:
        x_cpu = x.detach().cpu()
        b, c, h, w = x_cpu.shape
        imgs = (x_cpu.permute(0, 2, 3, 1).numpy() * 255.0).astype(np.uint8)

        outs: List[DetectionRecord] = []
        for i in range(b):
            res = self.yolo_model.predict(
                source=imgs[i],
                imgsz=self.imgsz,
                conf=self.conf,
                verbose=False,
            )

            if len(res) == 0 or res[0].boxes is None or res[0].boxes.xyxy is None:
                outs.append(
                    DetectionRecord(
                        bounding_boxes=torch.zeros((0, 4)),
                        objectness_scores=torch.zeros((0,)),
                        class_scores=torch.zeros((0, self.num_classes)),
                    )
                )
                continue

            boxes = res[0].boxes
            xyxy = boxes.xyxy.cpu()
            confs = boxes.conf.cpu()
            clss = boxes.cls.cpu().long()

            objectness = torch.ones_like(confs)
            class_scores = expand_class_scores(confs, clss, self.num_classes)

            outs.append(
                DetectionRecord(
                    bounding_boxes=xyxy,
                    objectness_scores=objectness,
                    class_scores=class_scores,
                )
            )
        return outs
