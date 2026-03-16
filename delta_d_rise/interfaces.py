from dataclasses import dataclass
from typing import List
import torch


@dataclass
class DetectionRecord:
    bounding_boxes: torch.Tensor
    objectness_scores: torch.Tensor
    class_scores: torch.Tensor

    def to(self, device: str) -> None:
        self.bounding_boxes = self.bounding_boxes.to(device)
        self.objectness_scores = self.objectness_scores.to(device)
        self.class_scores = self.class_scores.to(device)


class GeneralObjectDetectionModelWrapper(torch.nn.Module):
    """
    Generic detector wrapper interface for DELTA-D-RISE.

    Any detector-specific backend should inherit from this class and implement
    `predict`, returning one DetectionRecord per input image.
    """

    def predict(self, x: torch.Tensor) -> List[DetectionRecord]:
        raise NotImplementedError
