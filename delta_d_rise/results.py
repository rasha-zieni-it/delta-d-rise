from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class ExplainedDetection:
    det_idx: int
    box_xyxy: np.ndarray
    class_id: int
    confidence: float
    saliency_map: np.ndarray


@dataclass
class ExplanationResult:
    image_shape: Tuple[int, int]
    detections: List[ExplainedDetection]
