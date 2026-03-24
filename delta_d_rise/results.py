from dataclasses import dataclass
from typing import List, Optional, Tuple
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


@dataclass
class DetectionEvaluation:
    image: str
    det_idx: int
    class_name: str
    base_conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    robust_regions_count: int
    robust_regions_frac: float
    spearman_p50: float
    topk_overlap_p50: float
    deletion_auc_p50: float
    io_ratio_p50: float
    robustness_runs: int
    grid: str
    topk: int
    method: str
    preset: str


@dataclass
class SavedArtifacts:
    base_detections_image: Optional[str]
    raw_map_paths: List[str]
    heatmap_paths: List[str]
    summary_csv_path: Optional[str]


@dataclass
class FullDeltaDRISEResult:
    image_name: str
    image_shape: Tuple[int, int]
    detections: List[ExplainedDetection]
    evaluations: List[DetectionEvaluation]
    artifacts: SavedArtifacts
