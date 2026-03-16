from dataclasses import dataclass
from typing import Tuple


@dataclass
class DeltaDRISEConfig:
    num_masks: int = 1500
    mask_res: Tuple[int, int] = (32, 32)
    p_keep: float = 0.93
    iou_match_threshold: float = 0.15
    w_cls: float = 1.0
    w_loc: float = 1.0
    fusion_normalize: bool = True
    mean_fill: bool = True
    match_same_class: bool = True
