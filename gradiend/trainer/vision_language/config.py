"""Configuration for paired image–text continuation prediction."""

from dataclasses import dataclass
import math
from typing import Optional, Tuple


@dataclass(frozen=True)
class VisionLanguagePredictionConfig:
    """The class order corresponds to the manifest's ``a`` and ``b`` columns."""

    target_classes: Tuple[str, str]
    counterfactual_axis: str = "both"
    parameter_scope: str = "connector"
    max_length: int = 1024
    max_pixels: int = 256 * 28 * 28
    revision: Optional[str] = None
    base_model_dtype: Optional[str] = None
    min_neutral_likelihood_ratio: float = 0.99

    def __post_init__(self):
        classes = self.target_classes
        if (
            not isinstance(classes, (list, tuple))
            or len(classes) != 2
            or any(not isinstance(name, str) or not name.strip() for name in classes)
            or classes[0].strip() == classes[1].strip()
        ):
            raise ValueError("target_classes must contain two distinct non-empty class names")
        object.__setattr__(self, "target_classes", tuple(name.strip() for name in classes))
        if self.counterfactual_axis not in ("image", "text", "both"):
            raise ValueError("counterfactual_axis must be 'image', 'text', or 'both'")
        if self.parameter_scope not in ("language", "vision", "connector"):
            raise ValueError("parameter_scope must be 'language', 'vision', or 'connector'")
        if type(self.max_length) is not int or self.max_length < 2:
            raise ValueError("max_length must be an integer of at least 2")
        if type(self.max_pixels) is not int or self.max_pixels < 4 * 28 * 28:
            raise ValueError("max_pixels must be an integer of at least 3136 for Qwen2.5-VL")
        if self.revision is not None and (not isinstance(self.revision, str) or not self.revision.strip()):
            raise ValueError("revision must be a non-empty revision string when provided")
        if self.base_model_dtype not in (None, "bfloat16", "float32"):
            raise ValueError("base_model_dtype must be 'bfloat16' or 'float32' when provided")
        ratio = self.min_neutral_likelihood_ratio
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or not 0 < ratio <= 1:
            raise ValueError("min_neutral_likelihood_ratio must be in (0, 1]")
