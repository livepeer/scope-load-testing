"""Frame validation and prompt sensitivity using Pillow + numpy only."""

import io
from enum import Enum

import numpy as np
from PIL import Image


class FrameCheckResult(str, Enum):
    VALID = "valid"
    BLACK = "black"
    CORRUPT = "corrupt"
    WRONG_SIZE = "wrong_size"


def validate_frame(
    jpeg_bytes: bytes,
    expected_w: int,
    expected_h: int,
    variance_min: float = 5.0,
) -> FrameCheckResult:
    """Validate a JPEG frame for quality issues."""
    try:
        img = Image.open(io.BytesIO(jpeg_bytes))
        img.load()
    except Exception:
        return FrameCheckResult.CORRUPT

    if img.size != (expected_w, expected_h):
        return FrameCheckResult.WRONG_SIZE

    if np.array(img, dtype=np.float32).std() < variance_min:
        return FrameCheckResult.BLACK

    return FrameCheckResult.VALID


def check_prompt_sensitivity(
    frame_before: bytes,
    frame_after: bytes,
    min_diff: float = 10.0,
) -> bool:
    """Check that two frames are sufficiently different.

    Returns True if the mean absolute pixel difference >= min_diff,
    meaning the model responded to the prompt change.
    """
    try:
        a = np.array(Image.open(io.BytesIO(frame_before)).convert("RGB"), dtype=np.float32)
        b = np.array(Image.open(io.BytesIO(frame_after)).convert("RGB"), dtype=np.float32)
    except Exception:
        return False

    if a.shape != b.shape:
        return True  # different dimensions = definitely different

    mean_diff = float(np.abs(a - b).mean())
    return mean_diff >= min_diff
