import io
import numpy as np
import pytest
from PIL import Image
from loadtest.validators import FrameCheckResult, validate_frame, check_prompt_sensitivity


def make_jpeg(w: int, h: int, color: tuple[int, int, int]) -> bytes:
    """Helper to create a JPEG with a solid color."""
    img = Image.new("RGB", (w, h), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_noisy_jpeg(w: int, h: int, seed: int = 0) -> bytes:
    """Helper to create a JPEG with random noise."""
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# --- validate_frame ---


def test_validate_frame_valid():
    assert validate_frame(make_jpeg(512, 512, (128, 64, 200)), 512, 512) == FrameCheckResult.VALID


def test_validate_frame_valid_noisy():
    assert validate_frame(make_noisy_jpeg(512, 512), 512, 512) == FrameCheckResult.VALID


def test_validate_frame_black():
    assert validate_frame(make_jpeg(512, 512, (0, 0, 0)), 512, 512) == FrameCheckResult.BLACK


def test_validate_frame_near_black():
    # Very dark but uniform — should be BLACK (low variance)
    assert validate_frame(make_jpeg(512, 512, (2, 2, 2)), 512, 512) == FrameCheckResult.BLACK


def test_validate_frame_wrong_size():
    assert validate_frame(make_jpeg(256, 256, (128, 64, 200)), 512, 512) == FrameCheckResult.WRONG_SIZE


def test_validate_frame_corrupt():
    assert validate_frame(b"not a jpeg at all", 512, 512) == FrameCheckResult.CORRUPT


def test_validate_frame_empty():
    assert validate_frame(b"", 512, 512) == FrameCheckResult.CORRUPT


def test_validate_frame_custom_variance():
    # Solid color has std=0, so any variance_min > 0 catches it
    assert validate_frame(make_jpeg(512, 512, (100, 100, 100)), 512, 512, variance_min=1.0) == FrameCheckResult.BLACK


# --- check_prompt_sensitivity ---


def test_prompt_sensitivity_very_different():
    a = make_jpeg(512, 512, (255, 0, 0))
    b = make_jpeg(512, 512, (0, 0, 255))
    assert check_prompt_sensitivity(a, b) is True


def test_prompt_sensitivity_identical():
    img = make_jpeg(512, 512, (128, 128, 128))
    assert check_prompt_sensitivity(img, img) is False


def test_prompt_sensitivity_similar():
    a = make_jpeg(512, 512, (100, 100, 100))
    b = make_jpeg(512, 512, (105, 105, 105))
    # Mean diff = 5, default min_diff = 10 → too similar
    assert check_prompt_sensitivity(a, b, min_diff=10.0) is False


def test_prompt_sensitivity_different_sizes():
    a = make_jpeg(512, 512, (100, 100, 100))
    b = make_jpeg(256, 256, (200, 200, 200))
    # Different dimensions → definitely different
    assert check_prompt_sensitivity(a, b) is True


def test_prompt_sensitivity_corrupt_input():
    assert check_prompt_sensitivity(b"not jpeg", make_jpeg(512, 512, (0, 0, 0))) is False


def test_prompt_sensitivity_noisy_frames():
    a = make_noisy_jpeg(512, 512, seed=0)
    b = make_noisy_jpeg(512, 512, seed=99)
    # Two different random images should be different
    assert check_prompt_sensitivity(a, b) is True
