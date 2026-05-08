"""Test dataset management — prompt selection and video input generation.

Handles:
- Random prompt pool rotation per run (ensures all pools get coverage)
- Randomized prompt ordering within a pool (prevents same sequence every time)
- Synthetic video frame generation with varying visual complexity
- Reference image/clip selection from manifest for i2v/v2v
- Dataset coverage tracking across runs
"""

import io
import logging
import random
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from .scenarios import load_prompt_pool

logger = logging.getLogger(__name__)

MANIFEST_PATH = Path("config/datasets/manifest.yaml")


# ---------------------------------------------------------------------------
# Prompt selection
# ---------------------------------------------------------------------------


def select_prompts(
    pools: list[str],
    prompts_dir: Path,
    seed: int | None = None,
) -> tuple[str, list[str]]:
    """Pick a random prompt pool and shuffle its prompts.

    Returns (pool_name, shuffled_prompts). The pool is chosen randomly
    so that over many runs, all pools get exercised.
    """
    rng = random.Random(seed)
    pool_name = rng.choice(pools) if pools else "nature"

    try:
        prompts = load_prompt_pool(pool_name, prompts_dir)
    except FileNotFoundError:
        logger.warning("Prompt pool %s not found, using fallback", pool_name)
        prompts = ["a scenic landscape", "a bustling city street", "abstract flowing shapes"]

    rng.shuffle(prompts)
    return pool_name, prompts


# ---------------------------------------------------------------------------
# Synthetic video input frames
# ---------------------------------------------------------------------------

# Three visual styles that stress the pipeline differently:
#
# gradient  — smooth color transitions (tests temporal consistency)
# noise     — random texture (tests detail preservation under noise_scale)
# blocks    — hard edges and flat regions (tests edge handling)


def make_gradient_frame(width: int, height: int, frame_num: int) -> bytes:
    """Smooth hue-shifting gradient — tests temporal consistency."""
    t = (frame_num % 300) / 300.0
    r = int(128 + 127 * np.sin(t * 2 * np.pi))
    g = int(128 + 127 * np.sin(t * 2 * np.pi + 2.094))
    b = int(128 + 127 * np.sin(t * 2 * np.pi + 4.189))
    img = Image.new("RGB", (width, height), (r, g, b))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return buf.getvalue()


def make_noise_frame(width: int, height: int, frame_num: int) -> bytes:
    """Random noise per frame — tests detail preservation."""
    rng = np.random.RandomState(frame_num)
    arr = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return buf.getvalue()


def make_blocks_frame(width: int, height: int, frame_num: int) -> bytes:
    """Moving color blocks — tests edge handling and motion."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    block_size = 64
    offset = (frame_num * 4) % block_size  # slow drift

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            bx = (x + offset) // block_size
            by = (y + offset) // block_size
            r = (bx * 73 + frame_num * 3) % 256
            g = (by * 127 + frame_num * 7) % 256
            b = ((bx + by) * 47 + frame_num * 11) % 256
            y_end = min(y + block_size, height)
            x_end = min(x + block_size, width)
            arr[y:y_end, x:x_end] = (r, g, b)

    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return buf.getvalue()


FRAME_GENERATORS = {
    "gradient": make_gradient_frame,
    "noise": make_noise_frame,
    "blocks": make_blocks_frame,
}


def make_input_frame(
    width: int,
    height: int,
    frame_num: int,
    style: str = "gradient",
) -> bytes:
    """Generate a synthetic input frame in the given style."""
    generator = FRAME_GENERATORS.get(style, make_gradient_frame)
    return generator(width, height, frame_num)


def select_video_style(
    styles: list[str] | None = None,
    seed: int | None = None,
) -> str:
    """Pick a random video input style for this run."""
    choices = styles or list(FRAME_GENERATORS.keys())
    rng = random.Random(seed)
    return rng.choice(choices)


# ---------------------------------------------------------------------------
# Manifest-based asset selection (images and clips)
# ---------------------------------------------------------------------------


def load_manifest(manifest_path: Path | None = None) -> dict:
    """Load the dataset manifest."""
    path = manifest_path or MANIFEST_PATH
    if not path.exists():
        return {"prompts": {"pools": []}, "images": {"items": []}, "clips": {"items": []}}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def select_reference_image(
    tags: list[str] | None = None,
    manifest_path: Path | None = None,
    seed: int | None = None,
) -> dict | None:
    """Pick a reference image from the manifest, optionally filtered by tags.

    Returns the manifest entry dict, or None if no images available.
    """
    manifest = load_manifest(manifest_path)
    items = manifest.get("images", {}).get("items", [])
    if not items:
        return None

    if tags:
        tag_set = set(tags)
        items = [i for i in items if tag_set & set(i.get("tags", []))]

    if not items:
        return None

    rng = random.Random(seed)
    return rng.choice(items)


def select_video_clip(
    tags: list[str] | None = None,
    manifest_path: Path | None = None,
    seed: int | None = None,
) -> dict | None:
    """Pick a video clip from the manifest, optionally filtered by tags.

    Returns the manifest entry dict, or None if no clips available.
    """
    manifest = load_manifest(manifest_path)
    items = manifest.get("clips", {}).get("items", [])
    if not items:
        return None

    if tags:
        tag_set = set(tags)
        items = [i for i in items if tag_set & set(i.get("tags", []))]

    if not items:
        return None

    rng = random.Random(seed)
    return rng.choice(items)


def get_dataset_summary(manifest_path: Path | None = None) -> dict:
    """Get a summary of all available test datasets."""
    manifest = load_manifest(manifest_path)
    prompts_dir = Path("config/prompts")

    prompt_count = 0
    pools = []
    for f in sorted(prompts_dir.glob("*.yaml")):
        with open(f) as fh:
            data = yaml.safe_load(fh)
        count = len(data.get("prompts", []))
        prompt_count += count
        pools.append({"name": f.stem, "count": count})

    return {
        "prompt_pools": pools,
        "total_prompts": prompt_count,
        "reference_images": len(manifest.get("images", {}).get("items", [])),
        "video_clips": len(manifest.get("clips", {}).get("items", [])),
    }
