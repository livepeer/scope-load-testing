import io
from pathlib import Path

from PIL import Image

from loadtest.datasets import (
    make_blocks_frame,
    make_gradient_frame,
    make_input_frame,
    make_noise_frame,
    select_prompts,
    select_video_style,
)


def _make_pool(tmp_path: Path, name: str, prompts: list[str]) -> None:
    d = tmp_path / "prompts"
    d.mkdir(exist_ok=True)
    lines = "\n".join(f"  - '{p}'" for p in prompts)
    (d / f"{name}.yaml").write_text(f"prompts:\n{lines}\n")


# --- Prompt selection ---


def test_select_prompts_picks_from_pools(tmp_path: Path):
    _make_pool(tmp_path, "nature", ["lake", "ocean", "forest"])
    _make_pool(tmp_path, "urban", ["city", "bridge", "station"])

    pool_name, prompts = select_prompts(["nature", "urban"], tmp_path / "prompts", seed=42)

    assert pool_name in ("nature", "urban")
    assert len(prompts) == 3


def test_select_prompts_shuffles(tmp_path: Path):
    _make_pool(tmp_path, "nature", ["a", "b", "c", "d", "e", "f", "g", "h"])

    orders = set()
    for seed in range(10):
        _, prompts = select_prompts(["nature"], tmp_path / "prompts", seed=seed)
        orders.add(tuple(prompts))

    # With 10 different seeds, we should get multiple orderings
    assert len(orders) > 1


def test_select_prompts_different_seeds_different_pools(tmp_path: Path):
    _make_pool(tmp_path, "nature", ["lake"])
    _make_pool(tmp_path, "urban", ["city"])
    _make_pool(tmp_path, "abstract", ["shapes"])

    pools_chosen = set()
    for seed in range(20):
        pool_name, _ = select_prompts(["nature", "urban", "abstract"], tmp_path / "prompts", seed=seed)
        pools_chosen.add(pool_name)

    # Over 20 seeds, all 3 pools should get picked at least once
    assert pools_chosen == {"nature", "urban", "abstract"}


def test_select_prompts_fallback_on_missing_pool(tmp_path: Path):
    pool_name, prompts = select_prompts(["nonexistent"], tmp_path / "prompts")
    assert pool_name == "nonexistent"
    assert len(prompts) > 0  # fallback prompts


# --- Video frame generation ---


def _assert_valid_jpeg(data: bytes, width: int, height: int):
    img = Image.open(io.BytesIO(data))
    assert img.size == (width, height)
    assert img.mode == "RGB"


def test_gradient_frame():
    data = make_gradient_frame(512, 512, 0)
    _assert_valid_jpeg(data, 512, 512)

    # Different frame numbers produce different frames
    data2 = make_gradient_frame(512, 512, 100)
    assert data != data2


def test_noise_frame():
    data = make_noise_frame(512, 512, 0)
    _assert_valid_jpeg(data, 512, 512)

    data2 = make_noise_frame(512, 512, 1)
    assert data != data2


def test_blocks_frame():
    data = make_blocks_frame(512, 512, 0)
    _assert_valid_jpeg(data, 512, 512)

    data2 = make_blocks_frame(512, 512, 10)
    assert data != data2


def test_make_input_frame_default():
    data = make_input_frame(512, 512, 0)
    _assert_valid_jpeg(data, 512, 512)


def test_make_input_frame_each_style():
    for style in ("gradient", "noise", "blocks"):
        data = make_input_frame(256, 256, 0, style=style)
        _assert_valid_jpeg(data, 256, 256)


def test_make_input_frame_unknown_style_falls_back():
    data = make_input_frame(512, 512, 0, style="unknown_style")
    _assert_valid_jpeg(data, 512, 512)


def test_select_video_style():
    style = select_video_style(seed=42)
    assert style in ("gradient", "noise", "blocks")


def test_select_video_style_custom_list():
    style = select_video_style(styles=["noise"], seed=0)
    assert style == "noise"


def test_select_video_style_varies():
    styles = set()
    for seed in range(20):
        styles.add(select_video_style(seed=seed))
    assert len(styles) > 1
