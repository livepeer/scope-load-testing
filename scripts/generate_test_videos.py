"""Generate test input videos for load testing scenarios.

Requires opencv-python: pip install opencv-python
"""

from pathlib import Path

import cv2
import numpy as np


def create_solid_video(
    path: str,
    color: tuple[int, int, int],
    width: int = 512,
    height: int = 512,
    fps: int = 30,
    duration_s: int = 30,
) -> None:
    """Create a video with a solid color (BGR)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = color
    for _ in range(fps * duration_s):
        writer.write(frame)
    writer.release()
    print(f"  {path} ({width}x{height}, {fps}fps, {duration_s}s)")


def create_gradient_video(
    path: str,
    width: int = 512,
    height: int = 512,
    fps: int = 30,
    duration_s: int = 30,
) -> None:
    """Create a video with colors that shift over time."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    total_frames = fps * duration_s
    for i in range(total_frames):
        t = i / total_frames
        r = int(255 * t)
        g = int(255 * (1 - t))
        b = 128
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (b, g, r)
        writer.write(frame)
    writer.release()
    print(f"  {path} ({width}x{height}, {fps}fps, {duration_s}s, gradient)")


def create_scene_change_video(
    path: str,
    width: int = 512,
    height: int = 512,
    fps: int = 30,
    duration_s: int = 60,
) -> None:
    """Create a video that switches color at the midpoint."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    total_frames = fps * duration_s
    mid = total_frames // 2
    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        if i < mid:
            frame[:] = (0, 0, 255)  # red (BGR)
        else:
            frame[:] = (255, 0, 0)  # blue (BGR)
        writer.write(frame)
    writer.release()
    print(f"  {path} ({width}x{height}, {fps}fps, {duration_s}s, scene change)")


if __name__ == "__main__":
    videos_dir = "videos"
    print("Generating test videos:")
    create_solid_video(f"{videos_dir}/solid_red_512x512_30s.mp4", (0, 0, 255))
    create_solid_video(f"{videos_dir}/solid_green_512x512_30s.mp4", (0, 255, 0))
    create_gradient_video(f"{videos_dir}/gradient_512x512_30s.mp4")
    create_scene_change_video(f"{videos_dir}/scene_change_512x512_60s.mp4")
    print(f"Done. Videos in: {videos_dir}/")
