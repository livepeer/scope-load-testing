"""Generate reference video clips for v2v testing via SDK LV2V streams.

Starts longlive t2v streams, captures output frames for 10s, saves as
JPEG frame sequences in config/datasets/clips/{id}/.

Usage:
    export DAYDREAM_API_KEY=sk_...
    python scripts/generate_video_clips.py
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

import httpx
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gen-clips")

SDK_URL = os.environ.get("SDK_URL", "https://sdk.daydream.monster")
CLIPS_DIR = Path("config/datasets/clips")
MANIFEST_PATH = Path("config/datasets/manifest.yaml")
CAPTURE_DURATION_S = 10
CAPTURE_INTERVAL_S = 0.5

CLIP_SPECS = [
    {"id": "nature_lake", "prompt": "a calm mountain lake with gentle ripples and birds flying", "tags": ["nature", "water", "calm"]},
    {"id": "nature_waves", "prompt": "ocean waves rolling onto a sandy beach at sunset", "tags": ["nature", "water", "motion"]},
    {"id": "nature_forest", "prompt": "wind blowing through a forest with leaves falling", "tags": ["nature", "forest", "wind"]},
    {"id": "urban_traffic", "prompt": "busy city intersection at night with car headlights", "tags": ["urban", "traffic", "night"]},
    {"id": "urban_rain", "prompt": "rain falling on a city street with reflections and umbrellas", "tags": ["urban", "rain", "reflections"]},
    {"id": "urban_train", "prompt": "a subway train arriving at a platform with commuters", "tags": ["urban", "train", "motion"]},
    {"id": "abstract_flow", "prompt": "colorful liquid flowing and swirling in slow motion", "tags": ["abstract", "liquid", "slow"]},
    {"id": "abstract_particles", "prompt": "glowing particles drifting and colliding in dark space", "tags": ["abstract", "particles", "glow"]},
    {"id": "abstract_fractal", "prompt": "a fractal pattern zooming deeper with shifting colors", "tags": ["abstract", "fractal", "zoom"]},
    {"id": "motion_fire", "prompt": "a campfire with flames dancing and sparks rising", "tags": ["motion", "fire", "particles"]},
    {"id": "motion_water", "prompt": "a rushing river with white water rapids over rocks", "tags": ["motion", "water", "fast"]},
    {"id": "motion_wind", "prompt": "a field of tall grass swaying in strong wind under stormy sky", "tags": ["motion", "wind", "nature"]},
]


async def capture_clip(api_key: str, spec: dict) -> list[bytes]:
    """Start a t2v stream, capture frames, return frame list."""
    headers = {"Authorization": f"Bearer {api_key}"}
    timeout = httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0)

    async with httpx.AsyncClient(base_url=SDK_URL, headers=headers, timeout=timeout) as client:
        resp = await client.post("/stream/start", json={
            "model_id": "scope",
            "params": {"prompt": spec["prompt"], "pipeline_ids": ["longlive"]},
        })
        resp.raise_for_status()
        stream_id = resp.json()["stream_id"]

        try:
            # Wait for runner — fail fast on error phases
            for _ in range(120):
                status = await client.get(f"/stream/{stream_id}/status")
                if status.status_code == 404:
                    raise RuntimeError("Stream disappeared")
                phase = status.json().get("phase", "unknown")
                if phase in ("ready", "running", "connecting"):
                    break
                if phase in ("error", "failed"):
                    raise RuntimeError(f"Stream entered terminal phase: {phase}")
                await asyncio.sleep(5)
            else:
                raise TimeoutError("Runner not ready")

            # Wait for first frame and include it in results
            frames = []
            for _ in range(60):
                frame_resp = await client.get(f"/stream/{stream_id}/frame")
                if frame_resp.status_code == 200 and len(frame_resp.content) > 100:
                    frames.append(frame_resp.content)
                    break
                await asyncio.sleep(1)
            else:
                raise TimeoutError("No first frame")

            # Capture remaining frames
            start = time.monotonic()
            while time.monotonic() - start < CAPTURE_DURATION_S:
                frame_resp = await client.get(f"/stream/{stream_id}/frame")
                if frame_resp.status_code == 200 and len(frame_resp.content) > 100:
                    frames.append(frame_resp.content)
                await asyncio.sleep(CAPTURE_INTERVAL_S)

            return frames

        finally:
            try:
                await client.post(f"/stream/{stream_id}/stop")
            except Exception:
                pass


async def main():
    api_key = os.environ.get("DAYDREAM_API_KEY")
    if not api_key:
        print("Set DAYDREAM_API_KEY")
        sys.exit(1)

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            manifest = yaml.safe_load(f) or {}
    else:
        manifest = {}

    existing_ids = {item["id"] for item in manifest.get("clips", {}).get("items", [])}
    generated = []
    failed = []

    for spec in CLIP_SPECS:
        if spec["id"] in existing_ids:
            logger.info("SKIP %s (exists)", spec["id"])
            continue

        logger.info("Capturing %s: %s", spec["id"], spec["prompt"][:50])
        try:
            frames = await capture_clip(api_key, spec)
            if not frames:
                logger.warning("  No frames captured for %s", spec["id"])
                failed.append(spec["id"])
                continue

            clip_dir = CLIPS_DIR / spec["id"]
            clip_dir.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(frames):
                (clip_dir / f"frame_{i:04d}.jpg").write_bytes(frame)

            generated.append({
                "id": spec["id"],
                "file": str(clip_dir),
                "prompt": spec["prompt"],
                "tags": spec["tags"],
                "source": "sdk_capture",
                "resolution": "512x512",
                "format": "jpeg_sequence",
                "frame_count": len(frames),
                "duration_s": CAPTURE_DURATION_S,
            })
            logger.info("  OK: %d frames saved to %s", len(frames), clip_dir)

        except Exception as e:
            logger.error("  FAIL %s: %s", spec["id"], e)
            failed.append(spec["id"])

    if generated:
        clips_items = manifest.get("clips", {}).get("items", [])
        clips_items.extend(generated)
        if "clips" not in manifest:
            manifest["clips"] = {}
        manifest["clips"]["items"] = clips_items
        with open(MANIFEST_PATH, "w") as f:
            yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
        logger.info("Generated %d clips, %d failed", len(generated), len(failed))
    else:
        logger.info("No new clips generated")

    if failed:
        logger.warning("Failed: %s", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
