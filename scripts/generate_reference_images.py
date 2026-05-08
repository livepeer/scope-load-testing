"""Generate reference images for i2v testing via the storyboard MCP API.

Uses the storyboard create_media endpoint to generate 512x512 images
from prompts across multiple categories, then downloads them locally.

Usage:
    export DAYDREAM_API_KEY=sk_...
    python scripts/generate_reference_images.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import httpx
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gen-images")

SDK_URL = "https://sdk.daydream.monster"
IMAGES_DIR = Path("config/datasets/images")
MANIFEST_PATH = Path("config/datasets/manifest.yaml")

# One representative prompt per category for reference images
IMAGE_SPECS = [
    # Nature
    {"id": "nature_001", "prompt": "a serene mountain lake at sunrise with mist", "tags": ["nature", "water", "landscape"]},
    {"id": "nature_002", "prompt": "ocean waves crashing on a rocky coastline at golden hour", "tags": ["nature", "water", "coast"]},
    {"id": "nature_003", "prompt": "a dense forest with sunlight filtering through tall trees", "tags": ["nature", "forest", "light"]},
    {"id": "nature_004", "prompt": "a vast desert with rolling sand dunes under blue sky", "tags": ["nature", "desert", "minimal"]},
    {"id": "nature_005", "prompt": "a waterfall cascading into a tropical pool surrounded by ferns", "tags": ["nature", "water", "tropical"]},
    # Urban
    {"id": "urban_001", "prompt": "a bustling city street at night with neon signs reflecting on wet pavement", "tags": ["urban", "night", "neon"]},
    {"id": "urban_002", "prompt": "an aerial view of a modern skyline at sunset", "tags": ["urban", "skyline", "sunset"]},
    {"id": "urban_003", "prompt": "a quiet alley in an old European town with cobblestone streets", "tags": ["urban", "historic", "alley"]},
    {"id": "urban_004", "prompt": "a rooftop garden overlooking a dense urban landscape", "tags": ["urban", "garden", "rooftop"]},
    {"id": "urban_005", "prompt": "a bridge over a river at twilight with city lights reflecting", "tags": ["urban", "bridge", "twilight"]},
    # Abstract
    {"id": "abstract_001", "prompt": "swirling patterns of light and color in a cosmic void", "tags": ["abstract", "cosmic", "pattern"]},
    {"id": "abstract_002", "prompt": "geometric shapes dissolving and reforming in slow motion", "tags": ["abstract", "geometric", "transform"]},
    {"id": "abstract_003", "prompt": "liquid metal flowing and splitting into fractal patterns", "tags": ["abstract", "metal", "fractal"]},
    {"id": "abstract_004", "prompt": "ink drops expanding in water creating organic formations", "tags": ["abstract", "ink", "organic"]},
    {"id": "abstract_005", "prompt": "aurora-like ribbons of light flowing through darkness", "tags": ["abstract", "aurora", "light"]},
    # People
    {"id": "people_001", "prompt": "a dancer mid-leap in dramatic stage lighting", "tags": ["people", "dance", "motion"]},
    {"id": "people_002", "prompt": "a crowded marketplace with vendors and shoppers in warm light", "tags": ["people", "crowd", "market"]},
    {"id": "people_003", "prompt": "a musician playing saxophone on a rainy street corner", "tags": ["people", "music", "rain"]},
    {"id": "people_004", "prompt": "a portrait of an elderly person with weathered hands and kind eyes", "tags": ["people", "portrait", "close-up"]},
    {"id": "people_005", "prompt": "children playing in a sunlit park with autumn leaves", "tags": ["people", "children", "park"]},
]


def generate_image_via_sdk(api_key: str, prompt: str, retries: int = 3) -> str | None:
    """Generate an image via the SDK inference endpoint. Returns the image URL or None."""
    # Try multiple capabilities in order of preference
    capabilities = ["fal-ai/flux/schnell", "fal-ai/flux/dev"]

    for cap in capabilities:
        for attempt in range(retries):
            try:
                with httpx.Client(timeout=httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0)) as client:
                    resp = client.post(
                        f"{SDK_URL}/inference",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "capability": cap,
                            "prompt": prompt,
                            "width": 512,
                            "height": 512,
                        },
                    )
                    if resp.status_code == 503:
                        logger.warning("  No capacity for %s (attempt %d/%d)", cap, attempt + 1, retries)
                        time.sleep(5)
                        continue
                    resp.raise_for_status()
                    data = resp.json()

                    # Extract image URL from response
                    images = data.get("images", [])
                    if images and isinstance(images[0], dict):
                        return images[0].get("url")
                    elif images and isinstance(images[0], str):
                        return images[0]

                    # Try other common response shapes
                    if "image" in data and isinstance(data["image"], dict):
                        return data["image"].get("url")
                    if "url" in data:
                        return data["url"]

                    logger.warning("  Unexpected response shape for %s: %s", cap, list(data.keys()))
                    return None

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 503:
                    logger.warning("  No capacity for %s (attempt %d/%d)", cap, attempt + 1, retries)
                    time.sleep(5)
                    continue
                raise
    return None


def download_image(url: str, dest: Path) -> bool:
    """Download an image from a URL to a local file."""
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)
            return True
    except Exception as e:
        logger.error("Failed to download %s: %s", url, e)
        return False


def main():
    api_key = os.environ.get("DAYDREAM_API_KEY")
    if not api_key:
        print("Set DAYDREAM_API_KEY environment variable")
        sys.exit(1)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing manifest to check for duplicates
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            manifest = yaml.safe_load(f) or {}
    else:
        manifest = {}

    existing_ids = {item["id"] for item in manifest.get("images", {}).get("items", [])}
    generated = []
    failed = []

    for spec in IMAGE_SPECS:
        if spec["id"] in existing_ids:
            logger.info("SKIP %s (already exists)", spec["id"])
            continue

        logger.info("Generating %s: %s", spec["id"], spec["prompt"][:50])

        try:
            image_url = generate_image_via_sdk(api_key, spec["prompt"])

            if not image_url:
                logger.warning("No image URL returned for %s", spec["id"])
                failed.append(spec["id"])
                continue

            # Download
            dest = IMAGES_DIR / f"{spec['id']}.jpg"
            if download_image(image_url, dest):
                generated.append({
                    "id": spec["id"],
                    "file": str(dest),
                    "prompt": spec["prompt"],
                    "tags": spec["tags"],
                    "source": "storyboard_mcp",
                    "resolution": "512x512",
                    "url": image_url,
                })
                logger.info("  OK: %s (%d bytes)", dest.name, dest.stat().st_size)
            else:
                failed.append(spec["id"])

            # Rate limit
            time.sleep(1)

        except Exception as e:
            logger.error("  FAIL %s: %s", spec["id"], e)
            failed.append(spec["id"])

    # Update manifest
    if generated:
        images_items = manifest.get("images", {}).get("items", [])
        images_items.extend(generated)
        if "images" not in manifest:
            manifest["images"] = {}
        manifest["images"]["items"] = images_items

        with open(MANIFEST_PATH, "w") as f:
            yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

        logger.info("")
        logger.info("Generated %d images, %d failed", len(generated), len(failed))
        logger.info("Manifest updated: %s", MANIFEST_PATH)
    else:
        logger.info("No new images generated")

    if failed:
        logger.warning("Failed: %s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
