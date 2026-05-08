"""Check test datasets for duplicates, gaps, and quality.

Usage: python scripts/check_datasets.py
"""

from pathlib import Path

import yaml

PROMPTS_DIR = Path("config/prompts")
MANIFEST_PATH = Path("config/datasets/manifest.yaml")
TARGET_POOL_SIZE = 20


def check_prompts():
    """Check prompt pools for duplicates and coverage."""
    seen: dict[str, str] = {}
    pools: dict[str, list[str]] = {}
    dupes = 0

    for f in sorted(PROMPTS_DIR.glob("*.yaml")):
        with open(f) as fh:
            data = yaml.safe_load(fh)
        prompts = data.get("prompts", [])
        pools[f.stem] = prompts

        for p in prompts:
            key = p.strip().lower()
            if key in seen:
                print(f"  DUPLICATE: \"{p[:60]}\" in {f.stem} and {seen[key]}")
                dupes += 1
            seen[key] = f.stem

    print(f"\nPrompt pools ({len(pools)}):")
    for name, prompts in sorted(pools.items()):
        status = "OK" if len(prompts) >= TARGET_POOL_SIZE else f"LOW ({TARGET_POOL_SIZE - len(prompts)} needed)"
        print(f"  {name:15s} {len(prompts):3d} prompts  {status}")

    print(f"\nTotal: {len(seen)} unique prompts, {dupes} duplicates")
    return dupes


def check_manifest():
    """Check manifest for completeness."""
    if not MANIFEST_PATH.exists():
        print("\nManifest: NOT FOUND")
        return

    with open(MANIFEST_PATH) as f:
        manifest = yaml.safe_load(f)

    images = manifest.get("images", {}).get("items", [])
    clips = manifest.get("clips", {}).get("items", [])
    pools = manifest.get("prompts", {}).get("pools", [])

    print(f"\nManifest:")
    print(f"  Prompt pools registered: {len(pools)}")
    print(f"  Reference images: {len(images)}")
    print(f"  Video clips: {len(clips)}")

    # Check that registered pools match actual files
    registered = {p["name"] for p in pools}
    actual = {f.stem for f in PROMPTS_DIR.glob("*.yaml")}
    unregistered = actual - registered
    if unregistered:
        print(f"  WARNING: pools not in manifest: {unregistered}")

    missing_files = registered - actual
    if missing_files:
        print(f"  WARNING: manifest references missing pools: {missing_files}")


def suggest_gaps():
    """Suggest categories that could improve coverage."""
    with open(PROMPTS_DIR.parent / "prompts" / "nature.yaml") as f:
        nature = yaml.safe_load(f).get("prompts", [])

    existing_themes = set()
    for f in PROMPTS_DIR.glob("*.yaml"):
        existing_themes.add(f.stem)

    suggestions = [
        ("people", "faces, crowds, portraits, expressions"),
        ("animals", "wildlife, pets, underwater creatures"),
        ("weather", "storms, rain, snow, fog, lightning"),
        ("scifi", "space, robots, futuristic cities, alien landscapes"),
        ("fantasy", "dragons, castles, magic, mythical creatures"),
        ("food", "cooking, ingredients, restaurants, close-ups"),
        ("motion", "sports, dance, vehicles, flowing water (high-motion stress test)"),
    ]

    missing = [(name, desc) for name, desc in suggestions if name not in existing_themes]
    if missing:
        print(f"\nSuggested new pools:")
        for name, desc in missing:
            print(f"  {name:15s} — {desc}")


if __name__ == "__main__":
    print("=== Dataset Quality Check ===\n")
    dupes = check_prompts()
    check_manifest()
    suggest_gaps()
    print()
    exit(1 if dupes > 0 else 0)
