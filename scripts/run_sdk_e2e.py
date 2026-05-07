"""Run load test against Scope via the Daydream SDK service.

This uses the same path as storyboard and real users:
  Browser → SDK Service → Livepeer Orchestrator → Scope Runner

Usage:
    python scripts/run_sdk_e2e.py --api-key sk_... --durations 1,5,15
    python scripts/run_sdk_e2e.py --api-key sk_... --durations 1  # short only
"""

import asyncio
import logging
import os
import sys
import time

import click
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("e2e")

DEFAULT_SDK_URL = "https://sdk.daydream.monster"

PROMPTS = [
    "a serene mountain lake at sunrise with mist rising from the water",
    "ocean waves crashing on a rocky coastline at golden hour",
    "a dense forest with sunlight filtering through tall trees",
    "a vast desert with rolling sand dunes under a clear blue sky",
]


async def run_session(
    sdk_url: str, api_key: str, duration_label: str, duration_s: int
) -> dict:
    """Run a single longlive t2v session via the SDK and return results."""
    result = {
        "duration_label": duration_label,
        "duration_s": duration_s,
        "passed": False,
        "connect_time_s": None,
        "first_frame_s": None,
        "fps_samples": [],
        "frames_captured": 0,
        "error": None,
        "stream_id": None,
    }

    headers = {"Authorization": f"Bearer {api_key}"}
    timeout = httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0)

    async with httpx.AsyncClient(base_url=sdk_url, headers=headers, timeout=timeout) as client:
        try:
            # 1. Start stream
            logger.info("[%s] Starting stream (longlive t2v, %ds)...", duration_label, duration_s)
            connect_start = time.monotonic()

            resp = await client.post("/stream/start", json={
                "model_id": "scope",
                "params": {
                    "prompt": PROMPTS[0],
                    "pipeline_ids": ["longlive"],
                },
            })
            resp.raise_for_status()
            stream_data = resp.json()
            stream_id = stream_data["stream_id"]
            result["stream_id"] = stream_id
            logger.info("[%s] Stream created: %s", duration_label, stream_id)

            # 2. Wait for runner to be ready
            for attempt in range(120):  # 10 min max
                status_resp = await client.get(f"/stream/{stream_id}/status")
                if status_resp.status_code == 404:
                    raise RuntimeError("Stream disappeared while waiting for runner")
                phase = status_resp.json().get("phase", "unknown")
                if phase in ("ready", "running", "connecting"):
                    result["connect_time_s"] = time.monotonic() - connect_start
                    logger.info("[%s] Runner ready in %.1fs (phase=%s)", duration_label, result["connect_time_s"], phase)
                    break
                if phase in ("error", "failed"):
                    raise RuntimeError(f"Stream failed: {status_resp.json()}")
                await asyncio.sleep(5)
            else:
                raise TimeoutError("Runner not ready after 10 min")

            # 3. Wait for first frame
            logger.info("[%s] Waiting for first frame...", duration_label)
            ff_start = time.monotonic()
            for attempt in range(120):  # 2 min
                frame_resp = await client.get(f"/stream/{stream_id}/frame")
                if frame_resp.status_code == 200 and len(frame_resp.content) > 100:
                    result["first_frame_s"] = time.monotonic() - ff_start
                    result["frames_captured"] += 1
                    logger.info("[%s] First frame in %.1fs (%d bytes)", duration_label, result["first_frame_s"], len(frame_resp.content))
                    break
                await asyncio.sleep(1)
            else:
                raise TimeoutError("No frame after 2 min")

            # 4. Monitoring loop
            logger.info("[%s] Streaming for %ds...", duration_label, duration_s)
            stream_start = time.monotonic()
            prompt_idx = 0
            last_prompt_switch = stream_start

            while time.monotonic() - stream_start < duration_s:
                await asyncio.sleep(10)
                elapsed = time.monotonic() - stream_start

                # Capture frame
                try:
                    frame_resp = await client.get(f"/stream/{stream_id}/frame")
                    if frame_resp.status_code == 200 and len(frame_resp.content) > 100:
                        result["frames_captured"] += 1
                except Exception:
                    pass

                # Check status
                try:
                    status_resp = await client.get(f"/stream/{stream_id}/status")
                    if status_resp.status_code == 404:
                        raise RuntimeError("Stream disappeared mid-session")
                    phase = status_resp.json().get("phase", "unknown")
                    logger.info("[%s] %.0fs: phase=%s frames=%d", duration_label, elapsed, phase, result["frames_captured"])
                except httpx.HTTPStatusError:
                    pass

                # Prompt switch every 30s
                if time.monotonic() - last_prompt_switch > 30:
                    prompt_idx = (prompt_idx + 1) % len(PROMPTS)
                    try:
                        await client.post(f"/stream/{stream_id}/control", json={
                            "prompt": PROMPTS[prompt_idx],
                        })
                        logger.info("[%s] Switched prompt: %s", duration_label, PROMPTS[prompt_idx][:40])
                    except Exception as e:
                        logger.warning("[%s] Prompt switch failed: %s", duration_label, e)
                    last_prompt_switch = time.monotonic()

            result["passed"] = True

        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
            logger.error("[%s] Failed: %s", duration_label, result["error"])

        finally:
            # Stop stream
            if result["stream_id"]:
                try:
                    await client.post(f"/stream/{result['stream_id']}/stop")
                    logger.info("[%s] Stream stopped", duration_label)
                except Exception:
                    pass

    return result


async def main(sdk_url: str, api_key: str, durations: list[tuple[str, int]]):
    logger.info("=" * 60)
    logger.info("Scope Load Test via SDK")
    logger.info("SDK: %s", sdk_url)
    logger.info("Durations: %s", [f"{l}={s}s" for l, s in durations])
    logger.info("=" * 60)

    results = []
    for label, secs in durations:
        logger.info("")
        logger.info("--- Starting %s run (%ds) ---", label, secs)
        r = await run_session(sdk_url, api_key, label, secs)
        results.append(r)

        status = "PASS" if r["passed"] else f"FAIL: {r['error']}"
        logger.info(
            "--- %s: %s (connect=%.1fs ff=%.1fs frames=%d) ---",
            label, status,
            r["connect_time_s"] or 0,
            r["first_frame_s"] or 0,
            r["frames_captured"],
        )

        if label != durations[-1][0]:
            logger.info("Waiting 10s before next run...")
            await asyncio.sleep(10)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        logger.info(
            "  %-6s %s  connect=%.1fs  first_frame=%.1fs  frames=%d  %s",
            r["duration_label"], status,
            r["connect_time_s"] or 0,
            r["first_frame_s"] or 0,
            r["frames_captured"],
            r["error"] or "",
        )

    passed = sum(1 for r in results if r["passed"])
    logger.info("")
    logger.info("Result: %d/%d passed", passed, len(results))
    return 0 if passed == len(results) else 1


DURATION_MAP = {"1": ("1m", 60), "5": ("5m", 300), "15": ("15m", 900)}


@click.command()
@click.option("--sdk-url", default=DEFAULT_SDK_URL, help="SDK service URL")
@click.option("--api-key", envvar="DAYDREAM_API_KEY", required=True, help="Daydream API key")
@click.option("--durations", default="1,5,15", help="Comma-separated durations in minutes (e.g., 1,5,15)")
def cli(sdk_url: str, api_key: str, durations: str):
    """Run longlive t2v load test via Daydream SDK."""
    dur_list = []
    for d in durations.split(","):
        d = d.strip()
        if d in DURATION_MAP:
            dur_list.append(DURATION_MAP[d])
        else:
            click.echo(f"Unknown duration: {d}. Use 1, 5, or 15.")
            sys.exit(1)

    code = asyncio.run(main(sdk_url, api_key, dur_list))
    sys.exit(code)


if __name__ == "__main__":
    cli()
