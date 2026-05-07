"""Run longlive v2v load test via SDK — publishes video frames to keep stream alive.

Usage:
    python scripts/run_sdk_v2v_e2e.py --api-key sk_... --durations 1,5,15
"""

import asyncio
import io
import logging
import sys
import time

import click
import httpx
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("e2e-v2v")

DEFAULT_SDK_URL = "https://sdk.daydream.monster"

PROMPTS = [
    "a serene mountain lake at sunrise with mist rising from the water",
    "ocean waves crashing on a rocky coastline at golden hour",
    "a dense forest with sunlight filtering through tall trees",
    "a vast desert with rolling sand dunes under a clear blue sky",
]

V2V_GRAPH = {
    "nodes": [
        {"id": "input", "type": "source", "source_mode": "video"},
        {"id": "longlive", "type": "pipeline", "pipeline_id": "longlive"},
        {"id": "output", "type": "sink"},
    ],
    "edges": [
        {"from": "input", "from_port": "video", "to_node": "longlive", "to_port": "video", "kind": "stream"},
        {"from": "longlive", "from_port": "video", "to_node": "output", "to_port": "video", "kind": "stream"},
    ],
}


def make_test_frame(width: int = 512, height: int = 512, frame_num: int = 0) -> bytes:
    """Generate a synthetic JPEG frame with varying color."""
    t = (frame_num % 300) / 300.0
    r = int(128 + 127 * np.sin(t * 2 * np.pi))
    g = int(128 + 127 * np.sin(t * 2 * np.pi + 2.094))
    b = int(128 + 127 * np.sin(t * 2 * np.pi + 4.189))
    img = Image.new("RGB", (width, height), (r, g, b))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return buf.getvalue()


async def run_session(
    sdk_url: str, api_key: str, duration_label: str, duration_s: int
) -> dict:
    result = {
        "duration_label": duration_label,
        "duration_s": duration_s,
        "passed": False,
        "connect_time_s": None,
        "first_frame_s": None,
        "frames_published": 0,
        "frames_captured": 0,
        "error": None,
        "stream_id": None,
    }

    headers = {"Authorization": f"Bearer {api_key}"}
    timeout = httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0)

    publish_stop = asyncio.Event()
    publish_task = None

    async with httpx.AsyncClient(base_url=sdk_url, headers=headers, timeout=timeout) as client:
        try:
            # 1. Start stream with v2v graph
            logger.info("[%s] Starting v2v stream (%ds)...", duration_label, duration_s)
            connect_start = time.monotonic()

            resp = await client.post("/stream/start", json={
                "model_id": "scope",
                "params": {
                    "prompt": PROMPTS[0],
                    "pipeline_ids": ["longlive"],
                    "graph": V2V_GRAPH,
                    "noise_scale": 0.7,
                },
            })
            resp.raise_for_status()
            stream_data = resp.json()
            stream_id = stream_data["stream_id"]
            result["stream_id"] = stream_id
            logger.info("[%s] Stream: %s", duration_label, stream_id)

            # 2. Wait for runner
            for attempt in range(120):
                status_resp = await client.get(f"/stream/{stream_id}/status")
                if status_resp.status_code == 404:
                    raise RuntimeError("Stream disappeared waiting for runner")
                phase = status_resp.json().get("phase", "unknown")
                if phase in ("ready", "running", "connecting"):
                    result["connect_time_s"] = time.monotonic() - connect_start
                    logger.info("[%s] Runner ready in %.1fs", duration_label, result["connect_time_s"])
                    break
                if phase in ("error", "failed"):
                    raise RuntimeError(f"Stream failed: phase={phase}")
                await asyncio.sleep(5)
            else:
                raise TimeoutError("Runner not ready after 10 min")

            # 3. Start publishing input frames (background task)
            publish_seq = 0

            async def publisher():
                nonlocal publish_seq
                while not publish_stop.is_set():
                    frame = make_test_frame(frame_num=publish_seq)
                    try:
                        pub_resp = await client.post(
                            f"/stream/{stream_id}/publish",
                            params={"seq": publish_seq},
                            content=frame,
                            headers={"Content-Type": "image/jpeg"},
                        )
                        if pub_resp.is_success:
                            result["frames_published"] += 1
                            publish_seq += 1
                    except Exception:
                        pass
                    await asyncio.sleep(0.1)  # ~10 fps publish rate

            publish_task = asyncio.create_task(publisher())

            # 4. Wait for first output frame
            logger.info("[%s] Publishing frames, waiting for first output...", duration_label)
            ff_start = time.monotonic()
            for attempt in range(120):
                frame_resp = await client.get(f"/stream/{stream_id}/frame")
                if frame_resp.status_code == 200 and len(frame_resp.content) > 100:
                    result["first_frame_s"] = time.monotonic() - ff_start
                    result["frames_captured"] += 1
                    logger.info("[%s] First frame in %.1fs (%d bytes, published %d)",
                                duration_label, result["first_frame_s"], len(frame_resp.content), result["frames_published"])
                    break
                await asyncio.sleep(1)
            else:
                raise TimeoutError("No output frame after 2 min")

            # 5. Monitoring loop
            logger.info("[%s] Streaming for %ds...", duration_label, duration_s)
            stream_start = time.monotonic()
            prompt_idx = 0
            last_prompt_switch = stream_start

            while time.monotonic() - stream_start < duration_s:
                await asyncio.sleep(10)
                elapsed = time.monotonic() - stream_start

                # Capture output frame
                try:
                    frame_resp = await client.get(f"/stream/{stream_id}/frame")
                    if frame_resp.status_code == 200 and len(frame_resp.content) > 100:
                        result["frames_captured"] += 1
                except Exception:
                    pass

                # Status check
                try:
                    status_resp = await client.get(f"/stream/{stream_id}/status")
                    if status_resp.status_code == 404:
                        raise RuntimeError("Stream disappeared mid-session")
                    phase = status_resp.json().get("phase", "unknown")
                    logger.info("[%s] %.0fs: phase=%s pub=%d cap=%d",
                                duration_label, elapsed, phase, result["frames_published"], result["frames_captured"])
                except httpx.HTTPStatusError:
                    pass

                # Prompt switch every 30s
                if time.monotonic() - last_prompt_switch > 30:
                    prompt_idx = (prompt_idx + 1) % len(PROMPTS)
                    try:
                        await client.post(f"/stream/{stream_id}/control", json={"prompt": PROMPTS[prompt_idx]})
                        logger.info("[%s] Prompt: %s", duration_label, PROMPTS[prompt_idx][:40])
                    except Exception:
                        pass
                    last_prompt_switch = time.monotonic()

            result["passed"] = True

        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
            logger.error("[%s] Failed: %s", duration_label, result["error"])

        finally:
            publish_stop.set()
            if publish_task is not None:
                publish_task.cancel()
                try:
                    await publish_task
                except asyncio.CancelledError:
                    pass

            if result["stream_id"]:
                try:
                    await client.post(f"/stream/{result['stream_id']}/stop")
                    logger.info("[%s] Stopped (pub=%d cap=%d)", duration_label, result["frames_published"], result["frames_captured"])
                except Exception:
                    pass

    return result


async def main(sdk_url: str, api_key: str, durations: list[tuple[str, int]]):
    logger.info("=" * 60)
    logger.info("Scope Load Test: longlive v2v via SDK")
    logger.info("SDK: %s", sdk_url)
    logger.info("Durations: %s", [f"{l}={s}s" for l, s in durations])
    logger.info("=" * 60)

    results = []
    for label, secs in durations:
        logger.info("")
        logger.info("--- Starting %s v2v run (%ds) ---", label, secs)
        r = await run_session(sdk_url, api_key, label, secs)
        results.append(r)

        status = "PASS" if r["passed"] else f"FAIL: {r['error']}"
        logger.info("--- %s: %s (connect=%.1fs ff=%.1fs pub=%d cap=%d) ---",
                     label, status, r["connect_time_s"] or 0, r["first_frame_s"] or 0,
                     r["frames_published"], r["frames_captured"])

        if label != durations[-1][0]:
            logger.info("Waiting 10s...")
            await asyncio.sleep(10)

    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        logger.info("  %-6s %s  connect=%.1fs  ff=%.1fs  pub=%d  cap=%d  %s",
                     r["duration_label"], status, r["connect_time_s"] or 0,
                     r["first_frame_s"] or 0, r["frames_published"], r["frames_captured"],
                     r["error"] or "")

    passed = sum(1 for r in results if r["passed"])
    logger.info("")
    logger.info("Result: %d/%d passed", passed, len(results))
    return 0 if passed == len(results) else 1


DURATION_MAP = {"1": ("1m", 60), "5": ("5m", 300), "15": ("15m", 900)}


@click.command()
@click.option("--sdk-url", default=DEFAULT_SDK_URL)
@click.option("--api-key", envvar="DAYDREAM_API_KEY", required=True)
@click.option("--durations", default="1,5,15", help="e.g., 1,5,15")
def cli(sdk_url: str, api_key: str, durations: str):
    """Run longlive v2v load test via SDK with frame publishing."""
    dur_list = [DURATION_MAP[d.strip()] for d in durations.split(",") if d.strip() in DURATION_MAP]
    if not dur_list:
        click.echo("Invalid durations. Use 1, 5, or 15.")
        sys.exit(1)
    code = asyncio.run(main(sdk_url, api_key, dur_list))
    sys.exit(code)


if __name__ == "__main__":
    cli()
