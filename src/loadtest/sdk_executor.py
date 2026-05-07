"""Drives a single test scenario via the Daydream SDK service."""

import asyncio
import io
import logging
import time
from typing import Any

import numpy as np
from PIL import Image

from .config import LoadTestConfig
from .results import (
    ErrorCategory,
    PhaseTimings,
    RunResult,
    classify_error,
)
from .scenarios import Scenario
from .sdk_client import SDKClient
from .validators import FrameCheckResult, check_prompt_sensitivity, validate_frame

logger = logging.getLogger(__name__)


def _make_input_frame(width: int, height: int, frame_num: int) -> bytes:
    """Generate a synthetic JPEG input frame with varying color."""
    t = (frame_num % 300) / 300.0
    r = int(128 + 127 * np.sin(t * 2 * np.pi))
    g = int(128 + 127 * np.sin(t * 2 * np.pi + 2.094))
    b = int(128 + 127 * np.sin(t * 2 * np.pi + 4.189))
    img = Image.new("RGB", (width, height), (r, g, b))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return buf.getvalue()


class SDKExecutor:
    """Executes a test scenario via the Daydream SDK service."""

    def __init__(self, config: LoadTestConfig):
        self._config = config

    async def run(
        self,
        sdk_url: str,
        api_key: str,
        scenario: Scenario,
        prompts: list[str],
        orchestrator_id: str = "auto",
    ) -> RunResult:
        """Execute a complete test scenario. Always returns a RunResult."""
        result = RunResult(
            scenario=scenario.name,
            orchestrator_id=orchestrator_id,
            passed=False,
            labels={
                "pipeline": scenario.pipeline,
                "mode": scenario.mode,
                "duration_class": scenario.duration_class,
            },
        )

        total_start = time.monotonic()
        max_duration = self._config.budget.max_run_duration_mins * 60
        thresholds = self._config.thresholds
        stream_id = None
        publish_stop = asyncio.Event()
        publish_task = None

        try:
            async with asyncio.timeout(max_duration):
                async with SDKClient(sdk_url, api_key) as client:
                    # 1. Start stream
                    connect_start = time.monotonic()
                    params: dict[str, Any] = {"prompt": prompts[0] if prompts else "a scenic landscape"}
                    params["pipeline_ids"] = scenario.pipeline_ids

                    if scenario.mode in ("v2v", "i2v"):
                        params["graph"] = scenario.graph or {
                            "nodes": [
                                {"id": "input", "type": "source", "source_mode": "video"},
                                {"id": scenario.pipeline, "type": "pipeline", "pipeline_id": scenario.pipeline},
                                {"id": "output", "type": "sink"},
                            ],
                            "edges": [
                                {"from": "input", "from_port": "video", "to_node": scenario.pipeline, "to_port": "video", "kind": "stream"},
                                {"from": scenario.pipeline, "from_port": "video", "to_node": "output", "to_port": "video", "kind": "stream"},
                            ],
                        }
                        if "noise_scale" in scenario.parameters:
                            params["noise_scale"] = scenario.parameters["noise_scale"]

                    stream_data = await client.stream_start(params)
                    stream_id = stream_data["stream_id"]

                    # 2. Wait for runner
                    for _ in range(120):
                        status = await client.stream_status(stream_id)
                        if status is None:
                            raise RuntimeError("Stream disappeared waiting for runner")
                        phase = status.get("phase", "unknown")
                        if phase in ("ready", "running", "connecting"):
                            result.timings.connect_s = time.monotonic() - connect_start
                            result.cold_start = result.timings.connect_s > thresholds.cold_start_threshold_s
                            break
                        if phase in ("error", "failed"):
                            raise RuntimeError(f"Stream failed: phase={phase}")
                        await asyncio.sleep(5)
                    else:
                        raise TimeoutError("Runner not ready after 10 min")

                    # 3. Start publishing for v2v/i2v
                    publish_seq = 0
                    width = scenario.parameters.get("width", 512)
                    height = scenario.parameters.get("height", 512)

                    if scenario.mode in ("v2v", "i2v"):
                        async def _publisher():
                            nonlocal publish_seq
                            while not publish_stop.is_set():
                                frame = _make_input_frame(width, height, publish_seq)
                                try:
                                    await client.stream_publish(stream_id, frame, publish_seq)
                                    publish_seq += 1
                                except Exception:
                                    pass
                                await asyncio.sleep(0.1)

                        publish_task = asyncio.create_task(_publisher())

                    # 4. Wait for first output frame
                    ff_start = time.monotonic()
                    for _ in range(120):
                        frame_data = await client.stream_frame(stream_id)
                        if frame_data and len(frame_data) > 100:
                            result.timings.first_frame_s = time.monotonic() - ff_start
                            result.frames_validated += 1
                            break
                        await asyncio.sleep(1)
                    else:
                        raise TimeoutError("No output frame after 2 min")

                    # 5. Monitoring loop
                    duration_s = scenario.duration_mins * 60
                    stream_start = time.monotonic()
                    prompt_idx = 0
                    last_prompt_switch = stream_start
                    check_interval = thresholds.frame_check_interval_s

                    while time.monotonic() - stream_start < duration_s:
                        await asyncio.sleep(check_interval)
                        elapsed = time.monotonic() - stream_start

                        # Status check
                        status = await client.stream_status(stream_id)
                        if status is None:
                            raise RuntimeError("Stream disappeared mid-session")

                        # Frame capture + validation
                        try:
                            frame_data = await client.stream_frame(stream_id)
                            if frame_data and len(frame_data) > 100:
                                check = validate_frame(frame_data, width, height, thresholds.frame_variance_min)
                                result.frames_validated += 1
                                if check == FrameCheckResult.BLACK:
                                    result.frames_black += 1
                                elif check == FrameCheckResult.CORRUPT:
                                    result.frames_corrupt += 1
                        except Exception:
                            pass

                        # Prompt switching
                        if time.monotonic() - last_prompt_switch > thresholds.prompt_switch_interval_s and len(prompts) > 1:
                            frame_before = None
                            try:
                                frame_before = await client.stream_frame(stream_id)
                            except Exception:
                                pass

                            prompt_idx = (prompt_idx + 1) % len(prompts)
                            await client.stream_control(stream_id, {"prompt": prompts[prompt_idx]})
                            last_prompt_switch = time.monotonic()

                            if frame_before:
                                await asyncio.sleep(min(10, duration_s / 4))
                                try:
                                    frame_after = await client.stream_frame(stream_id)
                                    if frame_after:
                                        result.prompt_sensitivity_checks += 1
                                        if not check_prompt_sensitivity(frame_before, frame_after, thresholds.prompt_diff_min):
                                            result.prompt_sensitivity_failures += 1
                                except Exception:
                                    pass

                    result.timings.stream_duration_s = time.monotonic() - stream_start
                    result.passed = True

        except Exception as e:
            result.error_category = classify_error(e)
            result.error_message = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            logger.error("Run failed [%s/%s]: %s: %s", orchestrator_id, scenario.name, type(e).__name__, e)

        finally:
            publish_stop.set()
            if publish_task is not None:
                publish_task.cancel()
                try:
                    await publish_task
                except asyncio.CancelledError:
                    pass

            if stream_id:
                try:
                    async with SDKClient(sdk_url, api_key) as client:
                        await client.stream_stop(stream_id)
                except Exception:
                    pass

        result.timings.total_s = time.monotonic() - total_start
        return result
