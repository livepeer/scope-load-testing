"""Drives a single test scenario against a Scope instance."""

import asyncio
import logging
import time
from pathlib import Path

from .config import LoadTestConfig
from .results import (
    ErrorCategory,
    PhaseTimings,
    RunResult,
    classify_error,
    save_failure_logs,
)
from .scenarios import Scenario, build_session_body
from .scope_client import ScopeClient
from .validators import (
    FrameCheckResult,
    check_prompt_sensitivity,
    validate_frame,
)

logger = logging.getLogger(__name__)


class Executor:
    """Executes a single test scenario against one Scope instance."""

    def __init__(self, config: LoadTestConfig, data_dir: Path | None = None):
        self._config = config
        self._data_dir = data_dir or Path("data")

    async def _connect_phase(
        self, scope_url: str, app_id: str, api_key: str | None = None
    ) -> float:
        """Connect to cloud. Returns duration in seconds."""
        start = time.monotonic()
        timeout = self._config.thresholds.connect_timeout_s

        async with ScopeClient(scope_url) as client:
            await client.cloud_connect(app_id=app_id, api_key=api_key)

            deadline = start + timeout
            while time.monotonic() < deadline:
                status = await client.cloud_status()
                if status.get("connected"):
                    return time.monotonic() - start
                if status.get("error"):
                    raise RuntimeError(f"Cloud connect error: {status['error']}")
                await asyncio.sleep(1)

        raise TimeoutError(f"Cloud connect timed out after {timeout}s")

    async def _load_phase(self, scope_url: str, pipeline_ids: list[str]) -> float:
        """Load pipelines. Returns duration in seconds."""
        start = time.monotonic()
        timeout = self._config.thresholds.pipeline_load_timeout_s

        async with ScopeClient(scope_url) as client:
            await client.pipeline_load(pipeline_ids)

            deadline = start + timeout
            while time.monotonic() < deadline:
                status = await client.pipeline_status()
                if status.get("status") == "loaded":
                    return time.monotonic() - start
                if status.get("status") == "error":
                    raise RuntimeError(
                        f"Pipeline load error: {status.get('error_message')}"
                    )
                await asyncio.sleep(1)

        raise TimeoutError(f"Pipeline load timed out after {timeout}s")

    async def _stream_phase(
        self,
        scope_url: str,
        scenario: Scenario,
        prompts: list[str],
        result: RunResult,
    ) -> None:
        """Run the streaming session with monitoring loop."""
        thresholds = self._config.thresholds
        check_interval = thresholds.frame_check_interval_s
        switch_interval = thresholds.prompt_switch_interval_s
        expected_w = scenario.parameters.get("width", 512)
        expected_h = scenario.parameters.get("height", 512)
        duration_s = scenario.duration_mins * 60
        prompt_idx = 0

        async with ScopeClient(scope_url, timeout=60.0) as client:
            body = build_session_body(scenario, prompts[prompt_idx])
            await client.session_start(body)

            # Wait for first frame
            ff_start = time.monotonic()
            while time.monotonic() - ff_start < thresholds.first_frame_timeout_s:
                metrics = await client.session_metrics()
                session = metrics.get("sessions", {}).get("headless", {})
                if session.get("frames_out", 0) > 0:
                    result.timings.first_frame_s = time.monotonic() - ff_start
                    break
                await asyncio.sleep(1)
            else:
                raise TimeoutError("No first frame within timeout")

            # Monitoring loop
            stream_start = time.monotonic()
            last_check = 0.0
            last_prompt_switch = stream_start
            stall_start: float | None = None

            while time.monotonic() - stream_start < duration_s:
                now = time.monotonic()

                if now - last_check >= check_interval:
                    last_check = now

                    metrics = await client.session_metrics()
                    session = metrics.get("sessions", {}).get("headless", {})
                    gpu = metrics.get("gpu", {})

                    fps_out = session.get("fps_out", 0)
                    result.fps_samples.append(fps_out)

                    vram = gpu.get("vram_allocated_mb", 0)
                    if vram:
                        result.vram_samples.append(vram)

                    # Stall detection
                    if fps_out == 0:
                        if stall_start is None:
                            stall_start = now
                        elif now - stall_start > thresholds.stall_timeout_s:
                            raise RuntimeError(
                                f"Stream stalled for {thresholds.stall_timeout_s}s"
                            )
                    else:
                        stall_start = None

                    # Frame validation
                    try:
                        frame = await client.capture_frame(
                            sink_node_id=scenario.sink_node_id
                        )
                        check = validate_frame(
                            frame, expected_w, expected_h, thresholds.frame_variance_min
                        )
                        result.frames_validated += 1
                        if check == FrameCheckResult.BLACK:
                            result.frames_black += 1
                        elif check == FrameCheckResult.CORRUPT:
                            result.frames_corrupt += 1
                    except Exception:
                        pass

                # Prompt switching
                if (
                    now - last_prompt_switch >= switch_interval
                    and len(prompts) > 1
                ):
                    # Capture frame before switch
                    frame_before = None
                    try:
                        frame_before = await client.capture_frame(
                            sink_node_id=scenario.sink_node_id
                        )
                    except Exception:
                        pass

                    prompt_idx = (prompt_idx + 1) % len(prompts)
                    await client.session_parameters(
                        {"prompts": [{"text": prompts[prompt_idx], "weight": 100}]}
                    )
                    last_prompt_switch = now

                    # Wait and capture after
                    if frame_before:
                        await asyncio.sleep(min(10, duration_s / 4))
                        try:
                            frame_after = await client.capture_frame(
                                sink_node_id=scenario.sink_node_id
                            )
                            result.prompt_sensitivity_checks += 1
                            if not check_prompt_sensitivity(
                                frame_before,
                                frame_after,
                                thresholds.prompt_diff_min,
                            ):
                                result.prompt_sensitivity_failures += 1
                        except Exception:
                            pass

                await asyncio.sleep(1)

            result.timings.stream_duration_s = time.monotonic() - stream_start

            await client.session_stop()

    async def run(
        self,
        scope_url: str,
        orchestrator_id: str,
        scenario: Scenario,
        prompts: list[str],
        app_id: str,
        api_key: str | None = None,
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

        try:
            async with asyncio.timeout(max_duration):
                # 1. Connect
                connect_s = await self._connect_phase(scope_url, app_id, api_key)
                result.timings.connect_s = connect_s
                result.cold_start = (
                    connect_s > self._config.thresholds.cold_start_threshold_s
                )

                # 2. Load
                result.timings.pipeline_load_s = await self._load_phase(
                    scope_url, scenario.pipeline_ids
                )

                # 3. Stream
                if not prompts:
                    prompts = ["a test scene"]
                await self._stream_phase(scope_url, scenario, prompts, result)

                # 4. Disconnect
                async with ScopeClient(scope_url) as client:
                    await client.cloud_disconnect()

                result.passed = True

        except Exception as e:
            result.error_category = classify_error(e)
            result.error_message = str(e)
            logger.error(
                "Run failed [%s/%s]: %s: %s",
                orchestrator_id,
                scenario.name,
                type(e).__name__,
                e,
            )

            # Capture logs on failure
            try:
                async with ScopeClient(scope_url) as client:
                    logs = await client.get_logs(lines=100)
                    log_text = "\n".join(logs.get("logs", []))
                    save_failure_logs(
                        log_text, orchestrator_id, scenario.name, self._data_dir
                    )
            except Exception:
                pass

            # Force cleanup
            try:
                async with ScopeClient(scope_url) as client:
                    await client.session_stop()
                    await client.cloud_disconnect()
            except Exception:
                pass

        result.timings.total_s = time.monotonic() - total_start
        return result
