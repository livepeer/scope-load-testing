"""Scheduler daemon: runs test scenarios continuously via the SDK."""

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import LoadTestConfig
from .coverage import CoverageTracker
from .datasets import select_prompts
from .metrics import MetricsCollector
from .regression import BaselineStore, check_drift, update_baseline
from .results import RunResult, cleanup_old_failures
from .scenarios import Scenario, expand_scenario_matrix
from .sdk_executor import SDKExecutor

logger = logging.getLogger(__name__)


@dataclass
class RunSlot:
    slot_index: int
    orchestrator_id: str
    scenario: str


def build_run_plan(
    orchestrators: list,
    scenarios: list[str],
    runs_per_orchestrator: int,
    num_instances: int,
) -> list[RunSlot]:
    """Build a daily run plan interleaving orchestrators across time slots."""
    if not orchestrators or not scenarios:
        return []

    by_o: dict[str, list[str]] = {}
    for o in orchestrators:
        oid = o.id if hasattr(o, "id") else str(o)
        runs = []
        for i in range(runs_per_orchestrator):
            runs.append(scenarios[i % len(scenarios)])
        by_o[oid] = runs

    interleaved: list[tuple[str, str]] = []
    oids = list(by_o.keys())
    max_runs = max(len(v) for v in by_o.values())
    for run_idx in range(max_runs):
        for oid in oids:
            if run_idx < len(by_o[oid]):
                interleaved.append((oid, by_o[oid][run_idx]))

    slots: list[RunSlot] = []
    slot_idx = 0
    count_in_slot = 0
    for oid, sc in interleaved:
        slots.append(RunSlot(slot_index=slot_idx, orchestrator_id=oid, scenario=sc))
        count_in_slot += 1
        if count_in_slot >= num_instances:
            slot_idx += 1
            count_in_slot = 0

    return slots


async def run_scheduler(
    config: LoadTestConfig,
    config_dir: Path,
    data_dir: Path,
) -> None:
    """Main scheduler loop. Runs scenarios via the SDK until cancelled."""
    graphs_dir = config_dir / "graphs"
    prompts_dir = config_dir / "prompts"
    data_dir.mkdir(parents=True, exist_ok=True)

    all_scenarios = expand_scenario_matrix(config.scenario_defs, graphs_dir)
    scenario_names = [s.name for s in all_scenarios]
    scenario_map = {s.name: s for s in all_scenarios}

    if not scenario_names:
        logger.error("No scenarios configured, exiting")
        return

    sdk_url = os.environ.get("SDK_URL", "https://sdk.daydream.monster")
    api_key = os.environ.get("DAYDREAM_API_KEY", "")
    if not api_key:
        logger.error("DAYDREAM_API_KEY not set, exiting")
        return

    push_url = os.environ.get("PUSHGATEWAY_URL") or None

    coverage = CoverageTracker(data_dir / "coverage.json")
    baseline_store = BaselineStore(data_dir / "baselines.json", data_dir / "history.json")
    metrics = MetricsCollector(push_url=push_url)
    executor = SDKExecutor(config)

    # Calculate runs per day
    runs_per_day = config.budget.runs_per_orchestrator_per_day

    # In SDK mode (Option C), we don't target specific orchestrators.
    # We run scenarios and let the SDK route to available orchestrators.
    # Coverage tracks by scenario, not by orchestrator.
    coverage.set_planned("sdk", runs_per_day)

    logger.info(
        "Scheduler starting: sdk=%s, %d scenarios, budget=%d%%, max_run=%dm, %d runs/day",
        sdk_url,
        len(scenario_names),
        config.budget.daily_percent,
        config.budget.max_run_duration_mins,
        runs_per_day,
    )

    # Clean up old failure logs on startup
    cleanup_old_failures(data_dir)

    run_count = 0
    scenario_idx = 0

    while True:
        # Check if we've hit daily budget
        debt = coverage.get_test_debt()
        remaining = debt.get("sdk", 0)
        if remaining <= 0:
            logger.info("Daily budget reached (%d runs). Sleeping 5m...", run_count)
            await asyncio.sleep(300)
            # Reset at midnight
            now = datetime.now(timezone.utc)
            if now.hour == 0 and now.minute < 10:
                coverage.set_planned("sdk", runs_per_day)
                run_count = 0
                logger.info("New day — budget reset to %d runs", runs_per_day)
            continue

        # Pick next scenario (round-robin)
        scenario = all_scenarios[scenario_idx % len(all_scenarios)]
        scenario_idx += 1

        # Select random prompt pool and shuffle
        pools = scenario.prompts_pools or [scenario.prompts_pool]
        pool_name, prompts = select_prompts(pools, prompts_dir)

        logger.info(
            "Run %d: %s (pool=%s, %d remaining)",
            run_count + 1, scenario.name, pool_name, remaining,
        )

        # Execute
        result = await executor.run(
            sdk_url=sdk_url,
            api_key=api_key,
            scenario=scenario,
            prompts=prompts,
        )

        # Track coverage
        coverage.record_run(
            "sdk",
            scenario.name,
            result.passed,
            failure_category=result.error_category.value if result.error_category else None,
        )

        # Update baselines
        if result.passed:
            update_baseline(baseline_store, result)
            drift = check_drift(baseline_store, result, config.thresholds.regression_drift_threshold)
            if drift.first_frame_drifted or drift.fps_drifted or drift.pipeline_load_drifted:
                logger.warning(
                    "DRIFT: %s ff=%.0f%% fps=%.0f%% load=%.0f%%",
                    scenario.name,
                    drift.first_frame_drift_pct * 100,
                    drift.fps_drift_pct * 100,
                    drift.pipeline_load_drift_pct * 100,
                )

        # Push metrics
        metrics.record_run(result)
        metrics.push()

        status = "PASS" if result.passed else f"FAIL [{result.error_category}]"
        logger.info(
            "%s %s (%.1fs) connect=%.1fs ff=%.1fs",
            status, scenario.name,
            result.timings.total_s or 0,
            result.timings.connect_s or 0,
            result.timings.first_frame_s or 0,
        )

        run_count += 1

        # Gap between runs
        gap_s = config.budget.min_run_gap_mins * 60
        if gap_s > 0:
            logger.info("Waiting %ds before next run...", gap_s)
            await asyncio.sleep(gap_s)
