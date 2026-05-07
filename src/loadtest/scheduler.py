"""Scheduler daemon: budget calculation, run timing, orchestrator rotation."""

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import LoadTestConfig
from .coverage import CoverageTracker
from .discovery import (
    Orchestrator,
    OrchestratorRegistry,
    discover_orchestrators,
)
from .executor import Executor
from .metrics import MetricsCollector
from .regression import BaselineStore, check_drift, update_baseline
from .scenarios import Scenario, expand_scenario_matrix, load_prompt_pool

logger = logging.getLogger(__name__)


@dataclass
class RunSlot:
    slot_index: int
    orchestrator_id: str
    scenario: str


def build_run_plan(
    orchestrators: list[Orchestrator],
    scenarios: list[str],
    runs_per_orchestrator: int,
    num_instances: int,
) -> list[RunSlot]:
    """Build a daily run plan interleaving orchestrators across time slots.

    Returns a list of RunSlot ordered by slot_index. At most num_instances
    slots share the same slot_index.
    """
    if not orchestrators or not scenarios:
        return []

    # Build per-orchestrator run lists with rotating scenarios
    by_o: dict[str, list[str]] = {}
    for o in orchestrators:
        runs = []
        for i in range(runs_per_orchestrator):
            runs.append(scenarios[i % len(scenarios)])
        by_o[o.id] = runs

    # Interleave: round-robin across orchestrators
    interleaved: list[tuple[str, str]] = []
    oids = list(by_o.keys())
    max_runs = max(len(v) for v in by_o.values())
    for run_idx in range(max_runs):
        for oid in oids:
            if run_idx < len(by_o[oid]):
                interleaved.append((oid, by_o[oid][run_idx]))

    # Assign slot indices (at most num_instances per slot)
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
    """Main scheduler loop. Runs until cancelled."""
    graphs_dir = config_dir / "graphs"
    prompts_dir = config_dir / "prompts"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Expand scenario matrix
    all_scenarios = expand_scenario_matrix(config.scenario_defs, graphs_dir)
    scenario_names = [s.name for s in all_scenarios]
    scenario_map = {s.name: s for s in all_scenarios}

    if not scenario_names:
        logger.error("No scenarios configured, exiting")
        return

    coverage = CoverageTracker(data_dir / "coverage.json")
    baseline_store = BaselineStore(
        data_dir / "baselines.json", data_dir / "history.json"
    )

    push_url = os.environ.get("GRAFANA_PUSH_URL") or None
    metrics = MetricsCollector(push_url=push_url)
    executor = Executor(config, data_dir=data_dir)

    scope_instances = os.environ.get("SCOPE_INSTANCES", "").split(",")
    scope_instances = [s.strip() for s in scope_instances if s.strip()]
    if not scope_instances:
        logger.error("No SCOPE_INSTANCES configured, exiting")
        return

    scope_urls = [f"http://{inst}" for inst in scope_instances]

    discovery_url = os.environ.get("LIVEPEER_DISCOVERY_URL", "")
    livepeer_token = os.environ.get("LIVEPEER_TOKEN")
    app_id = os.environ.get("SCOPE_CLOUD_APP_ID", "")
    api_key = os.environ.get("SCOPE_CLOUD_API_KEY")

    registry = OrchestratorRegistry(
        max_consecutive_failures=config.discovery.max_consecutive_failures,
    )

    logger.info(
        "Scheduler starting: %d instances, %d scenarios, budget=%d%%, max_run=%dm",
        len(scope_urls),
        len(scenario_names),
        config.budget.daily_percent,
        config.budget.max_run_duration_mins,
    )

    last_discovery = 0.0
    discovery_interval = config.discovery.refresh_interval_hours * 3600

    while True:
        now = datetime.now(timezone.utc)

        # Periodic discovery
        if now.timestamp() - last_discovery > discovery_interval:
            logger.info("Running orchestrator discovery...")
            try:
                discovered = await discover_orchestrators(
                    discovery_url, livepeer_token
                )
                for o in discovered:
                    registry.upsert(o)
            except Exception as e:
                logger.error("Discovery failed: %s", e)

            last_discovery = now.timestamp()

            # Set daily plan for healthy orchestrators
            healthy = registry.get_healthy()
            runs_per_o = config.budget.runs_per_orchestrator_per_day
            for o in healthy:
                coverage.set_planned(o.id, runs_per_o)

            # Update coverage metric
            all_o = registry.get_all()
            if all_o:
                pct = (len(healthy) / len(all_o)) * 100
                metrics.orchestrator_coverage_percent.set(pct)

            logger.info(
                "Plan: %d healthy orchestrators, %d runs each, %d scenarios",
                len(healthy),
                runs_per_o,
                len(scenario_names),
            )

        # Find orchestrators with test debt
        debt = coverage.get_test_debt()
        candidates = [(oid, d) for oid, d in debt.items() if d > 0]
        if not candidates:
            logger.info("All orchestrators at budget for today, sleeping 5m...")
            await asyncio.sleep(300)
            continue

        # Assign runs to available scope instances
        tasks = []
        used_oids: set[str] = set()
        for scope_url in scope_urls:
            # Pick the orchestrator with most debt that isn't already assigned
            pick = None
            for oid, d in candidates:
                if oid not in used_oids:
                    pick = oid
                    used_oids.add(oid)
                    break
            if pick is None:
                break

            # Pick scenario: prefer uncovered ones for this orchestrator
            o_coverage = coverage.get_today().get(pick, {})
            covered = set(o_coverage.get("scenarios_covered", []))
            uncovered = [s for s in scenario_names if s not in covered]
            scenario_name = uncovered[0] if uncovered else scenario_names[0]
            scenario = scenario_map[scenario_name]

            # Load prompts
            try:
                prompts = load_prompt_pool(scenario.prompts_pool, prompts_dir)
            except FileNotFoundError:
                prompts = ["a scenic landscape"]

            orchestrator = registry.get(pick)

            tasks.append(
                _run_one(
                    executor=executor,
                    scope_url=scope_url,
                    orchestrator_id=pick,
                    orchestrator=orchestrator,
                    scenario=scenario,
                    prompts=prompts,
                    app_id=app_id,
                    api_key=api_key,
                    coverage=coverage,
                    baseline_store=baseline_store,
                    metrics=metrics,
                    config=config,
                )
            )

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logger.error("Run task failed: %s", r)

            metrics.push()

        # Wait before next batch
        gap_s = config.budget.min_run_gap_mins * 60
        logger.info("Batch complete, waiting %ds before next batch", gap_s)
        await asyncio.sleep(gap_s)


async def _run_one(
    *,
    executor: Executor,
    scope_url: str,
    orchestrator_id: str,
    orchestrator: Orchestrator | None,
    scenario: Scenario,
    prompts: list[str],
    app_id: str,
    api_key: str | None,
    coverage: CoverageTracker,
    baseline_store: BaselineStore,
    metrics: MetricsCollector,
    config: LoadTestConfig,
) -> None:
    """Execute one run and update all tracking state."""
    result = await executor.run(
        scope_url=scope_url,
        orchestrator_id=orchestrator_id,
        scenario=scenario,
        prompts=prompts,
        app_id=app_id,
        api_key=api_key,
    )

    # Record coverage
    coverage.record_run(
        orchestrator_id,
        scenario.name,
        result.passed,
        failure_category=result.error_category.value
        if result.error_category
        else None,
    )

    # Update baselines and check drift
    if result.passed:
        update_baseline(baseline_store, result)
        drift = check_drift(
            baseline_store, result, config.thresholds.regression_drift_threshold
        )
        if drift.first_frame_drifted or drift.fps_drifted or drift.pipeline_load_drifted:
            logger.warning(
                "Drift detected for %s on %s: ff=%.1f%% fps=%.1f%% load=%.1f%%",
                scenario.name,
                orchestrator_id,
                drift.first_frame_drift_pct * 100,
                drift.fps_drift_pct * 100,
                drift.pipeline_load_drift_pct * 100,
            )

    # Update orchestrator health
    if orchestrator:
        if result.passed:
            orchestrator.record_success()
        elif result.error_category in ("network", "orchestrator"):
            orchestrator.record_failure()
        orchestrator.record_tested()

    # Record metrics
    metrics.record_run(result)

    status = "PASS" if result.passed else f"FAIL [{result.error_category}]"
    logger.info(
        "%s %s/%s (%.1fs)",
        status,
        orchestrator_id,
        scenario.name,
        result.timings.total_s or 0,
    )
