"""Track which orchestrators have been tested and their daily progress."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


class CoverageTracker:
    """Tracks per-orchestrator, per-day test coverage. Persists to JSON."""

    def __init__(self, path: Path, max_days: int = 30):
        self._path = path
        self._max_days = max_days
        self._data: dict = self._load()

    def _load(self) -> dict:
        if not self._path.exists():
            return {}
        with open(self._path) as f:
            data = json.load(f)
        # Prune entries older than max_days
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=self._max_days)
        ).strftime("%Y-%m-%d")
        return {k: v for k, v in data.items() if k >= cutoff}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2)

    def _today_key(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _ensure_day(self, orchestrator_id: str) -> dict:
        day_key = self._today_key()
        if day_key not in self._data:
            self._data[day_key] = {}
        if orchestrator_id not in self._data[day_key]:
            self._data[day_key][orchestrator_id] = {
                "runs_completed": 0,
                "runs_planned": 0,
                "scenarios_covered": [],
                "failures": 0,
                "failure_categories": {},
            }
        return self._data[day_key][orchestrator_id]

    def get_today(self) -> dict:
        return self._data.get(self._today_key(), {})

    def set_planned(self, orchestrator_id: str, runs_planned: int) -> None:
        entry = self._ensure_day(orchestrator_id)
        entry["runs_planned"] = runs_planned
        self._save()

    def record_run(
        self,
        orchestrator_id: str,
        scenario: str,
        passed: bool,
        failure_category: str | None = None,
    ) -> None:
        entry = self._ensure_day(orchestrator_id)
        entry["runs_completed"] += 1
        if scenario not in entry["scenarios_covered"]:
            entry["scenarios_covered"].append(scenario)
        if not passed:
            entry["failures"] += 1
            if failure_category:
                cats = entry["failure_categories"]
                cats[failure_category] = cats.get(failure_category, 0) + 1
        self._save()

    def get_test_debt(self) -> dict[str, int]:
        """Return {orchestrator_id: remaining_runs} for today, sorted by most debt."""
        today = self.get_today()
        debt = {}
        for oid, entry in today.items():
            remaining = entry["runs_planned"] - entry["runs_completed"]
            debt[oid] = max(0, remaining)
        return dict(sorted(debt.items(), key=lambda x: -x[1]))
