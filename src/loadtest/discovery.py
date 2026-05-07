"""Livepeer orchestrator discovery and health tracking."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class Orchestrator:
    id: str
    address: str
    region: str | None = None
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_healthy: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_tested: datetime | None = None
    consecutive_failures: int = 0
    max_consecutive_failures: int = 5

    @property
    def status(self) -> str:
        if self.consecutive_failures >= self.max_consecutive_failures:
            return "blacklisted"
        if self.consecutive_failures > 0:
            return "unhealthy"
        return "healthy"

    def record_failure(self) -> None:
        self.consecutive_failures += 1

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.last_healthy = datetime.now(timezone.utc)

    def record_tested(self) -> None:
        self.last_tested = datetime.now(timezone.utc)


class OrchestratorRegistry:
    """In-memory registry of known orchestrators."""

    def __init__(self, max_consecutive_failures: int = 5):
        self._orchestrators: dict[str, Orchestrator] = {}
        self._max_failures = max_consecutive_failures

    def upsert(self, orchestrator: Orchestrator) -> None:
        orchestrator.max_consecutive_failures = self._max_failures
        self._orchestrators[orchestrator.id] = orchestrator

    def get(self, oid: str) -> Orchestrator | None:
        return self._orchestrators.get(oid)

    def get_all(self) -> list[Orchestrator]:
        return list(self._orchestrators.values())

    def get_healthy(self) -> list[Orchestrator]:
        return [o for o in self._orchestrators.values() if o.status == "healthy"]

    def reset_blacklists(self) -> int:
        """Reset all blacklisted orchestrators to healthy. Returns count reset."""
        count = 0
        for o in self._orchestrators.values():
            if o.status == "blacklisted":
                o.consecutive_failures = 0
                count += 1
        return count


async def discover_orchestrators(
    discovery_url: str,
    livepeer_token: str | None = None,
) -> list[Orchestrator]:
    """Query the Livepeer discovery endpoint for available orchestrators.

    Returns a list of Orchestrator records. The actual Livepeer API contract
    should be adapted here once the discovery endpoint schema is confirmed.
    """
    import httpx

    orchestrators = []
    try:
        headers = {}
        if livepeer_token:
            headers["Authorization"] = f"Bearer {livepeer_token}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(discovery_url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            for entry in data:
                orchestrators.append(
                    Orchestrator(
                        id=entry.get("id", entry.get("address", "unknown")),
                        address=entry.get("address", ""),
                        region=entry.get("region"),
                    )
                )
    except Exception as e:
        logger.error("Orchestrator discovery failed: %s", e)

    logger.info("Discovered %d orchestrators", len(orchestrators))
    return orchestrators
