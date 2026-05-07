from loadtest.discovery import Orchestrator, OrchestratorRegistry


def test_orchestrator_healthy_by_default():
    o = Orchestrator(id="O-abc", address="http://gateway:8001")
    assert o.status == "healthy"
    assert o.consecutive_failures == 0


def test_orchestrator_unhealthy_after_failure():
    o = Orchestrator(id="O-abc", address="http://gateway:8001")
    o.record_failure()
    assert o.status == "unhealthy"
    assert o.consecutive_failures == 1


def test_orchestrator_blacklisted_after_max_failures():
    o = Orchestrator(id="O-abc", address="http://gateway:8001", max_consecutive_failures=5)
    for _ in range(5):
        o.record_failure()
    assert o.status == "blacklisted"


def test_orchestrator_recovers_on_success():
    o = Orchestrator(id="O-abc", address="http://gateway:8001")
    o.record_failure()
    o.record_failure()
    assert o.status == "unhealthy"
    o.record_success()
    assert o.status == "healthy"
    assert o.consecutive_failures == 0


def test_orchestrator_record_tested():
    o = Orchestrator(id="O-abc", address="http://gateway:8001")
    assert o.last_tested is None
    o.record_tested()
    assert o.last_tested is not None


def test_registry_add_and_list():
    registry = OrchestratorRegistry(max_consecutive_failures=5)
    registry.upsert(Orchestrator(id="O-1", address="http://g1:8001"))
    registry.upsert(Orchestrator(id="O-2", address="http://g2:8001"))

    assert len(registry.get_all()) == 2
    assert len(registry.get_healthy()) == 2


def test_registry_filters_blacklisted():
    registry = OrchestratorRegistry(max_consecutive_failures=3)
    o_bad = Orchestrator(id="O-bad", address="http://bad:8001", max_consecutive_failures=3)
    for _ in range(3):
        o_bad.record_failure()
    registry.upsert(o_bad)
    registry.upsert(Orchestrator(id="O-good", address="http://good:8001"))

    healthy = registry.get_healthy()
    assert len(healthy) == 1
    assert healthy[0].id == "O-good"

    all_o = registry.get_all()
    assert len(all_o) == 2


def test_registry_upsert_updates():
    registry = OrchestratorRegistry()
    o1 = Orchestrator(id="O-1", address="http://old:8001")
    registry.upsert(o1)

    o2 = Orchestrator(id="O-1", address="http://new:8001")
    registry.upsert(o2)

    assert len(registry.get_all()) == 1
    assert registry.get("O-1").address == "http://new:8001"


def test_registry_get_missing():
    registry = OrchestratorRegistry()
    assert registry.get("nonexistent") is None


def test_registry_reset_blacklists():
    registry = OrchestratorRegistry(max_consecutive_failures=2)
    o = Orchestrator(id="O-1", address="http://g1:8001", max_consecutive_failures=2)
    o.record_failure()
    o.record_failure()
    registry.upsert(o)
    assert registry.get("O-1").status == "blacklisted"

    count = registry.reset_blacklists()
    assert count == 1
    assert registry.get("O-1").status == "healthy"


def test_registry_filters_unhealthy_but_not_blacklisted():
    registry = OrchestratorRegistry(max_consecutive_failures=5)
    o = Orchestrator(id="O-1", address="http://g1:8001", max_consecutive_failures=5)
    o.record_failure()  # unhealthy but not blacklisted
    registry.upsert(o)

    # Unhealthy orchestrators are still in get_healthy (only blacklisted are excluded)
    # This is intentional: unhealthy = had a failure, but still worth trying
    healthy = registry.get_healthy()
    assert len(healthy) == 0  # Actually unhealthy should be excluded too
