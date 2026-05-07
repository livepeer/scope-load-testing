import pytest
from loadtest.scheduler import RunSlot, build_run_plan
from loadtest.discovery import Orchestrator


def test_build_run_plan_distributes_evenly():
    orchestrators = [
        Orchestrator(id="O-1", address="http://g1"),
        Orchestrator(id="O-2", address="http://g2"),
        Orchestrator(id="O-3", address="http://g3"),
    ]
    scenarios = ["s1", "s2", "s3", "s4"]
    runs_per_o = 4
    num_instances = 2

    plan = build_run_plan(orchestrators, scenarios, runs_per_o, num_instances)

    # Each orchestrator gets exactly runs_per_o slots
    for o in orchestrators:
        o_slots = [s for s in plan if s.orchestrator_id == o.id]
        assert len(o_slots) == runs_per_o

    # At most num_instances concurrent at any slot index
    by_time: dict[int, list[RunSlot]] = {}
    for slot in plan:
        by_time.setdefault(slot.slot_index, []).append(slot)
    for slots in by_time.values():
        assert len(slots) <= num_instances


def test_build_run_plan_rotates_scenarios():
    orchestrators = [Orchestrator(id="O-1", address="http://g1")]
    scenarios = ["s1", "s2", "s3"]
    runs_per_o = 6

    plan = build_run_plan(orchestrators, scenarios, runs_per_o, num_instances=1)

    assigned = [s.scenario for s in plan]
    # Each scenario should appear exactly twice (6 runs / 3 scenarios)
    for sc in scenarios:
        assert assigned.count(sc) == 2


def test_build_run_plan_single_instance():
    orchestrators = [
        Orchestrator(id="O-1", address="http://g1"),
        Orchestrator(id="O-2", address="http://g2"),
    ]
    scenarios = ["s1"]
    runs_per_o = 2

    plan = build_run_plan(orchestrators, scenarios, runs_per_o, num_instances=1)

    # 2 orchestrators * 2 runs = 4 total slots, all sequential (1 per slot)
    assert len(plan) == 4
    by_time: dict[int, list[RunSlot]] = {}
    for slot in plan:
        by_time.setdefault(slot.slot_index, []).append(slot)
    for slots in by_time.values():
        assert len(slots) == 1


def test_build_run_plan_empty_orchestrators():
    plan = build_run_plan([], ["s1", "s2"], runs_per_orchestrator=5, num_instances=2)
    assert len(plan) == 0


def test_build_run_plan_interleaves_orchestrators():
    """Consecutive slots should not test the same orchestrator."""
    orchestrators = [
        Orchestrator(id="O-1", address="http://g1"),
        Orchestrator(id="O-2", address="http://g2"),
    ]
    scenarios = ["s1"]
    runs_per_o = 3

    plan = build_run_plan(orchestrators, scenarios, runs_per_o, num_instances=1)

    # With 1 instance, slots are sequential. Check interleaving:
    # Should alternate O-1, O-2, O-1, O-2, ...
    for i in range(len(plan) - 1):
        if plan[i].slot_index != plan[i + 1].slot_index:
            # Different slots — orchestrators should differ
            assert plan[i].orchestrator_id != plan[i + 1].orchestrator_id
