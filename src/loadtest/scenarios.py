"""Scenario matrix expansion and session body construction."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Scenario:
    name: str
    pipeline: str
    mode: str  # t2v, v2v, i2v
    duration_mins: int
    graph: dict[str, Any] | None
    prompts_pool: str
    parameters: dict[str, Any]
    prompts_pools: list[str] | None = None  # all available pools for rotation

    @property
    def pipeline_ids(self) -> list[str]:
        if self.graph:
            return [
                n["pipeline_id"]
                for n in self.graph["nodes"]
                if n.get("type") == "pipeline"
            ]
        return [self.pipeline]

    @property
    def duration_class(self) -> str:
        if self.duration_mins <= 2:
            return "short"
        if self.duration_mins <= 10:
            return "mid"
        return "long"

    @property
    def sink_node_id(self) -> str | None:
        if not self.graph:
            return None
        for n in self.graph["nodes"]:
            if n.get("type") == "sink":
                return n["id"]
        return None


def expand_scenario_matrix(
    scenario_defs: list[dict[str, Any]], graphs_dir: Path
) -> list[Scenario]:
    """Expand compact matrix config into concrete Scenario objects."""
    scenarios = []
    for entry in scenario_defs:
        pipeline = entry["pipeline"]
        graph_template = entry.get("graph_template")
        source_files = entry.get("source_files", {})
        graph = None
        if graph_template:
            graph_path = graphs_dir / f"{graph_template}.yaml"
            if not graph_path.exists():
                raise FileNotFoundError(f"Graph template not found: {graph_path}")
            with open(graph_path) as f:
                graph = yaml.safe_load(f)

        # Support both singular "prompts_pool" and plural "prompts_pools"
        pools = entry.get("prompts_pools") or [entry.get("prompts_pool", "nature")]

        for mode in entry.get("modes", ["t2v"]):
            for dur in entry.get("durations", [5]):
                name = f"{pipeline.replace('+', '_')}_{mode}_{dur}m"
                params = dict(entry.get("parameters", {}))
                if mode in source_files:
                    params["source_name"] = source_files[mode]
                scenarios.append(
                    Scenario(
                        name=name,
                        pipeline=pipeline,
                        mode=mode,
                        duration_mins=dur,
                        graph=graph,
                        prompts_pool=pools[0],  # default pool
                        prompts_pools=pools,     # all available pools
                        parameters=params,
                    )
                )
    return scenarios


def load_prompt_pool(pool_name: str, prompts_dir: Path) -> list[str]:
    """Load a named prompt pool from the prompts directory."""
    path = prompts_dir / f"{pool_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt pool not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f).get("prompts", [])


def build_session_body(scenario: Scenario, prompt: str) -> dict[str, Any]:
    """Build the POST /api/v1/session/start request body."""
    if scenario.graph:
        return {
            "input_mode": "video" if scenario.mode in ("v2v", "i2v") else "text",
            "graph": scenario.graph,
            "prompts": [{"text": prompt, "weight": 100}],
        }
    body: dict[str, Any] = {
        "pipeline_id": scenario.pipeline,
        "input_mode": "video" if scenario.mode in ("v2v", "i2v") else "text",
        "prompts": [{"text": prompt, "weight": 100}],
    }
    if scenario.mode in ("v2v", "i2v") and "source_name" in scenario.parameters:
        body["input_source"] = {
            "enabled": True,
            "source_type": "video_file",
            "source_name": scenario.parameters["source_name"],
        }
    return body
