import pytest
from pathlib import Path
from loadtest.scenarios import (
    expand_scenario_matrix,
    Scenario,
    build_session_body,
    load_prompt_pool,
)


def test_expand_matrix_single_pipeline():
    defs = [
        {
            "pipeline": "longlive",
            "modes": ["t2v", "v2v"],
            "durations": [1, 5],
            "prompts_pool": "nature",
            "parameters": {"width": 512},
        }
    ]
    scenarios = expand_scenario_matrix(defs, graphs_dir=Path("/nonexistent"))
    assert len(scenarios) == 4  # 2 modes * 2 durations
    names = {s.name for s in scenarios}
    assert "longlive_t2v_1m" in names
    assert "longlive_v2v_5m" in names


def test_expand_matrix_with_graph(tmp_path: Path):
    graphs_dir = tmp_path / "graphs"
    graphs_dir.mkdir()
    (graphs_dir / "chain_test.yaml").write_text(
        "nodes:\n"
        "  - {id: input, type: source, source_mode: video_file, source_name: /data/videos/test.mp4}\n"
        "  - {id: longlive, type: pipeline, pipeline_id: longlive}\n"
        "  - {id: rife, type: pipeline, pipeline_id: rife}\n"
        "  - {id: output, type: sink}\n"
        "edges:\n"
        "  - {from: input, from_port: video, to_node: longlive, to_port: video, kind: stream}\n"
        "  - {from: longlive, from_port: video, to_node: rife, to_port: video, kind: stream}\n"
        "  - {from: rife, from_port: video, to_node: output, to_port: video, kind: stream}\n"
    )
    defs = [
        {
            "pipeline": "longlive+rife",
            "modes": ["v2v"],
            "durations": [5],
            "graph_template": "chain_test",
            "prompts_pool": "nature",
            "parameters": {},
        }
    ]
    scenarios = expand_scenario_matrix(defs, graphs_dir)
    assert len(scenarios) == 1
    assert scenarios[0].graph is not None
    assert scenarios[0].pipeline_ids == ["longlive", "rife"]


def test_expand_matrix_missing_graph_template(tmp_path: Path):
    defs = [
        {
            "pipeline": "longlive+rife",
            "modes": ["v2v"],
            "durations": [5],
            "graph_template": "nonexistent",
            "prompts_pool": "nature",
            "parameters": {},
        }
    ]
    with pytest.raises(FileNotFoundError):
        expand_scenario_matrix(defs, tmp_path / "graphs")


def test_expand_matrix_source_files_for_v2v():
    defs = [
        {
            "pipeline": "longlive",
            "modes": ["t2v", "v2v", "i2v"],
            "durations": [1],
            "prompts_pool": "nature",
            "parameters": {"width": 512},
            "source_files": {
                "v2v": "/data/videos/gradient.mp4",
                "i2v": "/data/videos/red.mp4",
            },
        }
    ]
    scenarios = expand_scenario_matrix(defs, graphs_dir=Path("/nonexistent"))
    by_mode = {s.mode: s for s in scenarios}
    assert "source_name" not in by_mode["t2v"].parameters
    assert by_mode["v2v"].parameters["source_name"] == "/data/videos/gradient.mp4"
    assert by_mode["i2v"].parameters["source_name"] == "/data/videos/red.mp4"


def test_scenario_duration_class():
    base = dict(pipeline="longlive", mode="t2v", graph=None, prompts_pool="nature", parameters={})
    assert Scenario(name="t", duration_mins=1, **base).duration_class == "short"
    assert Scenario(name="t", duration_mins=5, **base).duration_class == "mid"
    assert Scenario(name="t", duration_mins=15, **base).duration_class == "long"


def test_build_session_body_t2v():
    s = Scenario(
        name="longlive_t2v_1m",
        pipeline="longlive",
        mode="t2v",
        duration_mins=1,
        graph=None,
        prompts_pool="nature",
        parameters={"width": 512, "height": 512},
    )
    body = build_session_body(s, "a mountain lake")
    assert body["pipeline_id"] == "longlive"
    assert body["input_mode"] == "text"
    assert body["prompts"] == [{"text": "a mountain lake", "weight": 100}]


def test_build_session_body_v2v_with_source():
    s = Scenario(
        name="longlive_v2v_5m",
        pipeline="longlive",
        mode="v2v",
        duration_mins=5,
        graph=None,
        prompts_pool="nature",
        parameters={"width": 512, "noise_scale": 0.7, "source_name": "/data/videos/test.mp4"},
    )
    body = build_session_body(s, "ocean waves")
    assert body["input_mode"] == "video"
    assert body["input_source"]["source_name"] == "/data/videos/test.mp4"


def test_build_session_body_graph():
    graph = {
        "nodes": [
            {"id": "input", "type": "source"},
            {"id": "longlive", "type": "pipeline", "pipeline_id": "longlive"},
            {"id": "output", "type": "sink"},
        ],
        "edges": [],
    }
    s = Scenario(
        name="chain_v2v_5m",
        pipeline="longlive+rife",
        mode="v2v",
        duration_mins=5,
        graph=graph,
        prompts_pool="nature",
        parameters={},
    )
    body = build_session_body(s, "ocean waves")
    assert "graph" in body
    assert body["input_mode"] == "video"
    assert "pipeline_id" not in body


def test_load_prompt_pool(tmp_path: Path):
    d = tmp_path / "prompts"
    d.mkdir()
    (d / "nature.yaml").write_text("prompts:\n  - 'lake'\n  - 'ocean'\n  - 'forest'\n")
    pool = load_prompt_pool("nature", d)
    assert len(pool) == 3
    assert "lake" in pool


def test_load_prompt_pool_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_prompt_pool("nonexistent", tmp_path)
