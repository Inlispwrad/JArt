from __future__ import annotations

from pathlib import Path

import pytest

from jart.graph import compile_workflow
from jart.templates import load_preset, load_template, preset_ids
from jart.workflow import Workflow, bind, load_values


def graph(nodes):
    return compile_workflow(Workflow.model_validate({"name": "t", "nodes": nodes}))


def codes(g):
    return {i.code for i in g.issues}


@pytest.mark.parametrize("preset", preset_ids())
def test_presets_compile_cleanly(preset):
    g = compile_workflow(load_preset(preset))
    assert g.issues == []
    assert load_template(preset).name == load_preset(preset).name


def test_graph_errors():
    assert "E_NODE_TYPE" in codes(graph([{"id": "a", "type": "nope"}]))
    assert "E_PARAM" in codes(graph([{"id": "a", "type": "generate", "params": {"quality": "ultra"}}]))
    assert "E_DUP_ID" in codes(graph([{"id": "a", "type": "text"}, {"id": "a", "type": "text"}]))
    assert "E_REQUIRED" in codes(graph([{"id": "g", "type": "generate"}]))
    assert "E_PORT" in codes(graph([{"id": "t", "type": "text"}, {"id": "o", "type": "output", "inputs": {"img": "t"}}]))
    assert "E_SRC" in codes(graph([{"id": "o", "type": "output", "inputs": {"image": "ghost"}}]))
    assert "E_TYPE" in codes(graph([{"id": "t", "type": "text"}, {"id": "o", "type": "output", "inputs": {"image": "t"}}]))
    double = [
        {"id": "t", "type": "text", "params": {"text": "x"}},
        {"id": "s", "type": "structure", "params": {"style": "写在参数里"}, "inputs": {"style": "t"}},
    ]
    assert "E_DOUBLE" in codes(graph(double))


def test_cycle_is_detected():
    nodes = [
        {"id": "a", "type": "postprocess", "inputs": {"image": "b"}},
        {"id": "b", "type": "postprocess", "inputs": {"image": "a"}},
    ]
    assert "E_CYCLE" in codes(graph(nodes))


def test_order_follows_dependencies_and_default_ports():
    g = graph([
        {"id": "o", "type": "output", "inputs": {"image": "g.image"}},
        {"id": "g", "type": "generate", "inputs": {"prompt": "t"}},
        {"id": "t", "type": "text", "params": {"text": "x"}},
    ])
    assert g.issues == []
    assert g.order == ["t", "g", "o"]
    assert g.nodes["g"].inputs == {"prompt": ("t", "text")}
    assert g.ancestors("o") == {"g", "t"}


def test_bind_reports_missing_and_unknown_inputs():
    _, issues = bind(load_preset("text"), {"nope": "x"}, base_dir=Path("."))
    assert {i.code for i in issues} == {"E_UNKNOWN_INPUT", "E_MISSING_INPUT"}
    missing = sorted(i.message for i in issues if i.code == "E_MISSING_INPUT")
    assert missing == ["缺少输入：内容（content）", "缺少输入：画法段（style）"]


def test_bind_fills_params_without_touching_the_template(tmp_path):
    preset = load_preset("swap")
    values = {
        "char": "char.png",
        "item": {"path": "dagger.png", "revision": 1},
        "ref": "asset:abcdef12",
        "item_role": "匕首设定",
        "style": "画法",
        "replace": "第一行\n\n第二行",
        "action": "动作",
    }
    wf, issues = bind(preset, values, base_dir=tmp_path)
    assert issues == []
    s = wf.node("s").params
    assert s["images"][1]["role"] == "匕首设定"
    assert s["replace"] == ["第一行", "第二行"]
    assert wf.node("char").params == {"path": str(tmp_path / "char.png"), "revision": 0}
    assert wf.node("item").params == {"path": str(tmp_path / "dagger.png"), "revision": 1}
    assert wf.node("ref").params == {"asset": "abcdef12", "revision": 0}
    assert wf.node("gen").params["size"] == "2048x1024"  # 插槽默认值
    assert "role" not in preset.node("s").params["images"][1]


def test_bind_rejects_wrong_value_types(tmp_path):
    _, issues = bind(load_preset("text"), {"style": 3, "content": "x"}, base_dir=tmp_path)
    assert [i.code for i in issues] == ["E_BAD_INPUT"]


def test_load_values_toml(tmp_path):
    path = tmp_path / "inputs.toml"
    path.write_text('style = """\n第一段\n第二段\n"""\nreplace = ["一", "二"]\n', "utf-8")
    assert load_values(path) == {"style": "第一段\n第二段\n", "replace": ["一", "二"]}
