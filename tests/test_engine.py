from __future__ import annotations

import pytest

from jart.engine import Engine
from jart.issues import JArtError
from jart.models import ModelError
from jart.models.fake import FakeImages
from jart.templates import load_preset
from jart.workflow import Workflow, bind


def chain_values(make_png):
    return {
        "char_desc": "一位刺客的人设图，全身，白底。",
        "item": str(make_png("dagger.png")),
        "item_role": "匕首设定",
        "ref": str(make_png("ref.png", (128, 64), (0, 0, 200, 255))),
        "style": "画法段",
        "replace": ["把{{ref}}里的刺客换成{{char}}。", "把{{ref}}里的短刀换成{{item}}。"],
        "action": "刺客冲向右侧，匕首刺向目标背后。",
        "background": "transparent",
    }


def bound(preset, values, tmp_path):
    wf, issues = bind(load_preset(preset), values, base_dir=tmp_path)
    assert not issues, issues
    return wf


def codes(issues):
    return {i.code if hasattr(i, "code") else i["code"] for i in issues}


def test_plan_lists_every_call_without_calling_the_model(engine, fake, make_png, tmp_path):
    plan = engine.plan(bound("chain", chain_values(make_png), tmp_path))
    assert plan.ok, plan.issues
    assert [c.node for c in plan.calls] == ["gen_char", "gen"]
    first, second = plan.calls
    assert (first.endpoint, first.prompt) == ("generations", "一位刺客的人设图，全身，白底。")
    assert second.endpoint == "edits"
    assert second.images[0]["source"] == "pending:gen_char"
    assert second.prompt.startswith("图1是人设。\n图2是匕首设定。\n图3是内容参考图。")
    assert all(c.estimate and c.estimate.usd for c in plan.calls)
    assert fake.calls == []


def test_chain_feeds_generated_character_downstream(engine, fake, store, make_png, tmp_path):
    wf = bound("chain", chain_values(make_png), tmp_path)
    plan = engine.plan(wf)
    record = engine.run(wf, plan)
    assert record["status"] == "succeeded", record["issues"]
    first, second = record["calls"]
    char = first["outputs"][0]["sha256"]
    assert second["request"]["images"][0]["sha256"] == char
    assert fake.calls[1].images[0].data == store.read(char)
    assert fake.calls[1].prompt == plan.calls[1].prompt == second["request"]["prompt"]
    assert record["plan_hash"] == plan.hash
    assert record["cost_usd"] == pytest.approx(first["cost_usd"] + second["cost_usd"])
    run_dir = store.runs_dir / record["id"]
    assert {o["node"] for o in record["outputs"]} == {"out_char", "out"}
    assert all((run_dir / o["file"]).is_file() for o in record["outputs"])
    assert (run_dir / second["outputs"][0]["file"]).is_file()  # 原图另存
    pp = record["postprocess"][0]
    assert pp["stats"]["alpha"]["haze_to_0"] > 0 and pp["stats"]["alpha"]["body_to_255"] > 0
    assert store.meta(pp["input"])["origin"] == "generated"
    assert store.meta(pp["output"])["parents"] == [pp["input"]]
    assert store.load_run(record["id"][:15])["id"] == record["id"]


def test_rule_errors_block_the_run(engine, fake, make_png, tmp_path):
    values = chain_values(make_png)
    values.pop("char_desc")
    wf = bound("swap", {**values, "char": str(make_png("char.png"))}, tmp_path)
    wf.node("s").params["action"] = ""
    plan = engine.plan(wf)
    assert not plan.ok and "E_NEED_ACTION" in codes(plan.issues)
    with pytest.raises(JArtError):
        engine.run(wf, plan)
    assert fake.calls == []


def test_changed_workflow_is_refused_before_any_call(engine, fake, make_png, tmp_path):
    wf = bound("chain", chain_values(make_png), tmp_path)
    plan = engine.plan(wf)
    wf.node("s").params["action"] = "改过的动作。"
    with pytest.raises(JArtError):
        engine.run(wf, plan)
    assert fake.calls == []


def test_until_runs_only_the_upstream_part(engine, fake, make_png, tmp_path):
    wf = bound("chain", chain_values(make_png), tmp_path)
    plan = engine.plan(wf, until="gen_char")
    assert [c.node for c in plan.calls] == ["gen_char"]
    record = engine.run(wf, plan, until="gen_char")
    assert record["status"] == "succeeded" and len(fake.calls) == 1


def test_pinned_result_is_reused_without_calling(engine, fake, make_png, tmp_path):
    wf = bound("chain", chain_values(make_png), tmp_path)
    first = engine.run(wf, engine.plan(wf, until="gen_char"), until="gen_char")
    adopted = first["calls"][0]["outputs"][0]["sha256"]
    wf.node("gen_char").params["pin"] = adopted[:10]
    plan = engine.plan(wf)
    assert plan.calls[0].pinned == adopted and plan.calls[0].estimate.usd == 0
    assert plan.calls[1].images[0]["source"] == adopted
    record = engine.run(wf, plan)
    assert record["status"] == "succeeded"
    assert record["calls"][0]["status"] == "pinned"
    assert len(fake.calls) == 2  # 第一次运行 1 次 + 这次只调成图
    assert record["calls"][1]["request"]["images"][0]["sha256"] == adopted


def test_model_failure_is_recorded_and_downstream_skipped(store, make_png, tmp_path):
    def fail(request):
        if request.images:
            return ModelError("Your request was rejected.", status=400, code="moderation_blocked")
        return None

    engine = Engine(store, FakeImages(fail=fail))
    wf = bound("chain", chain_values(make_png), tmp_path)
    record = engine.run(wf, engine.plan(wf))
    assert record["status"] == "failed"
    failed = record["calls"][1]
    assert failed["status"] == "failed" and failed["error"]["code"] == "moderation_blocked"
    assert failed["cost_usd"] == 0.0
    assert record["node_status"]["clean"] == "skipped" and record["node_status"]["out"] == "skipped"
    assert record["node_status"]["out_char"] == "ok"
    assert "E_MODEL" in codes(record["issues"])


def text_to_image(prompt="画一只猫。", **gen):
    return Workflow.model_validate({
        "name": "直出",
        "nodes": [
            {"id": "t", "type": "text", "params": {"text": prompt}},
            {"id": "g", "type": "generate", "params": gen, "inputs": {"prompt": "t"}},
            {"id": "o", "type": "output", "inputs": {"image": "g"}},
        ],
    })


def test_text_prompt_goes_straight_to_generations(engine, fake):
    wf = text_to_image(size="1024x1024", quality="low", n=2)
    record = engine.run(wf, engine.plan(wf))
    assert record["status"] == "succeeded"
    sent = fake.calls[0]
    assert (sent.endpoint, sent.prompt, sent.n, sent.images) == ("/v1/images/generations", "画一只猫。", 2, ())
    assert [o["file"] for o in record["calls"][0]["outputs"]] == ["out/g_1.png", "out/g_2.png"]


def test_empty_text_prompt_is_an_error(engine):
    plan = engine.plan(text_to_image(prompt="   "))
    assert "E_EMPTY_PROMPT" in codes(plan.issues)


def test_bad_generate_params_are_reported(engine):
    plan = engine.plan(text_to_image(size="1000x1000"))
    assert not plan.ok and "E_PARAM" in codes(plan.issues)
    plan = engine.plan(text_to_image(background="transparent", output_format="jpeg"))
    assert not plan.ok


def fix_workflow(target, mask, then_reuse=False):
    nodes = [
        {"id": "target", "type": "image", "params": target},
        {"id": "mask", "type": "image", "params": mask},
        {
            "id": "s",
            "type": "structure",
            "params": {"images": [{"name": "t", "kind": "edit_target"}, {"name": "m", "kind": "mask"}], "fixes": ["箭尾画成了两根，应为一根。"]},
            "inputs": {"t": "target", "m": "mask"},
        },
        {"id": "g", "type": "generate", "params": {"size": "1024x1024"}, "inputs": {"prompt": "s"}},
    ]
    if then_reuse:
        nodes += [
            {"id": "s2", "type": "structure", "params": {"images": [{"name": "x", "role": "人设"}], "content": "画面", "style": "画法"}, "inputs": {"x": "g"}},
            {"id": "g2", "type": "generate", "params": {"size": "1024x1024"}, "inputs": {"prompt": "s2"}},
        ]
    return Workflow.model_validate({"name": "修图", "nodes": nodes})


def test_mask_edit_sends_target_first_and_counts_revision(engine, fake, store, make_png):
    target = make_png("target.png", (1024, 1024))
    mask = make_png("mask.png", (1024, 1024), (0, 0, 0, 0))
    wf = fix_workflow({"path": str(target)}, {"path": str(mask)})
    record = engine.run(wf, engine.plan(wf))
    assert record["status"] == "succeeded", record["issues"]
    sent = fake.calls[0]
    assert sent.endpoint == "/v1/images/edits" and sent.mask is not None and len(sent.images) == 1
    assert sent.prompt == "1. 箭尾画成了两根，应为一根。"
    assert store.meta(record["calls"][0]["outputs"][0]["sha256"])["revision"] == 1


def test_twice_modified_image_cannot_be_reused(engine, make_png):
    target = make_png("target.png", (1024, 1024))
    mask = make_png("mask.png", (1024, 1024), (0, 0, 0, 0))
    wf = fix_workflow({"path": str(target), "revision": 1}, {"path": str(mask)}, then_reuse=True)
    plan = engine.plan(wf)
    assert "E_REVISION" in codes(plan.issues)  # g 的输出已被修改两次，不能再当参考图


def test_mask_must_be_png_with_alpha_and_same_size(engine, make_png):
    target = make_png("target.png", (1024, 1024))
    wf = fix_workflow({"path": str(target)}, {"path": str(make_png("rgb.png", (1024, 1024), mode="RGB"))})
    assert "E_MASK" in codes(engine.plan(wf).issues)
    wf = fix_workflow({"path": str(target)}, {"path": str(make_png("small.png", (512, 512), (0, 0, 0, 0)))})
    assert "E_MASK_SIZE" in codes(engine.plan(wf).issues)


def test_missing_image_source(engine):
    wf = Workflow.model_validate({
        "name": "缺图",
        "nodes": [
            {"id": "i", "type": "image"},
            {"id": "s", "type": "structure", "params": {"images": [{"name": "x", "role": "人设"}], "content": "c", "style": "s"}, "inputs": {"x": "i"}},
            {"id": "g", "type": "generate", "inputs": {"prompt": "s"}},
        ],
    })
    plan = engine.plan(wf)
    assert "E_NO_IMAGE" in codes(plan.issues) and plan.calls == []
