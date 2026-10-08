from __future__ import annotations

import pytest

from jart.issues import JArtError
from jart.structure import LIST_FIELDS, TEXT_FIELDS, StructureParams, assemble, resolve_format
from jart.values import ImageValue

ZH = resolve_format("zh")


def img(tag="a", revision=0, size=(100, 100)):
    return ImageValue(asset=tag * 64, width=size[0], height=size[1], revision=revision)


def build(params: dict, images: dict | None = None, fmt=ZH):
    p = StructureParams.model_validate(params)
    texts = {f: getattr(p, f) for f in TEXT_FIELDS}
    lists = {f: getattr(p, f) for f in LIST_FIELDS}
    return assemble("s", p, images or {}, texts, lists, fmt)


def codes(req):
    return {i.code for i in req.issues}


SWAP = {
    "images": [
        {"name": "char", "role": "人设"},
        {"name": "item", "role": "匕首设定"},
        {"name": "ref", "kind": "content_ref"},
    ],
    "style": "画法段原文",
    "replace": ["把{{ref}}里的刺客换成{{char}}。", "把{{ref}}里的短刀换成{{item}}。", "画面里没有别的主体。"],
    "action": "刺客从画面左侧冲向右侧，匕首刺向目标背后。",
    "size_note": "最终显示在约 130×65 的小图窗、一眼识别。",
}
SWAP_IMAGES = {"char": img("1"), "item": img("2"), "ref": img("3")}


def test_single_reference_uses_the_tested_phrase():
    req = build({"images": [{"name": "char", "role": "人设"}], "style": "画法段", "content": "她站在雨里。"}, {"char": img()})
    assert req.prompt == "参考图是人设。\n\n画法段\n\n她站在雨里。"
    assert not req.issues


def test_swap_numbers_images_in_declaration_order():
    req = build(SWAP, SWAP_IMAGES)
    assert req.prompt == (
        "图1是人设。\n图2是匕首设定。\n图3是内容参考图。\n\n"
        "画法段原文\n\n"
        "把图3里的刺客换成图1。\n把图3里的短刀换成图2。\n画面里没有别的主体。\n\n"
        "刺客从画面左侧冲向右侧，匕首刺向目标背后。\n\n"
        "最终显示在约 130×65 的小图窗、一眼识别。"
    )
    assert [(ri.index, ri.name) for ri in req.images] == [(1, "char"), (2, "item"), (3, "ref")]
    assert not req.issues


def test_user_text_is_never_rewritten():
    text = "  保留原样：英文 ABC、标点「」……\n第二行  "
    req = build({"content": text, "style": "画法"})
    assert req.prompt == "画法\n\n" + text.strip()


def test_content_ref_requires_replace_and_action():
    req = build({**SWAP, "replace": [], "action": ""}, SWAP_IMAGES)
    assert {"E_NEED_REPLACE", "E_NEED_ACTION"} <= codes(req)


def test_setting_image_missing_from_replace_warns():
    req = build({**SWAP, "replace": ["把{{ref}}里的刺客换成{{char}}。"]}, SWAP_IMAGES)
    assert "W_SETTING_UNUSED" in codes(req)
    literal = build({**SWAP, "replace": ["把图3里的刺客换成图1。", "短刀换成图2。"]}, SWAP_IMAGES)
    assert "W_SETTING_UNUSED" not in codes(literal)


def test_missing_style_warns_and_camera_with_reference_warns():
    req = build({**SWAP, "style": "", "camera": "越肩视角"}, SWAP_IMAGES)
    assert {"W_NO_STYLE", "W_CAMERA_WITH_REF"} <= codes(req)
    no_ref = build({"content": "一只猫。", "camera": "越肩视角", "style": "画法"})
    assert "W_CAMERA_WITH_REF" not in codes(no_ref)


def test_unknown_placeholder_is_an_error():
    req = build({**SWAP, "action": "{{nobody}} 冲过去"}, SWAP_IMAGES)
    assert "E_PLACEHOLDER" in codes(req)


def test_setting_image_needs_a_role():
    req = build({"images": [{"name": "x"}], "content": "画面", "style": "画法"}, {"x": img()})
    assert "E_ROLE" in codes(req)


def test_revision_limit():
    req = build(SWAP, {**SWAP_IMAGES, "char": img("1", revision=2)})
    assert "E_REVISION" in codes(req)
    ok = build(SWAP, {**SWAP_IMAGES, "char": img("1", revision=1)})
    assert "E_REVISION" not in codes(ok)


def test_too_many_images():
    params = {"images": [{"name": f"i{k}", "role": "设定"} for k in range(17)], "content": "x", "style": "y"}
    req = build(params, {f"i{k}": img() for k in range(17)})
    assert "E_TOO_MANY_IMAGES" in codes(req)


def test_fix_mode_writes_only_numbered_fixes():
    params = {"images": [{"name": "t", "kind": "edit_target"}, {"name": "m", "kind": "mask"}], "fixes": ["箭尾画成了两根，应为一根。", "左手少了一根手指，应为五根。"]}
    req = build(params, {"t": img("t"), "m": img("m")})
    assert req.prompt == "1. 箭尾画成了两根，应为一根。\n2. 左手少了一根手指，应为五根。"
    assert req.mask is not None and req.mask.index is None
    assert not req.issues


def test_fix_mode_with_extra_reference_declares_roles():
    params = {
        "images": [{"name": "t", "kind": "edit_target"}, {"name": "char", "role": "人设"}, {"name": "m", "kind": "mask"}],
        "fixes": ["脸画错了，应按{{char}}的人设画。"],
    }
    req = build(params, {"t": img("t"), "char": img("c"), "m": img("m")})
    assert req.prompt == "图1是要修改的图。\n图2是人设。\n\n1. 脸画错了，应按图2的人设画。"


def test_fix_mode_extras_warn():
    params = {"images": [{"name": "t", "kind": "edit_target"}], "fixes": ["错了，应该这样画。"], "style": "画法段"}
    assert "W_FIX_EXTRA" in codes(build(params, {"t": img()}))


def test_mask_rules():
    not_first = {"images": [{"name": "char", "role": "人设"}, {"name": "t", "kind": "edit_target"}, {"name": "m", "kind": "mask"}], "fixes": ["x"]}
    assert "E_MASK_TARGET" in codes(build(not_first, {"char": img("c"), "t": img("t"), "m": img("m")}))
    wrong_size = {"images": [{"name": "t", "kind": "edit_target"}, {"name": "m", "kind": "mask"}], "fixes": ["x"]}
    assert "E_MASK_SIZE" in codes(build(wrong_size, {"t": img("t", size=(10, 10)), "m": img("m", size=(20, 20))}))
    by_name = {"images": [{"name": "t", "kind": "edit_target"}, {"name": "m", "kind": "mask"}], "fixes": ["按{{m}}改"]}
    assert "E_PLACEHOLDER_MASK" in codes(build(by_name, {"t": img("t"), "m": img("m")}))


def test_empty_prompt_is_an_error():
    assert "E_EMPTY_PROMPT" in codes(build({}))


def test_param_validation():
    with pytest.raises(ValueError):
        StructureParams.model_validate({"images": [{"name": "a"}, {"name": "a"}]})
    with pytest.raises(ValueError):
        StructureParams.model_validate({"images": [{"name": "style"}]})


def test_format_override_and_unknown_language():
    fmt = resolve_format("zh", {"role_multi": "画像{n}：{role}"})
    req = build(SWAP, SWAP_IMAGES, fmt)
    assert req.prompt.startswith("画像1：人设\n画像2：匕首设定\n画像3：内容参考图")
    with pytest.raises(JArtError):
        resolve_format("en")
    with pytest.raises(JArtError):
        resolve_format("zh", {"nonsense": 1})
