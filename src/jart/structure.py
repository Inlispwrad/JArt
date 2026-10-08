"""结构节点：按 docs/图像生成规则_gpt-image-2.5.md 把设定输入和内容指定拼成发给模型的提示词。

拼接是固定规则的纯函数：不经 LLM，不改写、不增补用户的文字（规则 §2「关闭任何自动增补」，设计案 §6）。
平台自己生成的文字只有两种：
  - 每张图一句角色声明：「图1是人设。」；只有一张图时用实测过的说法「参考图是人设。」
  - 修图条目的编号：「1. …」
图片编号由声明顺序决定。用户在文字里用 {{名字}} 指代图片，拼接时换成「图N」，增删图片后编号不会写错。
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .issues import Issue, JArtError, error, warning
from .values import ImageValue, PromptRequest, RequestImage

MAX_IMAGES = 16  # edits 接口最多 16 张输入图（官方文档）
MAX_PROMPT_CHARS = 32_000  # GPT image 模型提示词上限（官方文档）
MAX_SOURCE_REVISION = 1  # 参考图只能是原图或修改过一次的图（规则 §2）

ImageKind = Literal["setting", "content_ref", "action_ref", "size_frame", "edit_target", "mask"]

TEXT_FIELDS = ("objects", "style", "camera", "content", "action", "size_note")
LIST_FIELDS = ("replace", "fixes")
SECTIONS = ("roles", "objects", "style", "camera", "content", "replace", "action", "size_note", "fixes")

LABELS = {
    "objects": "物件设定",
    "style": "风格（画法段）",
    "camera": "镜头构图",
    "content": "内容",
    "replace": "物件指向替换",
    "action": "动作描述",
    "size_note": "目标尺寸",
    "fixes": "修图条目",
}

PLACEHOLDER = re.compile(r"\{\{\s*([^{}\s]+)\s*\}\}")


@dataclass(frozen=True)
class PromptFormat:
    """平台生成的固定字句。按模板语言选内置格式；模板的 format 可以逐项覆盖。"""

    role_single: str
    role_multi: str
    ref_single: str
    ref_multi: str
    fix_item: str
    default_roles: Mapping[str, str]
    sections: tuple[str, ...] = SECTIONS


BUILTIN_FORMATS = {
    "zh": PromptFormat(
        role_single="参考图是{role}。",
        role_multi="图{n}是{role}。",
        ref_single="参考图",
        ref_multi="图{n}",
        fix_item="{n}. {text}",
        default_roles={
            "content_ref": "内容参考图",
            "action_ref": "动作参考图",
            "size_frame": "尺寸线框",
            "edit_target": "要修改的图",
        },
    ),
}


def resolve_format(language: str, overrides: Mapping[str, Any] | None = None) -> PromptFormat:
    overrides = dict(overrides or {})
    known = {f.name for f in dataclasses.fields(PromptFormat)}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise JArtError(f"format 里有不认识的项：{'、'.join(unknown)}")
    if "sections" in overrides:
        sections = tuple(overrides["sections"])
        bad = sorted(set(sections) - set(SECTIONS))
        if bad:
            raise JArtError(f"format.sections 里有不认识的段：{'、'.join(bad)}")
        overrides["sections"] = sections
    base = BUILTIN_FORMATS.get(language)
    if base is None:
        missing = sorted(known - set(overrides) - {"sections"})
        if missing:
            raise JArtError(f"语言 {language} 没有内置格式，模板的 format 要写全：{'、'.join(missing)}")
        return PromptFormat(**overrides)
    return dataclasses.replace(base, **overrides)


class ImageDecl(BaseModel):
    """结构节点的一张输入图。声明顺序就是发给接口的顺序。"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[^{}\s.]+$")
    kind: ImageKind = "setting"
    role: str | None = None


class StructureParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    images: list[ImageDecl] = Field(default_factory=list)
    objects: str = ""
    style: str = ""
    camera: str = ""
    content: str = ""
    replace: list[str] = Field(default_factory=list)
    action: str = ""
    size_note: str = ""
    fixes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_names(self):
        names = [d.name for d in self.images]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"图像名字重复：{'、'.join(dupes)}")
        clash = sorted(set(names) & {*TEXT_FIELDS, *LIST_FIELDS})
        if clash:
            raise ValueError(f"图像名字不能和字段同名：{'、'.join(clash)}")
        return self


def assemble(
    node: str | None,
    params: StructureParams,
    images: Mapping[str, ImageValue],
    texts: Mapping[str, str],
    lists: Mapping[str, Sequence[str]],
    fmt: PromptFormat,
) -> PromptRequest:
    """拼出最终提示词并检查规则。texts / lists 是各字段的最终取值（参数或连线）。"""
    issues: list[Issue] = []
    ordered = [d for d in params.images if d.kind != "mask"]
    masks = [d for d in params.images if d.kind == "mask"]
    single = len(ordered) == 1
    numbers = {d.name: fmt.ref_single if single else fmt.ref_multi.format(n=i) for i, d in enumerate(ordered, 1)}
    mask_names = {d.name for d in masks}

    def fill(text: str, field: str) -> str:
        def sub(m: re.Match) -> str:
            name = m.group(1)
            if name in numbers:
                return numbers[name]
            if name in mask_names:
                issues.append(error("E_PLACEHOLDER_MASK", f"{LABELS[field]}里的 {m.group(0)} 指向遮罩；遮罩不编号，不能在文字里指代", node))
            else:
                issues.append(error("E_PLACEHOLDER", f"{LABELS[field]}里的 {m.group(0)} 没有对应的图像", node))
            return m.group(0)

        return PLACEHOLDER.sub(sub, text)

    req_images = []
    for i, d in enumerate(ordered, 1):
        role = (d.role or fmt.default_roles.get(d.kind, "")).strip()
        if not role:
            issues.append(error("E_ROLE", f"图像 {d.name} 要写一句它是什么（如「人设」）", node))
        req_images.append(RequestImage(index=i, name=d.name, kind=d.kind, role=role, image=images[d.name]))
    mask = None
    if masks:
        mask = RequestImage(index=None, name=masks[0].name, kind="mask", role="", image=images[masks[0].name])

    text = {f: (texts.get(f) or "").strip() for f in TEXT_FIELDS}
    lines = {f: [s.strip() for s in lists.get(f, ()) if s.strip()] for f in LIST_FIELDS}

    issues += _check_rules(node, req_images, len(masks), mask, text, lines, numbers)

    fix_mode = bool(lines["fixes"])
    role_lines = []
    if len(req_images) >= (2 if fix_mode else 1):
        template = fmt.role_single if single else fmt.role_multi
        role_lines = [template.format(n=ri.index, role=ri.role) for ri in req_images]
    parts = {
        "roles": "\n".join(role_lines),
        **{f: fill(text[f], f) for f in TEXT_FIELDS},
        "replace": "\n".join(fill(s, "replace") for s in lines["replace"]),
        "fixes": "\n".join(fmt.fix_item.format(n=i, text=fill(s, "fixes")) for i, s in enumerate(lines["fixes"], 1)),
    }
    prompt = "\n\n".join(parts[s] for s in fmt.sections if parts.get(s))
    if not prompt:
        issues.append(error("E_EMPTY_PROMPT", "提示词是空的", node))
    elif len(prompt) > MAX_PROMPT_CHARS:
        issues.append(error("E_PROMPT_TOO_LONG", f"提示词 {len(prompt)} 字，超过 {MAX_PROMPT_CHARS} 字上限", node))
    return PromptRequest(prompt=prompt, images=tuple(req_images), mask=mask, issues=tuple(issues))


def _check_rules(node, req_images, mask_count, mask, text, lines, numbers) -> list[Issue]:
    issues: list[Issue] = []
    kinds = {ri.kind for ri in req_images}
    targets = [ri for ri in req_images if ri.kind == "edit_target"]

    if len(req_images) > MAX_IMAGES:
        issues.append(error("E_TOO_MANY_IMAGES", f"参考图最多 {MAX_IMAGES} 张，现在 {len(req_images)} 张", node))
    if len(targets) > 1:
        issues.append(error("E_MULTI_TARGET", "只能有一张「要修改的图」", node))
    if mask_count > 1:
        issues.append(error("E_MULTI_MASK", "只能有一张遮罩", node))
    if mask is not None:
        if not req_images or req_images[0].kind != "edit_target":
            issues.append(error("E_MASK_TARGET", "用遮罩时，要修改的图必须排第一张（遮罩对应第一张图）", node))
        else:
            t, m = req_images[0].image, mask.image
            if None not in (t.width, t.height, m.width, m.height) and (t.width, t.height) != (m.width, m.height):
                issues.append(error("E_MASK_SIZE", f"遮罩 {m.width}×{m.height} 和要修改的图 {t.width}×{t.height} 尺寸不同", node))
    for ri in req_images:
        if ri.image.revision > MAX_SOURCE_REVISION:
            issues.append(error(
                "E_REVISION",
                f"{ri.name} 已经被修改过 {ri.image.revision} 次；参考图和修改对象只能是原图或修改过一次的图。"
                "请回到合规的源图，把已确定的修改合并成一次",
                node,
            ))

    if lines["fixes"]:
        if not targets:
            issues.append(error("E_FIX_TARGET", "写了修图条目，但没有「要修改的图」（kind=edit_target）", node))
        extra = [LABELS[f] for f in TEXT_FIELDS if text[f]] + ([LABELS["replace"]] if lines["replace"] else [])
        if extra:
            issues.append(warning("W_FIX_EXTRA", f"修图只写错在哪和正确画法；这些段落也会一起发出：{'、'.join(extra)}", node))
        return issues

    if "content_ref" in kinds:
        if not lines["replace"]:
            issues.append(error("E_NEED_REPLACE", "有内容参考图时必须写物件指向替换：参考图里哪个元素换成哪张设定图、保留哪些、画面里有没有别的主体", node))
        if not text["action"]:
            issues.append(error("E_NEED_ACTION", "有内容参考图时必须写画面内容的动作描述：谁、从哪到哪、做什么动作、作用在哪里", node))
        if lines["replace"]:
            joined = "\n".join(lines["replace"])
            named = set(PLACEHOLDER.findall(joined))
            for ri in req_images:
                literal = re.search(re.escape(numbers[ri.name]) + r"(?!\d)", joined)
                if ri.kind == "setting" and ri.name not in named and not literal:
                    issues.append(warning(
                        "W_SETTING_UNUSED",
                        f"设定图 {ri.name}（{ri.role}）没有出现在替换里；替换写不全时，参考图里的元素可能被保留成别的东西",
                        node,
                    ))
    if not text["style"]:
        issues.append(warning("W_NO_STYLE", "没有接风格（画法段）", node))
    if text["camera"] and kinds & {"content_ref", "action_ref"}:
        issues.append(warning("W_CAMERA_WITH_REF", "已有内容 / 动作参考图时，透视交给参考图，不再叠加镜头词", node))
    return issues
