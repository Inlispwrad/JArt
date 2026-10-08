"""工作流 / 模板文件（JSON）与输入绑定。

模板只存结构和文字（设计案 §3）；图片、要换的文字通过「插槽」（slots）在运行时填进去。
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .issues import Issue, JArtError, error


class Slot(BaseModel):
    """模板暴露的一个输入。param 是节点参数里的路径，如 text、images.1.role；图像插槽填进图像节点。"""

    model_config = ConfigDict(extra="forbid")

    node: str
    param: str | None = None
    kind: Literal["text", "lines", "image"] = "text"
    label: str
    required: bool = True
    default: Any = None
    hint: str | None = None

    @model_validator(mode="after")
    def _param(self):
        if self.kind != "image" and not self.param:
            raise ValueError("文字插槽要写 param")
        return self


class NodeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z0-9_\-]+$")
    type: str
    title: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, str] = Field(default_factory=dict)  # 端口 → "节点" 或 "节点.端口"


class Workflow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    name: str
    description: str = ""
    language: str = "zh"  # 提示词语言，模板级设定（设计案 §6）
    format: dict[str, Any] | None = None  # 覆盖该语言内置的固定字句，见 structure.PromptFormat
    slots: dict[str, Slot] = Field(default_factory=dict)
    nodes: list[NodeSpec]

    def node(self, node_id: str) -> NodeSpec | None:
        return next((n for n in self.nodes if n.id == node_id), None)


def load_workflow(path: Path) -> Workflow:
    try:
        return Workflow.model_validate_json(Path(path).read_text("utf-8"))
    except FileNotFoundError as e:
        raise JArtError(f"找不到模板：{path}") from e
    except ValidationError as e:
        raise JArtError(f"模板格式不对（{path}）：\n{e}") from e


def bind(wf: Workflow, values: Mapping[str, Any], *, base_dir: Path) -> tuple[Workflow, list[Issue]]:
    """把插槽的值填进节点参数，返回新的工作流。图片的相对路径按 base_dir 解析。"""
    wf = wf.model_copy(deep=True)
    issues: list[Issue] = []
    for key in values:
        if key not in wf.slots:
            issues.append(error("E_UNKNOWN_INPUT", f"模板没有「{key}」这个输入"))
    for key, slot in wf.slots.items():
        node = wf.node(slot.node)
        if node is None:
            issues.append(error("E_SLOT_NODE", f"插槽 {key} 指向不存在的节点 {slot.node}"))
            continue
        value = values.get(key, slot.default)
        if value is None or value == "" or value == []:
            if slot.required:
                issues.append(error("E_MISSING_INPUT", f"缺少输入：{slot.label}（{key}）", slot.node))
            continue
        try:
            if slot.kind == "image":
                if node.type != "image":
                    raise ValueError(f"图像插槽要指向图像节点，{slot.node} 是 {node.type}")
                node.params.pop("asset", None)
                node.params.pop("path", None)
                node.params.update(_image_value(value, base_dir))
            else:
                _set_path(node.params, slot.param, _text_value(slot.kind, value))
        except ValueError as e:
            issues.append(error("E_BAD_INPUT", f"{slot.label}（{key}）：{e}", slot.node))
    return wf, issues


def load_values(path: Path) -> dict[str, Any]:
    """读输入文件（.toml 或 .json）。图片相对路径按文件所在目录解析由调用方处理。"""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except FileNotFoundError as e:
        raise JArtError(f"找不到输入文件：{path}") from e
    if path.suffix == ".toml":
        import tomllib

        try:
            return tomllib.loads(raw.decode("utf-8"))
        except tomllib.TOMLDecodeError as e:
            raise JArtError(f"输入文件格式不对（{path}）：{e}") from e
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise JArtError(f"输入文件格式不对（{path}）：{e}") from e
    if not isinstance(data, dict):
        raise JArtError("输入文件顶层要是 键 = 值")
    return data


def _text_value(kind: str, value: Any) -> Any:
    if kind == "text":
        if not isinstance(value, str):
            raise ValueError("要填文字")
        return value
    if isinstance(value, str):
        return [line for line in value.splitlines() if line.strip()]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return list(value)
    raise ValueError("要填文字列表")


def _image_value(value: Any, base_dir: Path) -> dict[str, Any]:
    revision = 0
    ref = value
    if isinstance(value, dict):
        revision = value.get("revision", 0)
        if not isinstance(revision, int) or revision < 0:
            raise ValueError("revision 要是非负整数")
        ref = f"asset:{value['asset']}" if value.get("asset") else value.get("path")
    if not isinstance(ref, str) or not ref:
        raise ValueError("要填图片路径，或 asset:素材编号")
    if ref.startswith("asset:"):
        return {"asset": ref.removeprefix("asset:"), "revision": revision}
    path = Path(ref).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return {"path": str(path), "revision": revision}


def _set_path(params: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cur: Any = params
    try:
        for i, key in enumerate(keys[:-1]):
            if isinstance(cur, list):
                cur = cur[int(key)]
            else:
                cur = cur.setdefault(key, [] if keys[i + 1].isdigit() else {})
        last = keys[-1]
        if isinstance(cur, list):
            cur[int(last)] = copy.deepcopy(value)
        else:
            cur[last] = copy.deepcopy(value)
    except (IndexError, ValueError, TypeError, AttributeError) as e:
        raise ValueError(f"参数路径 {path} 不存在") from e
