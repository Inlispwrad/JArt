"""模板：官方预设（src/jart/presets/*.json，设计案 §3）和用户自己的模板文件。"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

from .issues import JArtError
from .workflow import Workflow, load_workflow


def preset_ids() -> list[str]:
    folder = resources.files("jart").joinpath("presets")
    return sorted(p.name.removesuffix(".json") for p in folder.iterdir() if p.name.endswith(".json"))


def load_preset(preset_id: str) -> Workflow:
    text = resources.files("jart").joinpath("presets", f"{preset_id}.json").read_text("utf-8")
    return Workflow.model_validate_json(text)


def load_template(ref: str) -> Workflow:
    """模板文件路径，或官方预设的名字（swap / chain / text）。"""
    if ref in preset_ids():
        return load_preset(ref)
    path = Path(ref)
    if path.suffix == ".json" or path.exists():
        return load_workflow(path)
    raise JArtError(f"找不到模板 {ref}；官方预设有：{'、'.join(preset_ids())}")
