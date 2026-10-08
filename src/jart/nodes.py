"""节点类型：参数和端口。执行逻辑在 engine.py。

| 类型 | 输入 | 输出 |
|---|---|---|
| text 文本 | — | text |
| image 图像 | — | image |
| structure 结构 | 每张声明的图；各文字字段（可选，连线代替参数） | request |
| preview 预览 | request | request（原样传下去） |
| generate 生成 | prompt：text 或 request | image（第一张） |
| postprocess 后处理 | image | image |
| output 输出 | image | — |
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .sizes import check_size
from .structure import LIST_FIELDS, TEXT_FIELDS, StructureParams

DEFAULT_MODEL = "gpt-image-2.5-sunburst"


@dataclass(frozen=True)
class Port:
    types: tuple[str, ...]
    required: bool = False


class TextParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = ""


class ImageParams(BaseModel):
    """asset：素材库编号（sha256 或其前缀）；path：本地文件，运行时导入素材库。"""

    model_config = ConfigDict(extra="forbid")

    asset: str | None = None
    path: str | None = None
    revision: int = Field(0, ge=0)

    @model_validator(mode="after")
    def _one_source(self):
        if self.asset and self.path:
            raise ValueError("asset 和 path 只能写一个")
        return self


class PreviewParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GenerateParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = DEFAULT_MODEL
    size: str = "1024x1024"
    quality: Literal["low", "medium", "high", "xhigh", "max", "auto"] = "high"
    n: int = Field(1, ge=1, le=10)
    output_format: Literal["png", "webp", "jpeg"] = "png"
    background: Literal["transparent", "opaque", "auto"] | None = None
    output_compression: int | None = Field(None, ge=0, le=100)
    moderation: Literal["auto", "low"] | None = None
    pin: str | None = None  # 采用过的结果（素材编号）；设了就不调模型，直接输出它

    @model_validator(mode="after")
    def _check(self):
        check_size(self.size, self.model)
        if self.background == "transparent" and self.output_format == "jpeg":
            raise ValueError("透明背景要用 png 或 webp")
        if self.output_compression is not None and self.output_format == "png":
            raise ValueError("output_compression 只用于 webp / jpeg")
        if self.quality in ("xhigh", "max") and not self.model.startswith("gpt-image-2.5"):
            raise ValueError(f"{self.quality} 质量档只有 gpt-image-2.5 支持")
        return self


class AlphaCleanup(BaseModel):
    """≤low 归 0（淡雾），≥high 归 255（主体）；edge_px 设了就只在主体边缘这么宽的范围内保留中间值。"""

    model_config = ConfigDict(extra="forbid")

    low: int = Field(8, ge=0, le=254)
    high: int = Field(250, ge=1, le=255)
    edge_px: int | None = Field(None, ge=1, le=16)

    @model_validator(mode="after")
    def _order(self):
        if self.low >= self.high:
            raise ValueError("low 要小于 high")
        return self


class Resize(BaseModel):
    """cover：等比缩放后居中裁掉多余部分；contain：等比缩放放进去，空白处透明。"""

    model_config = ConfigDict(extra="forbid")

    width: int = Field(ge=1, le=8192)
    height: int = Field(ge=1, le=8192)
    mode: Literal["cover", "contain"] = "cover"


class PostprocessParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alpha: AlphaCleanup | None = Field(default_factory=AlphaCleanup)  # 默认开启（设计案 §6）
    resize: Resize | None = None


class OutputParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(None, pattern=r"^[^/\\:*?\"<>|]+$")


@dataclass(frozen=True)
class NodeType:
    name: str
    title: str
    params: type[BaseModel]
    outputs: dict[str, str]
    input_ports: Callable[[BaseModel], dict[str, Port]]

    @property
    def default_output(self) -> str | None:
        return next(iter(self.outputs), None)


def _structure_ports(p: StructureParams) -> dict[str, Port]:
    ports = {f: Port(("text",)) for f in (*TEXT_FIELDS, *LIST_FIELDS)}
    for d in p.images:
        ports[d.name] = Port(("image",), required=True)
    return ports


NODE_TYPES = {
    t.name: t
    for t in (
        NodeType("text", "文本", TextParams, {"text": "text"}, lambda p: {}),
        NodeType("image", "图像", ImageParams, {"image": "image"}, lambda p: {}),
        NodeType("structure", "结构", StructureParams, {"request": "request"}, _structure_ports),
        NodeType("preview", "预览", PreviewParams, {"request": "request"}, lambda p: {"request": Port(("request",), True)}),
        NodeType("generate", "生成", GenerateParams, {"image": "image"}, lambda p: {"prompt": Port(("text", "request"), True)}),
        NodeType("postprocess", "后处理", PostprocessParams, {"image": "image"}, lambda p: {"image": Port(("image",), True)}),
        NodeType("output", "输出", OutputParams, {}, lambda p: {"image": Port(("image",), True)}),
    )
}
