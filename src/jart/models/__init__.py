"""模型适配器：把一次出图请求原样发给模型方，返回图片和用量。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class InputImage:
    name: str
    data: bytes
    mime: str  # image/png | image/jpeg | image/webp


@dataclass(frozen=True)
class ModelRequest:
    model: str
    prompt: str
    size: str
    quality: str
    n: int = 1
    output_format: str = "png"
    background: str | None = None
    output_compression: int | None = None
    moderation: str | None = None
    user: str | None = None  # 匿名用户编号（哈希过），P2 起每次调用都带
    images: tuple[InputImage, ...] = ()  # 顺序即提示里的「图1、图2…」
    mask: InputImage | None = None  # 对应第一张图；Alpha=0 处可改

    @property
    def endpoint(self) -> str:
        return "/v1/images/edits" if self.images else "/v1/images/generations"

    def fields(self) -> dict[str, Any]:
        """除图片外发给接口的全部字段。"""
        out: dict[str, Any] = {
            "model": self.model,
            "prompt": self.prompt,
            "size": self.size,
            "quality": self.quality,
            "n": self.n,
            "output_format": self.output_format,
        }
        for key in ("background", "output_compression", "moderation", "user"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out


@dataclass
class ModelResult:
    images: list[bytes]
    usage: dict[str, Any] | None
    meta: dict[str, Any] = field(default_factory=dict)  # 接口回报的 size / quality / background 等


class ModelError(Exception):
    def __init__(self, message: str, *, status: int | None = None, code: str | None = None, detail: dict | None = None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code
        self.detail = detail or {}

    def to_dict(self) -> dict[str, Any]:
        return {"message": self.message, "status": self.status, "code": self.code, "detail": self.detail}


class ImageModel(Protocol):
    name: str

    def generate(self, request: ModelRequest) -> ModelResult: ...
