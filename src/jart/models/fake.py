"""不联网的假模型：按请求尺寸画一张确定性的图，用量按粗略规则编造。只用于测试和演练流程。

透明背景时会模仿 2.5 的真实毛病（主体 Alpha 254、背景 Alpha 1–8 的淡雾），好让后处理有东西可清。
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable
from io import BytesIO

from PIL import Image, ImageChops, ImageDraw

from ..sizes import parse_size
from . import ModelError, ModelRequest, ModelResult

_QUALITY = {"low": 0.15, "medium": 0.4, "high": 1.0, "xhigh": 2.0, "max": 4.0, "auto": 1.0}
_PIL_FORMAT = {"png": "PNG", "webp": "WEBP", "jpeg": "JPEG"}


class FakeImages:
    name = "fake"

    def __init__(self, fail: Callable[[ModelRequest], ModelError | None] | None = None):
        self.calls: list[ModelRequest] = []
        self._fail = fail

    def generate(self, request: ModelRequest) -> ModelResult:
        self.calls.append(request)
        if self._fail is not None and (err := self._fail(request)) is not None:
            raise err
        width, height = parse_size(request.size) or (1024, 1024)
        transparent = request.background == "transparent"
        seed = hashlib.sha256(request.prompt.encode() + b"".join(img.data for img in request.images)).digest()
        images = [
            _render(width, height, transparent, seed + bytes([i]), _PIL_FORMAT[request.output_format])
            for i in range(request.n)
        ]
        text_tokens = len(request.prompt)
        image_tokens = sum(_input_tokens(img.data) for img in request.images)
        per_image = round(1180 * (width * height) / (2048 * 1024) * _QUALITY[request.quality])
        usage = {
            "input_tokens": text_tokens + image_tokens,
            "input_tokens_details": {"text_tokens": text_tokens, "image_tokens": image_tokens},
            "output_tokens": per_image * request.n,
            "output_tokens_details": {"image_tokens": per_image * request.n, "text_tokens": 0},
            "total_tokens": text_tokens + image_tokens + per_image * request.n,
        }
        meta = {
            "size": f"{width}x{height}",
            "quality": request.quality,
            "background": "transparent" if transparent else "opaque",
            "output_format": request.output_format,
        }
        return ModelResult(images=images, usage=usage, meta=meta)


def _input_tokens(data: bytes) -> int:
    with Image.open(BytesIO(data)) as im:
        return round(im.width * im.height / 1e6 * 1850)


def _render(width: int, height: int, transparent: bool, seed: bytes, fmt: str) -> bytes:
    rng = random.Random(seed)
    color = tuple(rng.randrange(40, 216) for _ in range(3))
    box = (width // 4, height // 4, 3 * width // 4, 3 * height // 4)
    if transparent:
        im = Image.new("RGBA", (width, height), (*color, 0))
        haze = Image.frombytes("L", (width, height), rng.randbytes(width * height))
        haze = haze.point([v % 8 + 1 if v < 8 else 0 for v in range(256)])  # 约 3% 的像素 Alpha 1–8
        body = Image.new("L", (width, height), 0)
        ImageDraw.Draw(body).ellipse(box, fill=254)
        im.putalpha(ImageChops.lighter(haze, body))
    else:
        im = Image.new("RGB", (width, height), color)
        ImageDraw.Draw(im).ellipse(box, fill=tuple(255 - c for c in color))
    buf = BytesIO()
    im.save(buf, format=fmt)
    return buf.getvalue()
