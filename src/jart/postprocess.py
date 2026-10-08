"""后处理：透明清理、裁切适配尺寸（设计案 §2、§6；docs/API实测要点.md「透明背景」）。

gpt-image-2.5 的透明输出：主体内部 Alpha 是 253/254（没有 255），背景散着 Alpha 1–8 的彩色淡雾。
默认 ≤8 归 0、≥250 归 255，中间值（边缘抗锯齿）保留。原图在素材库里另存，不被覆盖。
"""

from __future__ import annotations

from typing import Any

from PIL import Image, ImageChops, ImageFilter, ImageOps

from .nodes import AlphaCleanup, Resize
from .store import has_alpha


def clean_alpha(im: Image.Image, low: int = 8, high: int = 250, edge_px: int | None = None) -> tuple[Image.Image, dict[str, Any]]:
    if not has_alpha(im):
        return im, {"skipped": "没有透明通道"}
    if im.mode not in ("RGBA", "LA"):
        im = im.convert("RGBA")
    alpha = im.getchannel("A")
    before = alpha.histogram()
    cleaned = alpha.point([0 if v <= low else 255 if v >= high else v for v in range(256)])
    far_removed = 0
    if edge_px:
        solid = cleaned.point([255 if v == 255 else 0 for v in range(256)])
        near = solid.filter(ImageFilter.MaxFilter(2 * edge_px + 1))
        partial = cleaned.point([255 if 0 < v < 255 else 0 for v in range(256)])
        far = ImageChops.subtract(partial, near)  # 中间值且离主体超过 edge_px 的像素
        far_removed = far.histogram()[255]
        cleaned = Image.composite(Image.new("L", cleaned.size, 0), cleaned, far)
    out = im.copy()
    out.putalpha(cleaned)
    after = cleaned.histogram()
    stats = {
        "pixels": im.width * im.height,
        "haze_to_0": sum(before[1 : low + 1]),
        "body_to_255": sum(before[high:255]),
        "far_partial_to_0": far_removed,
        "partial_left": sum(after[1:255]),
    }
    return out, stats


def fit(im: Image.Image, width: int, height: int, mode: str = "cover") -> Image.Image:
    """缩放到目标尺寸。RGBA 缩放时 Pillow 会先预乘 Alpha，边缘不会发黑。"""
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGBA" if has_alpha(im) else "RGB")
    if mode == "cover":
        return ImageOps.fit(im, (width, height), method=Image.Resampling.LANCZOS)
    scaled = ImageOps.contain(im, (width, height), method=Image.Resampling.LANCZOS).convert("RGBA")
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    canvas.paste(scaled, ((width - scaled.width) // 2, (height - scaled.height) // 2))
    return canvas


def apply(im: Image.Image, alpha: AlphaCleanup | None, resize: Resize | None) -> tuple[Image.Image, dict[str, Any]]:
    """先清理透明（在原分辨率上），再缩放。"""
    stats: dict[str, Any] = {}
    if alpha is not None:
        im, stats["alpha"] = clean_alpha(im, alpha.low, alpha.high, alpha.edge_px)
    if resize is not None:
        im = fit(im, resize.width, resize.height, resize.mode)
        stats["resize"] = {"width": resize.width, "height": resize.height, "mode": resize.mode}
    return im, stats
