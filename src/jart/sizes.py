"""出图尺寸。

约束来自 OpenAI 文档（gpt-image-2 系列，2026-10-08 查）：宽高都是 16 的倍数；长边不超过 3840；
长短边之比不超过 3:1；总像素在 655,360 到 8,294,400 之间。其他模型只检查格式。
"""

from __future__ import annotations

import re

MAX_EDGE = 3840
MAX_RATIO = 3
MIN_PIXELS = 655_360
MAX_PIXELS = 8_294_400

_SIZE = re.compile(r"(\d+)x(\d+)")


def parse_size(size: str) -> tuple[int, int] | None:
    """'2048x1024' → (2048, 1024)；'auto' → None；格式不对抛 ValueError。"""
    if size == "auto":
        return None
    m = _SIZE.fullmatch(size)
    if not m:
        raise ValueError(f"尺寸写成 宽x高（如 2048x1024）或 auto，收到 {size!r}")
    return int(m.group(1)), int(m.group(2))


def check_size(size: str, model: str) -> None:
    """尺寸不能用时抛 ValueError，消息里列出所有问题。"""
    wh = parse_size(size)
    if wh is None or not model.startswith("gpt-image-2"):
        return
    w, h = wh
    problems = []
    if w % 16 or h % 16:
        problems.append("宽和高都须是 16 的倍数")
    if max(w, h) > MAX_EDGE:
        problems.append(f"边长不能超过 {MAX_EDGE}")
    if max(w, h) > MAX_RATIO * min(w, h):
        problems.append("长短边之比不能超过 3:1")
    if not MIN_PIXELS <= w * h <= MAX_PIXELS:
        problems.append(f"总像素须在 {MIN_PIXELS:,} 到 {MAX_PIXELS:,} 之间")
    if problems:
        raise ValueError(f"尺寸 {size} 不可用：" + "；".join(problems))
