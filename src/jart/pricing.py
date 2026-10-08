"""价格、实际花费与运行前的预估。

价格：OpenAI 模型页 gpt-image-2.5-sunburst（2026-10-08 查），每百万 token：文本输入 $5、图像输入 $8、
图像输出 $30；文本输出不计费；flare 同价。直调 Images API 不享受缓存价。

预估：官方没有给 2.5 的输出 token 公式。这里用实测数据——先用用户 2026-10-07 的实测
（docs/API实测要点.md），之后用本机历次真实调用的用量；同质量档没有同尺寸数据时按面积外推。
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .sizes import parse_size


@dataclass(frozen=True)
class Price:
    """美元 / 百万 token。"""

    text_in: float
    image_in: float
    image_out: float


_GPT_IMAGE_2 = Price(text_in=5.0, image_in=8.0, image_out=30.0)
PRICES = {
    "gpt-image-2.5-sunburst": _GPT_IMAGE_2,
    "gpt-image-2.5-flare": _GPT_IMAGE_2,
    "gpt-image-2": _GPT_IMAGE_2,
}

# 用户实测：sunburst 2048×1024 high 每张 1180 个输出 token；1086×1448 立绘输入约 2904 token
SEED_OUTPUT = {("gpt-image-2.5-sunburst", "high"): [((2048, 1024), 1180.0)]}
SEED_INPUT_PER_MP = 2904 / (1086 * 1448 / 1e6)

_SNAPSHOT = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def base_model(model: str) -> str:
    return _SNAPSHOT.sub("", model)


def price_for(model: str) -> Price | None:
    return PRICES.get(base_model(model))


def cost_usd(model: str, usage: dict[str, Any] | None) -> float | None:
    """按接口返回的 usage 算花费。没有输入明细时，输入按图像价算（宁高勿低）。"""
    price = price_for(model)
    if price is None or not usage:
        return None
    inp = usage.get("input_tokens_details") or {}
    if inp:
        text_in, image_in = inp.get("text_tokens", 0), inp.get("image_tokens", 0)
    else:
        text_in, image_in = 0, usage.get("input_tokens", 0)
    out = usage.get("output_tokens_details") or {}
    image_out = out.get("image_tokens", 0) if out else usage.get("output_tokens", 0)
    usd = (text_in * price.text_in + image_in * price.image_in + image_out * price.image_out) / 1e6
    return round(usd, 6)


@dataclass(frozen=True)
class Estimate:
    usd: float | None
    basis: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Estimator:
    def __init__(self, history: Iterable[dict[str, Any]] = ()):
        self.output: dict[tuple[str, str], list[tuple[tuple[int, int], float]]] = defaultdict(list)
        for key, samples in SEED_OUTPUT.items():
            self.output[key].extend(samples)
        self.input_per_mp: list[float] = [SEED_INPUT_PER_MP]
        self.observed: dict[tuple[str, str], int] = defaultdict(int)
        for call in history:
            self._learn(call)

    def _learn(self, call: dict[str, Any]) -> None:
        if call.get("backend") != "openai" or call.get("status") != "succeeded" or not call.get("usage"):
            return
        req, usage = call.get("request") or {}, call["usage"]
        try:
            size = parse_size((call.get("response") or {}).get("size") or req["size"])
            key = (base_model(req["model"]), req["quality"])
        except (KeyError, TypeError, ValueError):
            return
        out = usage.get("output_tokens_details") or {}
        image_out = out.get("image_tokens", 0) if out else usage.get("output_tokens", 0)
        count = len(call.get("outputs") or []) or req.get("n", 1)
        if size and image_out:
            self.output[key].append((size, image_out / count))
            self.observed[key] += 1
        image_in = (usage.get("input_tokens_details") or {}).get("image_tokens")
        mp = sum((im["width"] * im["height"]) / 1e6 for im in req.get("images") or [] if im.get("width"))
        if image_in and mp:
            self.input_per_mp.append(image_in / mp)

    def estimate(
        self,
        model: str,
        size: str,
        quality: str,
        n: int,
        prompt: str,
        image_sizes: Sequence[tuple[int | None, int | None]] = (),
    ) -> Estimate:
        price = price_for(model)
        if price is None:
            return Estimate(None, f"没有 {model} 的价格")
        wh = parse_size(size)
        if wh is None or quality == "auto":
            return Estimate(None, "auto 尺寸或质量档无法预估")
        samples = self.output.get((base_model(model), quality), [])
        exact = [tokens for s, tokens in samples if s == wh]
        if exact:
            out_tokens = max(exact)
            basis = "按同尺寸同质量的实测"
        elif samples:
            out_tokens = max(tokens / (s[0] * s[1]) for s, tokens in samples) * wh[0] * wh[1]
            basis = "按同质量档的实测、面积外推"
        else:
            return Estimate(None, f"{quality} 质量档还没有实测数据")
        per_mp = max(self.input_per_mp)
        # 尺寸未知的输入图（auto 尺寸的上游输出）按 1 百万像素估
        image_in = sum(per_mp * ((w * h) / 1e6 if w and h else 1.0) for w, h in image_sizes)
        text_in = len(prompt)  # 中文大致一字一 token，偏高估
        usd = (text_in * price.text_in + image_in * price.image_in + out_tokens * n * price.image_out) / 1e6
        return Estimate(round(usd, 4), basis)
