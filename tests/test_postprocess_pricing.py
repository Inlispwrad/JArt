from __future__ import annotations

import pytest
from PIL import Image

from jart.postprocess import clean_alpha, fit
from jart.pricing import Estimator, cost_usd
from jart.sizes import check_size


def test_clean_alpha_thresholds_keep_color_and_edges():
    alphas = [0, 3, 8, 9, 128, 249, 250, 253, 254, 255]
    im = Image.new("RGBA", (len(alphas), 1))
    for x, a in enumerate(alphas):
        im.putpixel((x, 0), (10, 20, 30, a))
    out, stats = clean_alpha(im)
    assert [out.getpixel((x, 0))[3] for x in range(len(alphas))] == [0, 0, 0, 9, 128, 249, 255, 255, 255, 255]
    assert out.getpixel((1, 0))[:3] == (10, 20, 30)
    assert stats["haze_to_0"] == 2 and stats["body_to_255"] == 3 and stats["partial_left"] == 3


def test_edge_px_drops_partial_alpha_far_from_the_body():
    im = Image.new("RGBA", (40, 9), (0, 0, 0, 0))
    for x in range(10):
        for y in range(9):
            im.putpixel((x, y), (255, 0, 0, 255))
    im.putpixel((11, 4), (255, 0, 0, 128))  # 主体边缘外 2px
    im.putpixel((30, 4), (255, 0, 0, 128))  # 远处
    out, stats = clean_alpha(im, edge_px=3)
    assert out.getpixel((11, 4))[3] == 128 and out.getpixel((30, 4))[3] == 0
    assert stats["far_partial_to_0"] == 1


def test_clean_alpha_skips_opaque_images():
    out, stats = clean_alpha(Image.new("RGB", (4, 4)))
    assert stats == {"skipped": "没有透明通道"}


def test_fit_cover_and_contain():
    im = Image.new("RGBA", (200, 100), (255, 0, 0, 255))
    assert fit(im, 50, 50, "cover").size == (50, 50)
    contained = fit(im, 50, 50, "contain")
    assert contained.size == (50, 50)
    assert contained.getpixel((25, 2))[3] == 0 and contained.getpixel((25, 25))[3] == 255


def test_sizes():
    check_size("2048x1024", "gpt-image-2.5-sunburst")
    check_size("auto", "gpt-image-2.5-sunburst")
    for bad in ("1000x1000", "3840x1024", "512x512", "4096x2048"):
        with pytest.raises(ValueError):
            check_size(bad, "gpt-image-2.5-sunburst")
    with pytest.raises(ValueError):
        check_size("big", "gpt-image-2.5-sunburst")


USAGE = {
    "input_tokens": 1100,
    "input_tokens_details": {"text_tokens": 100, "image_tokens": 1000},
    "output_tokens": 1180,
    "output_tokens_details": {"image_tokens": 1180, "text_tokens": 0},
}


def test_cost_from_usage():
    expected = round((100 * 5 + 1000 * 8 + 1180 * 30) / 1e6, 6)
    assert cost_usd("gpt-image-2.5-sunburst", USAGE) == expected
    assert cost_usd("gpt-image-2.5-flare-2026-09-08", USAGE) == expected
    assert cost_usd("gpt-image-2.5-sunburst", {"input_tokens": 1000, "output_tokens": 10}) == round((1000 * 8 + 10 * 30) / 1e6, 6)
    assert cost_usd("some-other-model", USAGE) is None
    assert cost_usd("gpt-image-2.5-sunburst", None) is None


def test_estimate_from_seed_and_area():
    est = Estimator()
    exact = est.estimate("gpt-image-2.5-sunburst", "2048x1024", "high", 1, "", [])
    assert exact.usd == round(1180 * 30 / 1e6, 4) and "同尺寸" in exact.basis
    half = est.estimate("gpt-image-2.5-sunburst", "1024x1024", "high", 1, "", [])
    assert half.usd == round(590 * 30 / 1e6, 4) and "外推" in half.basis
    assert est.estimate("gpt-image-2.5-sunburst", "2048x1024", "low", 1, "", []).usd is None
    assert est.estimate("gpt-image-2.5-sunburst", "auto", "high", 1, "", []).usd is None
    with_refs = est.estimate("gpt-image-2.5-sunburst", "2048x1024", "high", 1, "图1是人设。", [(1086, 1448)])
    assert with_refs.usd == round((6 * 5 + 2904 * 8 + 1180 * 30) / 1e6, 4)


def test_estimate_learns_from_real_calls_only():
    def call(backend, tokens):
        return {
            "backend": backend,
            "status": "succeeded",
            "request": {"model": "gpt-image-2.5-sunburst", "size": "1024x1024", "quality": "low", "n": 1, "images": []},
            "response": {"size": "1024x1024"},
            "usage": {"output_tokens": tokens},
            "outputs": [{}],
        }

    est = Estimator([call("fake", 999), call("openai", 200)])
    assert est.estimate("gpt-image-2.5-sunburst", "1024x1024", "low", 2, "", []).usd == round(400 * 30 / 1e6, 4)
