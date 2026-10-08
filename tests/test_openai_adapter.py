from __future__ import annotations

import base64
import json

import httpx
import pytest

from jart.models import InputImage, ModelError, ModelRequest
from jart.models.openai_images import OpenAIImages

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
USAGE = {"input_tokens": 10, "output_tokens": 1180, "total_tokens": 1190}
OK = {"created": 1, "size": "1024x1024", "quality": "high", "data": [{"b64_json": base64.b64encode(PNG).decode()}], "usage": USAGE}


def capture(body, status=200):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        seen["body"] = request.read()
        return httpx.Response(status, json=body)

    return seen, httpx.MockTransport(handler)


def request(**kw):
    return ModelRequest(model="gpt-image-2.5-sunburst", prompt="图1是人设。", size="1024x1024", quality="high", **kw)


def test_generations_send_json_and_rely_on_network_key_when_no_env_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("JART_OPENAI_BASE_URL", raising=False)
    seen, transport = capture(OK)
    result = OpenAIImages(transport=transport).generate(request(background="transparent"))
    sent = seen["request"]
    assert str(sent.url) == "https://api.openai.com/v1/images/generations"
    assert "authorization" not in sent.headers
    assert json.loads(seen["body"]) == {
        "model": "gpt-image-2.5-sunburst",
        "prompt": "图1是人设。",
        "size": "1024x1024",
        "quality": "high",
        "n": 1,
        "output_format": "png",
        "background": "transparent",
    }
    assert result.images == [PNG] and result.usage == USAGE
    assert result.meta == {"created": 1, "size": "1024x1024", "quality": "high"}


def test_edits_multipart_keeps_image_order_and_puts_mask_last():
    seen, transport = capture(OK)
    images = (InputImage("char", b"FIRST-IMAGE", "image/png"), InputImage("ref", b"SECOND-IMAGE", "image/jpeg"))
    mask = InputImage("m", b"THE-MASK", "image/png")
    OpenAIImages(api_key="test-key", transport=transport).generate(request(images=images, mask=mask))
    sent, body = seen["request"], seen["body"]
    assert sent.url.path == "/v1/images/edits"
    assert sent.headers["authorization"] == "Bearer test-key"
    assert sent.headers["content-type"].startswith("multipart/form-data")
    first = body.index(b'name="image[]"; filename="image1.png"')
    second = body.index(b'name="image[]"; filename="image2.jpg"')
    last = body.index(b'name="mask"; filename="mask.png"')
    assert first < body.index(b"FIRST-IMAGE") < second < body.index(b"SECOND-IMAGE") < last < body.index(b"THE-MASK")
    assert b'name="prompt"\r\n\r\n' + "图1是人设。".encode() in body
    assert b'name="n"\r\n\r\n1\r\n' in body


def test_api_error_keeps_code_and_hides_key():
    body = {"error": {"message": "Incorrect API key provided: sk-proj-abc123456789xyz.", "type": "invalid_request_error", "code": "invalid_api_key"}}
    _, transport = capture(body, status=401)
    with pytest.raises(ModelError) as e:
        OpenAIImages(api_key="x", transport=transport).generate(request())
    assert e.value.status == 401 and e.value.code == "invalid_api_key"
    assert "abc123" not in e.value.message and "sk-***" in e.value.message


def test_moderation_block_details_are_kept():
    details = {"categories": ["harassment"], "moderation_stage": "input"}
    body = {"error": {"message": "Your request was rejected.", "code": "moderation_blocked", "moderation_details": details}}
    _, transport = capture(body, status=400)
    with pytest.raises(ModelError) as e:
        OpenAIImages(transport=transport).generate(request())
    assert e.value.code == "moderation_blocked" and e.value.detail["moderation_details"] == details


def test_network_errors_are_wrapped():
    def handler(request):
        raise httpx.ConnectError("boom")

    with pytest.raises(ModelError) as e:
        OpenAIImages(transport=httpx.MockTransport(handler)).generate(request())
    assert e.value.message.startswith("网络错误")
