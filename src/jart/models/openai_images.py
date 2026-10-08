"""gpt-image-2.5 直调 Images API（docs/API实测要点.md）。

- 没有参考图走 /v1/images/generations（JSON）；有参考图走 /v1/images/edits（multipart），
  多张图用重复的 image[] 字段，顺序即提示里的「图1、图2…」；mask 对应第一张图。
- 提示词原样发送。不走 Responses API 的 image_generation 工具（主模型会改写提示）。
- 认证：有 OPENAI_API_KEY 环境变量就带上（本机）；没有就不带——云端环境由网络密钥在出口注入
  （绑定 api.openai.com）。这里不打印、不记录 key。
"""

from __future__ import annotations

import base64
import os
import re

import httpx

from . import ModelError, ModelRequest, ModelResult

DEFAULT_BASE_URL = "https://api.openai.com/v1"
_EXT = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}
_KEYLIKE = re.compile(r"sk-[A-Za-z0-9_\-*]{4,}")


class OpenAIImages:
    name = "openai"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 900.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.base_url = (base_url or os.environ.get("JART_OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self._api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self._client = httpx.Client(timeout=httpx.Timeout(timeout, connect=30.0), transport=transport)

    def generate(self, request: ModelRequest) -> ModelResult:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        fields = request.fields()
        try:
            if request.images:
                files = [
                    ("image[]", (f"image{i}.{_EXT.get(img.mime, 'png')}", img.data, img.mime))
                    for i, img in enumerate(request.images, 1)
                ]
                if request.mask is not None:
                    files.append(("mask", ("mask.png", request.mask.data, request.mask.mime)))
                data = {k: str(v) for k, v in fields.items()}
                resp = self._client.post(f"{self.base_url}/images/edits", data=data, files=files, headers=headers)
            else:
                resp = self._client.post(f"{self.base_url}/images/generations", json=fields, headers=headers)
        except httpx.HTTPError as e:
            raise ModelError(f"网络错误：{type(e).__name__}: {e}") from e
        if resp.status_code >= 400:
            raise _error(resp)
        body = resp.json()
        images = [base64.b64decode(item["b64_json"]) for item in body.get("data") or [] if item.get("b64_json")]
        if not images:
            raise ModelError("接口没有返回图片", status=resp.status_code)
        meta = {k: body[k] for k in ("created", "size", "quality", "background", "output_format") if k in body}
        return ModelResult(images=images, usage=body.get("usage"), meta=meta)


def _error(resp: httpx.Response) -> ModelError:
    try:
        err = resp.json().get("error") or {}
    except ValueError:
        err = {}
    if not isinstance(err, dict):
        err = {"message": str(err)}
    message = _KEYLIKE.sub("sk-***", err.get("message") or resp.text[:500] or f"HTTP {resp.status_code}")
    detail = {k: v for k, v in err.items() if k != "message"}
    return ModelError(message, status=resp.status_code, code=err.get("code"), detail=detail)
