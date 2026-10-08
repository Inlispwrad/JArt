"""本地素材库与运行记录（P1 本机版；P2 换成数据库＋对象存储）。

目录（默认 ~/.jart，可用 JART_HOME 或 --home 改）：
  assets/ab/<sha256>.png   素材按内容哈希存，同一张图只存一份
  assets/ab/<sha256>.json  元数据：尺寸、格式、来源、修改次数（revision）、上游素材
  runs/<运行编号>/run.json  每次运行的留底；out/ 下是这次的原图和输出
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from .issues import JArtError
from .values import ImageValue

FORMATS = {"PNG": ("png", "image/png"), "JPEG": ("jpg", "image/jpeg"), "WEBP": ("webp", "image/webp")}


def default_home() -> Path:
    return Path(os.environ.get("JART_HOME") or Path.home() / ".jart")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def has_alpha(im: Image.Image) -> bool:
    return im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)


class Store:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root else default_home()
        self.assets_dir = self.root / "assets"
        self.runs_dir = self.root / "runs"

    # ---------- 素材 ----------

    def put(
        self,
        data: bytes,
        *,
        origin: str,
        revision: int = 0,
        parents: Sequence[str] = (),
        extra: dict[str, Any] | None = None,
    ) -> ImageValue:
        """存一张图并返回它。已有同样内容时沿用旧记录；revision 取较大值（宁严勿松）。"""
        sha = hashlib.sha256(data).hexdigest()
        meta_path = self._meta_path(sha)
        if meta_path.exists():
            meta = self.meta(sha)
            if revision > meta["revision"]:
                meta["revision"] = revision
                _write_json(meta_path, meta)
            return _value(meta)
        try:
            with Image.open(BytesIO(data)) as im:
                fmt, (width, height), alpha = im.format, im.size, has_alpha(im)
        except (UnidentifiedImageError, OSError) as e:
            raise JArtError("无法识别的图片") from e
        if fmt not in FORMATS:
            raise JArtError(f"只支持 PNG / JPEG / WEBP，收到 {fmt}")
        ext, mime = FORMATS[fmt]
        meta = {
            "sha256": sha,
            "format": ext,
            "mime": mime,
            "width": width,
            "height": height,
            "alpha": alpha,
            "bytes": len(data),
            "revision": revision,
            "origin": origin,
            "parents": list(parents),
            "created_at": now_iso(),
            **(extra or {}),
        }
        path = self._data_path(sha, ext)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_bytes(path, data)
        _write_json(meta_path, meta)
        return _value(meta)

    def import_file(self, path: Path | str, *, revision: int = 0) -> ImageValue:
        path = Path(path)
        if not path.is_file():
            raise JArtError(f"找不到图片：{path}")
        return self.put(path.read_bytes(), origin="upload", revision=revision, extra={"source_name": path.name})

    def resolve(self, ref: str) -> str:
        """素材编号（完整 sha256 或至少 6 位前缀）→ 完整 sha256。"""
        ref = ref.removeprefix("sha256:").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{6,64}", ref):
            raise JArtError(f"素材编号不对：{ref}")
        folder = self.assets_dir / ref[:2]
        found = sorted({p.stem for p in folder.glob(f"{ref}*.json")}) if folder.is_dir() else []
        if not found:
            raise JArtError(f"素材库里没有 {ref}")
        if len(found) > 1:
            raise JArtError(f"编号 {ref} 对应多张图，请写长一点")
        return found[0]

    def meta(self, sha: str) -> dict[str, Any]:
        try:
            return json.loads(self._meta_path(sha).read_text("utf-8"))
        except FileNotFoundError as e:
            raise JArtError(f"素材库里没有 {sha[:12]}") from e

    def value(self, sha: str) -> ImageValue:
        return _value(self.meta(sha))

    def file(self, sha: str) -> Path:
        return self._data_path(sha, self.meta(sha)["format"])

    def read(self, sha: str) -> bytes:
        return self.file(sha).read_bytes()

    def _meta_path(self, sha: str) -> Path:
        return self.assets_dir / sha[:2] / f"{sha}.json"

    def _data_path(self, sha: str, ext: str) -> Path:
        return self.assets_dir / sha[:2] / f"{sha}.{ext}"

    # ---------- 运行记录 ----------

    def new_run(self) -> tuple[str, Path]:
        run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True)
        return run_id, run_dir

    def save_run(self, record: dict[str, Any]) -> None:
        _write_json(self.runs_dir / record["id"] / "run.json", record)

    def load_run(self, ref: str) -> dict[str, Any]:
        if not re.fullmatch(r"[0-9a-f\-]{4,40}", ref):
            raise JArtError(f"运行编号不对：{ref}")
        found = sorted(self.runs_dir.glob(f"{ref}*/run.json")) if self.runs_dir.is_dir() else []
        if not found:
            raise JArtError(f"没有运行记录 {ref}")
        if len(found) > 1:
            raise JArtError(f"{ref} 对应多次运行，请写长一点")
        return json.loads(found[0].read_text("utf-8"))

    def list_runs(self) -> list[dict[str, Any]]:
        if not self.runs_dir.is_dir():
            return []
        records = []
        for path in sorted(self.runs_dir.glob("*/run.json"), reverse=True):
            try:
                records.append(json.loads(path.read_text("utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return records

    def history_calls(self) -> Iterator[dict[str, Any]]:
        for record in self.list_runs():
            yield from record.get("calls", [])


def _value(meta: dict[str, Any]) -> ImageValue:
    return ImageValue(
        asset=meta["sha256"],
        width=meta["width"],
        height=meta["height"],
        revision=meta["revision"],
        alpha=meta["alpha"],
    )


def _write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _write_json(path: Path, obj: Any) -> None:
    _write_bytes(path, json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"))
