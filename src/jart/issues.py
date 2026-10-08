"""检查结果与错误类型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

Level = Literal["error", "warning"]


@dataclass(frozen=True)
class Issue:
    """一条检查结果。error 会拦下运行，warning 只提示。"""

    level: Level
    code: str
    message: str
    node: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        where = f"[{self.node}] " if self.node else ""
        return f"{where}{self.message}"


def error(code: str, message: str, node: str | None = None) -> Issue:
    return Issue("error", code, message, node)


def warning(code: str, message: str, node: str | None = None) -> Issue:
    return Issue("warning", code, message, node)


def has_errors(issues) -> bool:
    return any(i.level == "error" for i in issues)


class JArtError(Exception):
    """给用户看的错误（输入不对、找不到素材等）。"""
