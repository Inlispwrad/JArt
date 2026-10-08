"""在节点之间流动的值。文本就是 str；图像和拼好的请求用下面的类型。"""

from __future__ import annotations

from dataclasses import dataclass

from .issues import Issue

PENDING = "pending:"


@dataclass(frozen=True)
class ImageValue:
    """一张图。已存在时有 asset（sha256）；计划阶段还没生成的图只有 producer（产出它的节点）。

    revision 是被模型修改过的次数：上传和新生成的图是 0，拿「要修改的图」去改一次加 1。
    规则 §2：参考图和修改对象只能是原图或修改过一次的图。
    """

    asset: str | None = None
    producer: str | None = None
    width: int | None = None
    height: int | None = None
    revision: int = 0
    alpha: bool | None = None

    def source(self) -> str:
        return self.asset if self.asset else f"{PENDING}{self.producer}"


def source_label(source: str) -> str:
    if source.startswith(PENDING):
        return f"〈{source[len(PENDING):]} 的输出〉"
    return source[:12]


@dataclass(frozen=True)
class RequestImage:
    """请求里的一张图。index 是发给接口的顺序（提示里的「图N」）；遮罩没有 index。"""

    index: int | None
    name: str
    kind: str
    role: str
    image: ImageValue

    def entry(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "kind": self.kind,
            "role": self.role,
            "source": self.image.source(),
            "revision": self.image.revision,
            "width": self.image.width,
            "height": self.image.height,
        }


@dataclass(frozen=True)
class PromptRequest:
    """结构节点的产物：最终提示词全文 + 图片顺序 + 遮罩 + 检查结果。"""

    prompt: str
    images: tuple[RequestImage, ...] = ()
    mask: RequestImage | None = None
    issues: tuple[Issue, ...] = ()
