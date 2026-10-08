"""引擎：编排节点、生成计划（预览）、按确认过的计划执行并留底。

两步走（设计案 §2「预览：确认后才扣费运行」）：
  plan()  不调模型。算出每次模型调用的提示词全文、图片顺序、参数和预估费用，以及所有检查结果。
  run()   重新走一遍图，每次调用前核对与计划一致，才发给模型；每次调用都留底（规则：直调 Images API，
          记录实际请求原文；docs/API实测要点.md：提示全文、图片顺序与哈希、参数、用量、耗时、输出哈希）。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from PIL import Image

from . import postprocess
from .graph import CompiledNode, compile_workflow
from .issues import Issue, JArtError, error, has_errors, warning
from .models import ImageModel, InputImage, ModelError, ModelRequest
from .pricing import Estimate, Estimator, cost_usd
from .sizes import parse_size
from .store import Store, now_iso
from .structure import LIST_FIELDS, MAX_PROMPT_CHARS, TEXT_FIELDS, PromptFormat, assemble, resolve_format
from .values import PENDING, ImageValue, PromptRequest, RequestImage
from .workflow import Workflow

EventHook = Callable[[str, dict[str, Any]], None]

MAX_INPUT_BYTES = 50 * 1024 * 1024  # 每张输入图 < 50MB（官方文档）
MAX_MASK_BYTES = 4 * 1024 * 1024  # 遮罩 < 4MB


class NodeFailure(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class PlannedCall:
    node: str
    title: str | None
    endpoint: str  # generations | edits
    params: dict[str, Any]
    prompt: str
    images: list[dict[str, Any]]
    mask: dict[str, Any] | None
    pinned: str | None = None
    estimate: Estimate | None = None

    def fingerprint(self) -> dict[str, Any]:
        """决定「是不是同一次调用」的内容；图片只看来源和角色，不看尺寸。"""

        def strip(entry):
            return {k: entry[k] for k in ("index", "name", "kind", "role", "source", "revision")}

        return {
            "node": self.node,
            "endpoint": self.endpoint,
            "params": self.params,
            "prompt": self.prompt,
            "images": [strip(e) for e in self.images],
            "mask": strip(self.mask) if self.mask else None,
            "pinned": self.pinned,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.fingerprint(),
            "title": self.title,
            "images": self.images,
            "mask": self.mask,
            "estimate": self.estimate.to_dict() if self.estimate else None,
        }


@dataclass
class Plan:
    workflow: str
    language: str
    calls: list[PlannedCall]
    issues: list[Issue]

    @property
    def ok(self) -> bool:
        return not has_errors(self.issues)

    @property
    def hash(self) -> str:
        blob = json.dumps([c.fingerprint() for c in self.calls], ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def estimated_usd(self) -> tuple[float, int]:
        """（能预估部分的合计，无法预估的调用数）"""
        known = [c.estimate.usd for c in self.calls if c.estimate and c.estimate.usd is not None]
        return round(sum(known), 4), len(self.calls) - len(known)

    def to_dict(self) -> dict[str, Any]:
        total, unknown = self.estimated_usd()
        return {
            "workflow": self.workflow,
            "language": self.language,
            "hash": self.hash,
            "ok": self.ok,
            "calls": [c.to_dict() for c in self.calls],
            "issues": [i.to_dict() for i in self.issues],
            "estimated_usd": total,
            "unestimated_calls": unknown,
        }


class Engine:
    def __init__(self, store: Store, model: ImageModel, *, end_user: str | None = None):
        self.store = store
        self.model = model
        self.end_user = end_user  # 匿名用户编号（哈希过），作为 user 字段随每次调用发出

    def plan(self, wf: Workflow, *, until: str | None = None) -> Plan:
        p = _Pass(self, wf, "plan", until=until)
        p.walk()
        return Plan(workflow=wf.name, language=wf.language, calls=p.calls, issues=p.issues)

    def run(self, wf: Workflow, plan: Plan, *, until: str | None = None, on_event: EventHook | None = None) -> dict[str, Any]:
        if not plan.ok:
            raise JArtError("计划里有错误，先修正再运行")
        if self.plan(wf, until=until).hash != plan.hash:
            raise JArtError("工作流和确认过的计划不一致，请重新预览后再运行")
        p = _Pass(self, wf, "run", until=until, plan=plan, on_event=on_event)
        p.walk()
        return p.finish()


class _Pass:
    """沿拓扑顺序走一遍图。plan 模式不调模型；run 模式执行并留底。"""

    def __init__(self, engine: Engine, wf: Workflow, mode: str, *, until: str | None, plan: Plan | None = None, on_event: EventHook | None = None):
        self.engine = engine
        self.store = engine.store
        self.mode = mode
        self.on_event = on_event or (lambda kind, data: None)
        self.graph = compile_workflow(wf)
        self.issues: list[Issue] = list(self.graph.issues)
        self.fmt: PromptFormat | None = None
        try:
            self.fmt = resolve_format(wf.language, wf.format)
        except JArtError as e:
            self.issues.append(error("E_FORMAT", str(e)))
        self.scope = self._scope(until)
        self.values: dict[tuple[str, str], Any] = {}
        self.status: dict[str, str] = {}
        self.produced: dict[str, str] = {}  # 节点 → 本次产出的素材（第一张）
        self.calls: list[PlannedCall] = []
        self.planned = {c.node: c for c in plan.calls} if plan else {}
        self.estimator = Estimator(self.store.history_calls()) if mode == "plan" else None
        if mode == "run":
            self.run_id, self.run_dir = self.store.new_run()
            (self.run_dir / "workflow.json").write_text(wf.model_dump_json(indent=2), "utf-8")
            self.record: dict[str, Any] = {
                "id": self.run_id,
                "workflow": wf.name,
                "language": wf.language,
                "backend": engine.model.name,
                "plan_hash": plan.hash if plan else None,
                "until": until,
                "status": "running",
                "started_at": now_iso(),
                "finished_at": None,
                "calls": [],
                "postprocess": [],
                "outputs": [],
                "cost_usd": 0.0,
                "issues": [],
            }
            self._save()

    def _scope(self, until: str | None) -> list[str]:
        if until is None:
            return list(self.graph.order)
        if until not in self.graph.nodes:
            self.issues.append(error("E_UNTIL", f"没有节点 {until}"))
            return []
        keep = self.graph.ancestors(until) | {until}
        return [n for n in self.graph.order if n in keep]

    # ---------- 遍历 ----------

    def walk(self) -> None:
        if has_errors(self.issues):
            return
        for nid in self.scope:
            node = self.graph.nodes[nid]
            if any(self.status.get(src) != "ok" for src, _ in node.inputs.values()):
                self.status[nid] = "skipped"
                continue
            inputs = {port: self.values[ref] for port, ref in node.inputs.items()}
            try:
                outputs = self._evaluate(node, inputs)
            except NodeFailure as e:
                self.status[nid] = "failed"
                self.issues.append(error(e.code, e.message, nid))
                continue
            except JArtError as e:
                self.status[nid] = "failed"
                self.issues.append(error("E_NODE", str(e), nid))
                continue
            for port, value in outputs.items():
                self.values[(nid, port)] = value
            self.status[nid] = "ok"

    def _evaluate(self, node: CompiledNode, inputs: dict[str, Any]) -> dict[str, Any]:
        kind = node.type.name
        if kind == "text":
            return {"text": node.params.text}
        if kind == "image":
            return {"image": self._image(node)}
        if kind == "structure":
            return {"request": self._structure(node, inputs)}
        if kind == "preview":
            return {"request": inputs["request"]}
        if kind == "generate":
            return {"image": self._generate(node, inputs["prompt"])}
        if kind == "postprocess":
            return {"image": self._postprocess(node, inputs["image"])}
        if kind == "output":
            self._output(node, inputs["image"])
            return {}
        raise AssertionError(kind)

    # ---------- 各类节点 ----------

    def _image(self, node: CompiledNode) -> ImageValue:
        p = node.params
        if p.path:
            return self.store.import_file(p.path, revision=p.revision)
        if p.asset:
            value = self.store.value(self.store.resolve(p.asset))
            if p.revision > value.revision:
                value = ImageValue(value.asset, None, value.width, value.height, p.revision, value.alpha)
            return value
        raise NodeFailure("E_NO_IMAGE", "没有提供图像")

    def _structure(self, node: CompiledNode, inputs: dict[str, Any]) -> PromptRequest:
        p = node.params
        texts = {f: inputs[f] if f in inputs else getattr(p, f) for f in TEXT_FIELDS}
        lists = {f: inputs[f].splitlines() if f in inputs else getattr(p, f) for f in LIST_FIELDS}
        images = {d.name: inputs[d.name] for d in p.images}
        req = assemble(node.id, p, images, texts, lists, self.fmt)
        if self.mode == "plan":
            self.issues.extend(req.issues)
        elif has_errors(req.issues):
            raise NodeFailure("E_STRUCTURE", "；".join(i.message for i in req.issues if i.level == "error"))
        return req

    def _generate(self, node: CompiledNode, prompt: str | PromptRequest) -> ImageValue:
        p = node.params
        if isinstance(prompt, PromptRequest):
            req = prompt
        else:
            req = PromptRequest(prompt=prompt.strip())
            if not req.prompt:
                raise NodeFailure("E_EMPTY_PROMPT", "提示词是空的")
            if len(req.prompt) > MAX_PROMPT_CHARS:
                raise NodeFailure("E_PROMPT_TOO_LONG", f"提示词 {len(req.prompt)} 字，超过 {MAX_PROMPT_CHARS} 字上限")
        call = PlannedCall(
            node=node.id,
            title=node.title,
            endpoint="edits" if req.images else "generations",
            params=p.model_dump(exclude={"pin"}, exclude_none=True),
            prompt=req.prompt,
            images=[ri.entry() for ri in req.images],
            mask=req.mask.entry() if req.mask else None,
            pinned=self.store.resolve(p.pin) if p.pin else None,
        )
        target = next((ri for ri in req.images if ri.kind == "edit_target"), None)
        revision = target.image.revision + 1 if target else 0
        if self.mode == "plan":
            return self._plan_generate(node, call, req, revision)
        return self._run_generate(node, call, req, revision)

    def _plan_generate(self, node: CompiledNode, call: PlannedCall, req: PromptRequest, revision: int) -> ImageValue:
        p = node.params
        self.calls.append(call)
        if call.pinned:
            call.estimate = Estimate(0.0, "已采用，不调用模型")
            return self.store.value(call.pinned)
        if p.size == "auto" or p.quality == "auto":
            self.issues.append(warning("W_AUTO", "size 或 quality 用 auto 时无法预估费用，结果也不稳定；最好写明尺寸和质量档", node.id))
        problem = self._mask_problem(req)
        if problem:
            self.issues.append(error("E_MASK", problem, node.id))
        sizes = [(ri.image.width, ri.image.height) for ri in req.images]
        call.estimate = self.estimator.estimate(p.model, p.size, p.quality, p.n, req.prompt, sizes)
        wh = parse_size(p.size)
        return ImageValue(
            producer=node.id,
            width=wh[0] if wh else None,
            height=wh[1] if wh else None,
            revision=revision,
            alpha=True if p.background == "transparent" else None,
        )

    def _run_generate(self, node: CompiledNode, call: PlannedCall, req: PromptRequest, revision: int) -> ImageValue:
        planned = self.planned.get(node.id)
        if planned is None or not self._same_call(planned, call):
            raise NodeFailure("E_PLAN_CHANGED", "这次调用和确认过的计划不一致，请重新预览后再运行")
        if call.pinned:
            self.produced[node.id] = call.pinned
            pinned = {"node": node.id, "title": node.title, "status": "pinned", "pinned": call.pinned, "cost_usd": 0.0}
            self.record["calls"].append(pinned)
            self._save()
            self.on_event("call_pinned", pinned)
            return self.store.value(call.pinned)
        problem = self._mask_problem(req)
        if problem:
            raise NodeFailure("E_MASK", problem)

        p = node.params
        request = ModelRequest(
            model=p.model,
            prompt=req.prompt,
            size=p.size,
            quality=p.quality,
            n=p.n,
            output_format=p.output_format,
            background=p.background,
            output_compression=p.output_compression,
            moderation=p.moderation,
            user=self.engine.end_user,
            images=tuple(self._input_image(ri) for ri in req.images),
            mask=self._input_image(req.mask) if req.mask else None,
        )
        record: dict[str, Any] = {
            "node": node.id,
            "title": node.title,
            "backend": self.engine.model.name,
            "endpoint": request.endpoint,
            "request": {
                **request.fields(),
                "images": [self._image_record(ri) for ri in req.images],
                "mask": self._image_record(req.mask) if req.mask else None,
            },
            "status": "running",
        }
        self.on_event("call_start", record)
        started = time.monotonic()
        try:
            result = self.engine.model.generate(request)
        except ModelError as e:
            record.update(status="failed", elapsed_s=round(time.monotonic() - started, 2), error=e.to_dict(), cost_usd=0.0)
            self.record["calls"].append(record)
            self._save()
            self.on_event("call_failed", record)
            raise NodeFailure("E_MODEL", f"模型调用失败：{e.message}") from e
        elapsed = round(time.monotonic() - started, 2)

        parents = [ri.image.asset for ri in req.images]
        values, outputs = [], []
        for i, data in enumerate(result.images, 1):
            value = self.store.put(data, origin="generated", revision=revision, parents=parents, extra={"node": node.id, "run": self.run_id})
            stem = node.id if len(result.images) == 1 else f"{node.id}_{i}"
            values.append(value)
            outputs.append({"sha256": value.asset, "width": value.width, "height": value.height, "revision": value.revision, "file": self._export(value, stem)})
        record.update(
            status="succeeded",
            elapsed_s=elapsed,
            response=result.meta,
            usage=result.usage,
            cost_usd=cost_usd(p.model, result.usage),
            outputs=outputs,
        )
        self.record["calls"].append(record)
        self._save()
        self.on_event("call_done", record)
        self.produced[node.id] = values[0].asset
        return values[0]

    def _postprocess(self, node: CompiledNode, image: ImageValue) -> ImageValue:
        p = node.params
        if self.mode == "plan":
            width, height = (p.resize.width, p.resize.height) if p.resize else (image.width, image.height)
            alpha = True if p.resize and p.resize.mode == "contain" else image.alpha
            return ImageValue(producer=node.id, width=width, height=height, revision=image.revision, alpha=alpha)
        with Image.open(BytesIO(self.store.read(image.asset))) as im:
            im.load()
            out, stats = postprocess.apply(im, p.alpha, p.resize)
        buf = BytesIO()
        out.save(buf, format="PNG")
        value = self.store.put(
            buf.getvalue(),
            origin="postprocessed",
            revision=image.revision,  # 像素清理不算模型修改
            parents=[image.asset],
            extra={"node": node.id, "run": self.run_id, "ops": p.model_dump(exclude_none=True)},
        )
        self.record["postprocess"].append({"node": node.id, "input": image.asset, "output": value.asset, "stats": stats})
        self.produced[node.id] = value.asset
        return value

    def _output(self, node: CompiledNode, image: ImageValue) -> None:
        if self.mode == "plan":
            return
        name = node.params.name or node.id
        self.record["outputs"].append({"node": node.id, "title": node.title, "sha256": image.asset, "file": self._export(image, name)})

    # ---------- 辅助 ----------

    def _same_call(self, planned: PlannedCall, actual: PlannedCall) -> bool:
        a, b = planned.fingerprint(), actual.fingerprint()
        images_a, images_b = a.pop("images"), b.pop("images")
        mask_a, mask_b = a.pop("mask"), b.pop("mask")
        if a != b or len(images_a) != len(images_b) or (mask_a is None) != (mask_b is None):
            return False
        pairs = list(zip(images_a, images_b))
        if mask_a is not None:
            pairs.append((mask_a, mask_b))
        for x, y in pairs:
            if {k: v for k, v in x.items() if k != "source"} != {k: v for k, v in y.items() if k != "source"}:
                return False
            src = x["source"]
            if src != y["source"] and not (src.startswith(PENDING) and self.produced.get(src[len(PENDING):]) == y["source"]):
                return False
        return True

    def _mask_problem(self, req: PromptRequest) -> str | None:
        """遮罩要是带透明通道的 PNG、小于 4MB、和第一张图同尺寸。只检查已经存在的图。"""
        if req.mask is None or req.mask.image.asset is None:
            return None
        meta = self.store.meta(req.mask.image.asset)
        if meta["format"] != "png" or not meta["alpha"]:
            return "遮罩必须是带透明通道的 PNG"
        if meta["bytes"] > MAX_MASK_BYTES:
            return "遮罩要小于 4MB"
        target = req.images[0].image if req.images else None
        if target and target.asset:
            tm = self.store.meta(target.asset)
            if (tm["width"], tm["height"]) != (meta["width"], meta["height"]):
                return f"遮罩 {meta['width']}×{meta['height']} 和要修改的图 {tm['width']}×{tm['height']} 尺寸不同"
        return None

    def _input_image(self, ri: RequestImage) -> InputImage:
        meta = self.store.meta(ri.image.asset)
        if meta["bytes"] > MAX_INPUT_BYTES:
            raise NodeFailure("E_IMAGE_TOO_BIG", f"{ri.name} 超过 50MB")
        return InputImage(name=ri.name, data=self.store.read(ri.image.asset), mime=meta["mime"])

    def _image_record(self, ri: RequestImage) -> dict[str, Any]:
        meta = self.store.meta(ri.image.asset)
        return {
            "index": ri.index,
            "name": ri.name,
            "kind": ri.kind,
            "role": ri.role,
            "sha256": ri.image.asset,
            "width": meta["width"],
            "height": meta["height"],
            "revision": ri.image.revision,
            "bytes": meta["bytes"],
        }

    def _export(self, value: ImageValue, stem: str) -> str:
        ext = self.store.meta(value.asset)["format"]
        out_dir = self.run_dir / "out"
        out_dir.mkdir(exist_ok=True)
        target, k = out_dir / f"{stem}.{ext}", 2
        while target.exists():
            target, k = out_dir / f"{stem}-{k}.{ext}", k + 1
        shutil.copyfile(self.store.file(value.asset), target)
        return target.relative_to(self.run_dir).as_posix()

    def _save(self) -> None:
        self.store.save_run(self.record)

    def finish(self) -> dict[str, Any]:
        failed = [n for n in self.scope if self.status.get(n) != "ok"]
        costs = [c.get("cost_usd") for c in self.record["calls"]]
        self.record.update(
            status="succeeded" if self.scope and not failed and not has_errors(self.issues) else "failed",
            finished_at=now_iso(),
            cost_usd=round(sum(c for c in costs if c), 6),
            node_status={n: self.status.get(n, "skipped") for n in self.scope},
            issues=[i.to_dict() for i in self.issues],
        )
        self._save()
        return self.record
