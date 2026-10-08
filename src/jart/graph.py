"""把工作流编译成可执行的图：校验参数、连线、类型，拓扑排序。"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from .issues import Issue, error
from .nodes import NODE_TYPES, NodeType
from .structure import LIST_FIELDS, TEXT_FIELDS
from .workflow import Workflow


@dataclass
class CompiledNode:
    id: str
    type: NodeType
    title: str | None
    params: BaseModel
    inputs: dict[str, tuple[str, str]] = field(default_factory=dict)  # 端口 → (上游节点, 上游端口)


@dataclass
class Graph:
    nodes: dict[str, CompiledNode]
    order: list[str]
    issues: list[Issue]

    def ancestors(self, node_id: str) -> set[str]:
        seen: set[str] = set()
        stack = [node_id]
        while stack:
            for src, _ in self.nodes[stack.pop()].inputs.values():
                if src not in seen:
                    seen.add(src)
                    stack.append(src)
        return seen


def compile_workflow(wf: Workflow) -> Graph:
    issues: list[Issue] = []
    nodes: dict[str, CompiledNode] = {}
    specs = []
    seen: set[str] = set()
    for spec in wf.nodes:
        if spec.id in seen:
            issues.append(error("E_DUP_ID", f"节点编号重复：{spec.id}", spec.id))
            continue
        seen.add(spec.id)
        specs.append(spec)
        node_type = NODE_TYPES.get(spec.type)
        if node_type is None:
            issues.append(error("E_NODE_TYPE", f"没有 {spec.type} 这种节点", spec.id))
            continue
        try:
            params = node_type.params.model_validate(spec.params)
        except ValidationError as e:
            for err in e.errors():
                loc = ".".join(str(x) for x in err["loc"])
                msg = err["msg"].removeprefix("Value error, ")
                issues.append(error("E_PARAM", f"参数 {loc}：{msg}" if loc else f"参数：{msg}", spec.id))
            continue
        nodes[spec.id] = CompiledNode(spec.id, node_type, spec.title, params)

    for spec in specs:
        node = nodes.get(spec.id)
        if node is None:
            continue
        ports = node.type.input_ports(node.params)
        for port, ref in spec.inputs.items():
            if port not in ports:
                issues.append(error("E_PORT", f"没有输入端口 {port}", spec.id))
                continue
            src_id, _, src_port = ref.partition(".")
            src = nodes.get(src_id)
            if src is None:
                issues.append(error("E_SRC", f"输入 {port} 指向的节点 {src_id} 不存在或无效", spec.id))
                continue
            src_port = src_port or src.type.default_output
            out_type = src.type.outputs.get(src_port) if src_port else None
            if out_type is None:
                issues.append(error("E_SRC_PORT", f"节点 {src_id} 没有输出 {src_port or ''}".rstrip(), spec.id))
                continue
            if out_type not in ports[port].types:
                want = " / ".join(ports[port].types)
                issues.append(error("E_TYPE", f"输入 {port} 要 {want}，{src_id}.{src_port} 给的是 {out_type}", spec.id))
                continue
            node.inputs[port] = (src_id, src_port)
        for port, spec_port in ports.items():
            if spec_port.required and port not in spec.inputs:
                issues.append(error("E_REQUIRED", f"缺少输入连线：{port}", spec.id))
        if node.type.name == "structure":
            for f in (*TEXT_FIELDS, *LIST_FIELDS):
                if f in node.inputs and getattr(node.params, f):
                    issues.append(error("E_DOUBLE", f"{f} 同时写了参数和连线，只能留一个", spec.id))

    order, stuck = _toposort(nodes, [s.id for s in specs if s.id in nodes])
    if stuck:
        issues.append(error("E_CYCLE", f"连线成环：{'、'.join(stuck)}"))
    return Graph(nodes, order, issues)


def _toposort(nodes: dict[str, CompiledNode], declared: list[str]) -> tuple[list[str], list[str]]:
    """按声明顺序稳定排序；返回（顺序，成环的节点）。"""
    placed: list[str] = []
    done: set[str] = set()
    pending = list(declared)
    while pending:
        for nid in pending:
            if all(src in done for src, _ in nodes[nid].inputs.values()):
                placed.append(nid)
                done.add(nid)
                pending.remove(nid)
                break
        else:
            return placed, pending
    return placed, []
