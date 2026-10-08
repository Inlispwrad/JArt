"""命令行（P1 本机版）。

  jart presets [名字]                      官方预设，以及它要填的输入
  jart plan 模板 [-i 输入.toml] [-s 键=值]    预览：每次模型调用的提示词全文、图片顺序、预估费用、检查结果
  jart run  模板 [同上] [--fake] [-y]        预览，确认后运行
  jart runs                                运行记录
  jart show 运行编号                         一次运行的留底
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .engine import Engine, Plan
from .issues import Issue, JArtError, has_errors
from .models.fake import FakeImages
from .models.openai_images import OpenAIImages
from .store import Store
from .templates import load_preset, load_template, preset_ids
from .values import source_label
from .workflow import Workflow, bind, load_values

STATUS = {"succeeded": "成功", "failed": "失败", "running": "未完成", "pinned": "已采用"}


class _Exit(Exception):
    def __init__(self, code: int):
        self.code = code


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    try:
        return args.func(args)
    except _Exit as e:
        return e.code
    except JArtError as e:
        print(f"错误：{e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("已中断", file=sys.stderr)
        return 130


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--home", help="素材库和运行记录的目录（默认 $JART_HOME 或 ~/.jart）")

    parser = argparse.ArgumentParser(prog="jart", description="JArt 引导式生图引擎（P1 本机版）")
    parser.add_argument("--version", action="version", version=f"jart {__version__}")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("presets", parents=[common], help="官方预设；带名字时列出它要填的输入")
    p.add_argument("name", nargs="?")
    p.set_defaults(func=cmd_presets)

    for name, func, help_text in (("plan", cmd_plan, "预览，不调用模型"), ("run", cmd_run, "预览，确认后运行")):
        p = sub.add_parser(name, parents=[common], help=help_text)
        p.add_argument("template", help="模板文件，或官方预设名（swap / chain / text）")
        p.add_argument("-i", "--inputs", help="输入文件（.toml 或 .json）；里面的图片路径相对于这个文件")
        p.add_argument("-s", "--set", action="append", default=[], metavar="键=值", help="填一个输入；同一个键写多次就是多行")
        p.add_argument("--pin", action="append", default=[], metavar="节点=素材", help="采用已有结果，这个生成节点不再调用模型")
        p.add_argument("--until", metavar="节点", help="只跑到这个节点（连同上游）")
        p.add_argument("--fake", action="store_true", help="用假模型（不联网、不花钱），演练流程")
        p.add_argument("--json", action="store_true", help="计划以 JSON 输出")
        if name == "run":
            p.add_argument("-y", "--yes", action="store_true", help="不再询问，直接运行")
            p.add_argument("--out", help="把输出节点的图另外复制到这个目录")
        p.set_defaults(func=func)

    p = sub.add_parser("runs", parents=[common], help="运行记录")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_runs)

    p = sub.add_parser("show", parents=[common], help="一次运行的留底")
    p.add_argument("run", help="运行编号（可以只写前几位）")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_show)
    return parser


# ---------- 命令 ----------


def cmd_presets(args: argparse.Namespace) -> int:
    if not args.name:
        for pid in preset_ids():
            wf = load_preset(pid)
            print(f"{pid:8}{wf.name}\n{'':8}{wf.description}")
        return 0
    wf = load_template(args.name)
    print(f"{wf.name}\n{wf.description}\n\n输入：")
    for key, slot in wf.slots.items():
        kind = {"text": "文字", "lines": "多行", "image": "图片"}[slot.kind]
        if slot.default is not None:
            need = f"默认 {slot.default}"
        else:
            need = "必填" if slot.required else "可选"
        print(f"  {key:<12} {slot.label}（{kind}，{need}）")
        if slot.hint:
            print(f"  {'':<12} {slot.hint}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    wf, engine = _prepare(args)
    plan = engine.plan(wf, until=args.until)
    _show_plan(args, wf, plan)
    return 0 if plan.ok else 2


def cmd_run(args: argparse.Namespace) -> int:
    wf, engine = _prepare(args)
    plan = engine.plan(wf, until=args.until)
    _show_plan(args, wf, plan)
    if not plan.ok:
        print("\n有错误，没有运行。")
        return 2
    if not args.yes:
        if not sys.stdin.isatty():
            raise JArtError("不是交互终端；确认无误后加 --yes 运行")
        tip = "（假模型，不花钱）" if args.fake else "会调用模型并产生费用"
        if input(f"\n确认运行？{tip} [y/N] ").strip().lower() not in ("y", "yes", "是"):
            print("没有运行。")
            return 3
    print()
    record = engine.run(wf, plan, until=args.until, on_event=_progress)
    run_dir = engine.store.runs_dir / record["id"]
    print(f"\n运行 {record['id']}：{STATUS.get(record['status'], record['status'])}，花费 {_usd(record['cost_usd'])}")
    for item in record["issues"]:
        if item["level"] == "error":
            print(f"  错误 [{item['node']}] {item['message']}")
    for out in record["outputs"]:
        print(f"  输出 {out.get('title') or out['node']}：{run_dir / out['file']}")
    print(f"  留底：{run_dir / 'run.json'}")
    if args.out and record["outputs"]:
        dest = Path(args.out)
        dest.mkdir(parents=True, exist_ok=True)
        for out in record["outputs"]:
            shutil.copy2(run_dir / out["file"], dest / Path(out["file"]).name)
        print(f"  已复制到 {dest}")
    return 0 if record["status"] == "succeeded" else 1


def cmd_runs(args: argparse.Namespace) -> int:
    runs = Store(args.home).list_runs()[: args.limit]
    if not runs:
        print("还没有运行记录")
    for r in runs:
        done = sum(1 for c in r["calls"] if c.get("status") == "succeeded")
        status = STATUS.get(r["status"], r["status"])
        print(f"{r['id']}  {status}  {r['backend']:6}  {_usd(r.get('cost_usd'))}  调用 {done} 次  {r['workflow']}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    store = Store(args.home)
    r = store.load_run(args.run)
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return 0
    run_dir = store.runs_dir / r["id"]
    print(f"运行 {r['id']}　{r['workflow']}　{STATUS.get(r['status'], r['status'])}　{r['backend']}　花费 {_usd(r.get('cost_usd'))}")
    print(f"开始 {r['started_at']}　结束 {r['finished_at']}")
    for c in r["calls"]:
        title = f"「{c['title']}」" if c.get("title") else ""
        print(f"\n── {c['node']}{title}　{STATUS.get(c['status'], c['status'])}")
        if c["status"] == "pinned":
            print(f"   采用 {c['pinned'][:12]}")
            continue
        req = c["request"]
        print(f"   {c['endpoint']}　{req['model']}　{req['size']}　{req['quality']}　n={req['n']}")
        for im in req.get("images") or []:
            print(f"   图{im['index']}  {im['name']}：{im['role']}　{im['sha256'][:12]}　{im['width']}×{im['height']}　修改 {im['revision']} 次")
        if req.get("mask"):
            m = req["mask"]
            print(f"   遮罩  {m['name']}　{m['sha256'][:12]}　{m['width']}×{m['height']}")
        print("   提示词：")
        for line in req["prompt"].splitlines():
            print(f"   │ {line}")
        if c.get("usage"):
            u = c["usage"]
            print(f"   用量：输入 {u.get('input_tokens')}，输出 {u.get('output_tokens')}　花费 {_usd(c.get('cost_usd'))}　耗时 {c.get('elapsed_s')}s")
        if c.get("error"):
            print(f"   错误：{c['error']['message']}")
        for o in c.get("outputs") or []:
            print(f"   → {o['sha256'][:12]}  {run_dir / o['file']}")
    for out in r["outputs"]:
        print(f"\n输出 {out.get('title') or out['node']}：{run_dir / out['file']}")
    return 0


# ---------- 辅助 ----------


def _prepare(args: argparse.Namespace) -> tuple[Workflow, Engine]:
    wf = load_template(args.template)
    values: dict[str, Any] = {}
    if args.inputs:
        path = Path(args.inputs)
        for key, value in load_values(path).items():
            slot = wf.slots.get(key)
            values[key] = _absolute(value, path.parent) if slot and slot.kind == "image" else value
    from_cli: dict[str, Any] = {}
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep:
            raise JArtError(f"--set 要写成 键=值：{item}")
        key = key.strip()
        if key in from_cli:
            prev = from_cli[key]
            from_cli[key] = (prev if isinstance(prev, list) else [prev]) + [value]
        else:
            from_cli[key] = value
    values.update(from_cli)
    wf, issues = bind(wf, values, base_dir=Path.cwd())
    if has_errors(issues):
        _print_issues(issues)
        raise _Exit(2)
    for item in args.pin:
        node_id, _, asset = item.partition("=")
        spec = wf.node(node_id.strip())
        if spec is None or spec.type != "generate" or not asset:
            raise JArtError(f"--pin 要写成 生成节点=素材编号：{item}")
        spec.params["pin"] = asset.strip()
    model = FakeImages() if args.fake else OpenAIImages()
    return wf, Engine(Store(args.home), model)


def _absolute(value: Any, base: Path) -> Any:
    if isinstance(value, str) and not value.startswith("asset:") and not Path(value).expanduser().is_absolute():
        return str(base / value)
    if isinstance(value, dict) and isinstance(value.get("path"), str):
        return {**value, "path": _absolute(value["path"], base)}
    return value


def _show_plan(args: argparse.Namespace, wf: Workflow, plan: Plan) -> None:
    if args.json:
        print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))
        return
    print(f"模板：{wf.name}　语言：{plan.language}　计划 {plan.hash[:12]}")
    for i, call in enumerate(plan.calls, 1):
        title = f"「{call.title}」" if call.title else ""
        print(f"\n── 调用 {i}/{len(plan.calls)}：{call.node}{title}")
        if call.pinned:
            print(f"   已采用 {call.pinned[:12]}，不调用模型")
            continue
        pr = call.params
        bits = [f"接口 {call.endpoint}", f"模型 {pr['model']}", f"尺寸 {pr['size']}", f"质量 {pr['quality']}", f"张数 {pr['n']}"]
        if pr.get("background"):
            bits.append(f"背景 {pr['background']}")
        print("   " + "　".join(bits))
        if call.images:
            print("   图片顺序：")
            for im in call.images:
                print(f"     图{im['index']}  {im['name']}：{im['role']}　{source_label(im['source'])}　{_dims(im)}　修改 {im['revision']} 次")
            if call.mask:
                print(f"     遮罩  {call.mask['name']}　{source_label(call.mask['source'])}　{_dims(call.mask)}")
        print("   提示词（原样发送）：")
        for line in call.prompt.splitlines() or [""]:
            print(f"   │ {line}")
        if call.estimate:
            usd = _usd(call.estimate.usd) if call.estimate.usd is not None else "无法预估"
            print(f"   预估：{usd}（{call.estimate.basis}）")
    _print_issues(plan.issues)
    if plan.calls:
        total, unknown = plan.estimated_usd()
        print(f"\n合计预估：{_usd(total)}" + (f"，另有 {unknown} 次调用无法预估" if unknown else ""))


def _print_issues(issues: list[Issue]) -> None:
    if not issues:
        return
    print()
    for issue in sorted(issues, key=lambda i: i.level != "error"):
        print(f"{'错误' if issue.level == 'error' else '警告'}　{issue}")


def _progress(kind: str, data: dict[str, Any]) -> None:
    if kind == "call_start":
        print(f"→ {data['node']} 调用模型中（{data['endpoint']}）…", flush=True)
    elif kind == "call_done":
        out = data["outputs"][0]
        print(f"✓ {data['node']}　{data['elapsed_s']}s　{_usd(data.get('cost_usd'))}　→ {out['sha256'][:12]}", flush=True)
    elif kind == "call_failed":
        print(f"✗ {data['node']}　{data['error']['message']}", flush=True)
    elif kind == "call_pinned":
        print(f"= {data['node']}　采用 {data['pinned'][:12]}，不调用模型", flush=True)


def _dims(entry: dict[str, Any]) -> str:
    if entry.get("width") and entry.get("height"):
        return f"{entry['width']}×{entry['height']}"
    return "尺寸待定"


def _usd(value: float | None) -> str:
    return f"${value:.4f}" if value is not None else "未知"


if __name__ == "__main__":
    sys.exit(main())
