from __future__ import annotations

from jart.cli import main
from jart.store import Store


def test_presets_listing(capsys):
    assert main(["presets"]) == 0
    out = capsys.readouterr().out
    assert "swap" in out and "chain" in out and "text" in out
    assert main(["presets", "swap"]) == 0
    assert "物件指向替换" in capsys.readouterr().out


def test_plan_shows_the_exact_prompt(tmp_path, capsys):
    args = ["plan", "text", "--home", str(tmp_path / "home"), "-s", "style=画法段", "-s", "content=一只猫坐在窗台上。"]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "│ 画法段\n   │ \n   │ 一只猫坐在窗台上。" in out
    assert "合计预估" in out


def test_missing_input_exits_with_2(tmp_path, capsys):
    assert main(["plan", "text", "--home", str(tmp_path)]) == 2
    assert "缺少输入：画法段（style）" in capsys.readouterr().out


def test_fake_run_then_runs_and_show(tmp_path, capsys):
    home = tmp_path / "home"
    args = ["run", "text", "--home", str(home), "--fake", "-y", "-s", "style=画法段", "-s", "content=一只猫。", "--out", str(tmp_path / "out")]
    assert main(args) == 0
    runs = Store(home).list_runs()
    assert len(runs) == 1 and runs[0]["status"] == "succeeded"
    assert (tmp_path / "out" / "out.png").is_file()
    capsys.readouterr()
    assert main(["runs", "--home", str(home)]) == 0
    assert runs[0]["id"] in capsys.readouterr().out
    assert main(["show", runs[0]["id"][:15], "--home", str(home)]) == 0
    assert "│ 一只猫。" in capsys.readouterr().out


def test_run_without_tty_needs_yes(tmp_path, capsys):
    args = ["run", "text", "--home", str(tmp_path), "--fake", "-s", "style=画法段", "-s", "content=一只猫。"]
    assert main(args) == 2
    assert "--yes" in capsys.readouterr().err


def test_inputs_file_paths_are_relative_to_the_file(tmp_path, make_png, capsys):
    make_png("in/char.png")
    make_png("in/dagger.png")
    make_png("in/ref.png")
    inputs = tmp_path / "in" / "inputs.toml"
    inputs.write_text(
        'char = "char.png"\nitem = "dagger.png"\nref = "ref.png"\nitem_role = "匕首设定"\n'
        'style = "画法段"\nreplace = ["把{{ref}}里的刺客换成{{char}}。", "把{{ref}}里的短刀换成{{item}}。"]\n'
        'action = "刺客冲向右侧。"\n',
        "utf-8",
    )
    assert main(["plan", "swap", "-i", str(inputs), "--home", str(tmp_path / "home")]) == 0
    assert "图1  char：人设" in capsys.readouterr().out
