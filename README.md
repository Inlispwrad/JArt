# JArt

引导式生图平台：节点里写提示词、接参考图，结构节点按规则拼成最终提示词，预览确认后调用模型，结果可以接到下一个节点。

当前进度：**P1 引擎（本机版）**——节点执行、链式生成、gpt-image-2.5 直调、后处理、计量留底，命令行使用。

```bash
uv sync
uv run jart presets                 # 官方预设
uv run jart plan swap -i 输入.toml   # 预览：提示词全文、图片顺序、预估费用、检查结果
uv run jart run  swap -i 输入.toml   # 确认后运行；加 --fake 用假模型演练
uv run pytest
```

文档（`docs/`）：

- [设计案](docs/设计案.md)、[产品流程](docs/产品流程.md)
- [图像生成规则](docs/图像生成规则_gpt-image-2.5.md)：结构节点实现的就是这套规则
- [API 实测要点](docs/API实测要点.md)
- [引擎说明](docs/引擎说明.md)：节点、拼接规则、检查、留底、费用
