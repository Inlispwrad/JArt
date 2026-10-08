# JArt 开发约定

- 引导式生图平台，个人 side project。设计与规则在 `docs/`；出图规则以 `docs/图像生成规则_gpt-image-2.5.md` 为准，平台只实现规则，不新增、不修改规则。
- 文档里"用户定 / 用户原话"是用户确认过的；"推导 / 建议 / 待定"没有确认，不当定论。新写的文档沿用这套标注。
- 结构节点拼提示词是固定规则的纯函数：不经 LLM，不改写、不增补用户文字。改拼接格式时同步 `docs/引擎说明.md` 和 `tests/test_structure.py`。
- OpenAI key：代码里不写 key，也不读取或打印 key。云端环境由网络密钥在出口注入；本机用 `OPENAI_API_KEY`。
- 真实调用要花钱：测试一律用 `FakeImages` 或 `httpx.MockTransport`。
- 文档和界面文字用中文。
- 命令：`uv sync`、`uv run pytest`、`uv run jart …`。
