# gpt-image-2.5 接口实测要点（2026-10-07，用户本机实验）

## 型号与价格
- `gpt-image-2.5-sunburst`：官方称"能力最强、编辑精度最高"；`gpt-image-2.5-flare`：官方称"最快"。两者价格、质量档、速率限制相同（官方文档；flare 未实测）。
- 价格（官方 pricing）：图像输出 $30/百万 token，图像输入 $8，文本输入 $5；Batch 半价。
- 速率（最低档 Build）：20 张/分钟，250k TPM。
- 不支持 seed，同一输入每次结果不同。

## 接口
- 无参考图：`POST /v1/images/generations`（JSON）。
- 有参考图或编辑：`POST /v1/images/edits`（multipart），多张图用重复的 `image[]` 字段，最多 16 张。图片顺序就是提示里"图1/图2…"的编号。
- 参数：`model`、`prompt`、`size`（任意 WxH，边长须为 16 的倍数，如 `2048x1024`）、`quality`（low/medium/high/xhigh/max/auto）、`n`、`output_format`（png/webp/jpeg）、`background`（transparent 须配 png/webp）。
- 直接调 Images API 时，提示原样到达。不要走 Responses API 的 image_generation 工具：主模型会改写提示（返回 revised_prompt）。

## 实测数据
- 2048×1024 high：每张 1180 个图像输出 token，25–35 秒。
- 参考图按图像 token 计输入：一张 1086×1448 立绘约 2904 token；三张设定图加一张参考图约 6000 token。
- 透明背景（background=transparent）有真 Alpha，但有两个问题，进游戏前必须后处理：
  - 主体内部 Alpha 全是 253/254，没有 255；
  - 背景散布 Alpha 1–8 的彩色淡雾，占画面 3–7%。
  - 处理：极淡值（如 ≤8）归 0，主体（如 ≥250）归 255，只在边缘 1–3px 保留抗锯齿。

## 遮罩编辑（局部修图）
- 在 edits 里加 `mask` 字段：一张与目标图同尺寸的 PNG，**Alpha=0 的区域是允许修改的区域**，其余 Alpha=255。实测 mask 可以和多张参考图同时传（目标图放第一张，mask 对应第一张图），两次调用都成功。ComfyUI 的 GPT Image 节点限制用 mask 时只能一张图，那是节点自己的限制，不是接口限制。
- 官方说明 mask 只是引导，不保证严格按边界：模型实际会把整张图重画一遍，遮罩外也有轻微变化（实测平均色差约 7.5/255）。
- **不要把修改区域抠下来羽化贴回原图**：接缝处会出现重影和模糊，用户明确否决。直接使用模型输出。
- 遮罩要避开不想动的元素：遮罩若框到相邻物件（实测框到箭尾），模型会把它一起改掉。
- 一次编辑可以修多处，提示按编号逐条写"错在哪＋正确画法"。修改次数受规则约束（见规则文档：参考图只能是原图或改过一次的图）。

## 参考实现（Python，标准库＋Pillow）
```python
import base64, json, urllib.request, uuid, io
from PIL import Image

def multipart(fields, files):
    b = uuid.uuid4().hex; out = bytearray()
    for k, v in fields.items():
        out += f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    for k, (name, data, ctype) in files:
        out += f'--{b}\r\nContent-Disposition: form-data; name="{k}"; filename="{name}"\r\nContent-Type: {ctype}\r\n\r\n'.encode() + data + b'\r\n'
    out += f'--{b}--\r\n'.encode()
    return bytes(out), f'multipart/form-data; boundary={b}'

def edit(prompt, images, key, mask=None, size="2048x1024", quality="high"):
    """images: [(filename, png_bytes)]，顺序即"图1、图2…"；mask: png_bytes（Alpha=0 处可改）"""
    fields = {"model": "gpt-image-2.5-sunburst", "prompt": prompt, "size": size,
              "quality": quality, "n": "1", "output_format": "png"}
    files = [("image[]", (n, d, "image/png")) for n, d in images]
    if mask is not None:
        files.append(("mask", ("mask.png", mask, "image/png")))
    body, ctype = multipart(fields, files)
    req = urllib.request.Request("https://api.openai.com/v1/images/edits", data=body,
                                 headers={"Authorization": "Bearer " + key, "Content-Type": ctype})
    with urllib.request.urlopen(req, timeout=900) as r:
        resp = json.load(r)
    return base64.b64decode(resp["data"][0]["b64_json"]), resp.get("usage")
```
- 在云端环境里，key 由网络密钥在请求离开时注入（绑定 api.openai.com），代码不要读取或打印 key；本地跑时从仓库外的配置文件或环境变量读取。
- 每次调用应留底：最终提示全文、图片顺序与哈希、参数、用量、耗时、输出哈希。
