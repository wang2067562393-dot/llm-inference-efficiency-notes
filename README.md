# ⚡ LLM Inference Efficiency Notes

> 大模型推理效率学习笔记：从 KV Cache 压缩、视觉 Token 削减到投机解码，附实验评测方法论。
> 目标：用一页纸讲清每种方法的「瓶颈 → 直觉 → 做法 → 代价」，并坚持所有结论可复算、可证伪。

![tests](https://github.com/wang2067562393-dot/llm-inference-efficiency-notes/actions/workflows/tests.yml/badge.svg)
![status](https://img.shields.io/badge/status-WIP%20建设中-F59E0B)
![focus](https://img.shields.io/badge/focus-推理加速-6366F1)
![method](https://img.shields.io/badge/method-多seed对照%20%7C%20归一化评分%20%7C%20防污染-16A34A)

## 一张图看懂这些方法在优化什么

自回归推理分两个阶段，瓶颈完全不同：

```
一次推理
├── Prefill（处理 prompt，算力瓶颈，可并行）
│   ├── 文本长上下文：KV Cache 体积随上下文线性膨胀
│   │   └── ① KV Cache 压缩：H2O / SnapKV / PyramidKV / ChunkKV …
│   └── 多模态：图像 token 动辄数百个，prefill 很贵
│       └── ② 视觉 Token 削减：FastV / VisionZip …
└── Decode（逐 token 生成，显存带宽瓶颈，batch=1 时算力闲置）
    └── ③ 投机解码：小模型起草、大模型一次验证
        └── EAGLE / EAGLE-2 / EAGLE-3、Medusa、Lookahead …
```

| 方法族 | 阶段 | 核心思路 | 主要风险 |
|---|---|---|---|
| KV Cache 压缩 | Prefill + Decode | 丢弃/合并「不重要」的历史 KV | 长程信息丢失，"大海捞针"题掉点 |
| 视觉 Token 削减 | Prefill | 576 个图像 patch 大量冗余，保留显著的 | 小目标 / OCR 类问题掉点 |
| 投机解码 | Decode | 用草稿换并行验证，接受长度决定收益 | 猜错反变慢，需要质量保真门 |

## 📒 笔记索引

1. [KV Cache 压缩](notes/01-kv-cache-compression.md)：KV Cache 为什么贵、按召回/按观测窗口的淘汰路线
2. [视觉 Token 削减](notes/02-visual-token-reduction.md)：LLaVA 架构与 patch 重要性选择
3. [投机解码与 EAGLE](notes/03-speculative-decoding.md)：接受率为什么决定墙钟收益，动态草稿深度
4. [实验与评测方法论](notes/04-evaluation-methodology.md)：配对多 seed、样本标准差、3σ 启发式、归一化锚点、Dev/Private 隔离、reward hacking
5. [参考资料清单](references.md)：公开论文与开源仓库

## 🧰 可运行小工具（`tools/`，附单元测试与 CI）

光有文字不够，笔记里的公式都落成了可执行代码：

| 工具 | 作用 | 示例 |
|---|---|---|
| `tools/kv_budget.py` | KV Cache 显存估算（层数/KV 头/维度/序列长度/精度） | `python tools/kv_budget.py` → 32 层 GQA(8 头)、16K、BF16 ≈ 2.0 GiB |
| `tools/paired_eval.py` | 配对多 seed 统计：样本标准差、Δ/σ、归一化分 | `python tools/paired_eval.py --baseline .60 .61 .62 --reference .66 .67 .68 --upper-anchor 1` |
| `tools/attention_numpy.py` | NumPy 手写 Scaled Dot-Product Attention（含 causal mask 与自测） | `python tools/attention_numpy.py --selftest` |

```bash
pip install -r tools/requirements.txt
python -m unittest discover -s tools/tests -v   # 9 个测试
```

## 学习原则

- **先自己算，再对答案**：所有统计量（均值、样本标准差、Δ/σ、归一化分）亲手复算；
- **质量与速度分开度量**：加速结论必须在质量保真门下讨论，不用一个加权分掩盖掉点；
- **公开集选法，私有集验收**：绝不用最终测试集的反馈继续调方法；
- **失败也记录**：负结果和被证伪的假说与正结果同等重要。

> ⚠️ 本仓库只记录公开论文与开源代码的学习理解，不含任何未公开的任务数据与内部材料。
