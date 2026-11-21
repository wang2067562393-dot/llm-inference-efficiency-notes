# ⚡ LLM Inference Efficiency Notes

> 大模型推理效率的学习笔记，以及配套的、把笔记里的**判定规则**写成代码并加测试的工具。
> 一句话分工：**笔记解释方法为什么可能有效，代码负责判定「这次提升是不是真的」。**

![tests](https://github.com/wang2067562393-dot/llm-inference-efficiency-notes/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![tests](https://img.shields.io/badge/tests-43%20passed-2EA44E)
![status](https://img.shields.io/badge/status-持续更新-F59E0B)

> 🧭 **第一次进来**：先看 [GETTING_STARTED.md](GETTING_STARTED.md)（3 小时主线 + 四问法 + 自查表）。

## 这个仓库是什么 / 不是什么

**是**：
- 三组方法的瓶颈拆解（KV Cache 压缩、视觉 Token 削减、投机解码），每篇都按「瓶颈 → 直觉 → 做法 → 代价 → 失败模式」组织；
- 一篇评测方法论（多 seed 配对、3σ 幅度门槛、归一化锚点、Dev/Private 隔离、reward hacking 清单）；
- 两个能跑的 CLI：`python -m evalkit stats`（成对对照判定）和 `python -m evalkit lint`（评测任务包发布前自检），合计 43 个单元测试。

**不是**：
- 不是论文复现。笔记里的性能数字来自原论文，**我自己的实测还没做**（计划见下方「已知局限」）；
- 不是 benchmark 榜单。这里没有「我测得某方法快 2 倍」这种结论，因为我还没跑。

把这条写在开头，是因为这类仓库最容易犯的错就是：把读论文的理解写得像自己做过的实验。

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

## 🧰 可运行代码

### evalkit：把评测纪律写成代码

笔记里的每一条判定规则，都在 `evalkit/` 里有对应实现——因为规则写在文字里可以自己糊弄自己，写成函数就会在样本量、单位、锚点不合法时直接抛错。

```bash
# 1) 成对对照：3σ 幅度门槛 + 归一化评分
python -m evalkit stats --baseline 0.60 0.61 0.62 \
                        --reference 0.66 0.67 0.68 \
                        --upper 1 --accept-range 0.15 0.8

# 2) 评测任务包发布前自检（13 条规则）
python -m evalkit lint path/to/task --json
```

| 模块 | 覆盖的规则 | 测试 |
|---|---|---|
| `evalkit.stats` | 3σ 用 Baseline 样本标准差（n−1）；上锚点须事先声明；小数/百分数混用检测；σ=0 时返回「无定义」而不是无穷大 | 21 |
| `evalkit.lint` | 答案泄漏（gold/answer/private 命名的文件进选手可见目录）、oracle 泄漏（`solution/` 内容被复制进 `environment/`）、manifest sha256 不符、超时缺失等 13 条 | 16 |

`lint` 只做静态读取，不执行包内代码，所以可以安全地跑在不可信的提交上；有 high 级问题时退出码为 1，可直接接 CI。

### tools/：笔记配套的小工具

| 工具 | 作用 | 依赖 |
|---|---|---|
| `tools/kv_budget.py` | KV Cache 显存估算（层数 / KV 头 / 维度 / 序列长度 / 精度） | 标准库 |
| `tools/attention_numpy.py` | NumPy 手写 Scaled Dot-Product Attention（含 causal mask 与自测） | numpy |

```bash
python -m unittest discover -s tests -v            # evalkit：37 个，零依赖
pip install -r tools/requirements.txt
python -m unittest discover -s tools/tests -v      # tools：6 个
```

## 🧪 评测纪律（这个仓库真正的重点）

1. **3σ 不是假设检验**。它是任务验收的经验门槛，只能说「改善明显大于训练波动」，不能说「有 99.7% 概率有效」。
2. **σ 用 Baseline 的样本标准差（n−1）**。用总体标准差或配对差值标准差，都会让结果看起来更显著。
3. **上锚点必须事先声明**。用 Reference 的成绩反推上锚点，分数恒等于 1，判定失去意义。
4. **质量门与主分分离**。先过质量硬门，门内才比较速度；候选自报的时间和分数不作数。
5. **Dev / Private 隔离不只是文件分开**。真正的问题是「人会不会根据 Private 的反馈继续调」，协议和留痕都要管住。
6. **σ = 0 不是好消息**。所有 seed 完全一致，通常意味着 seed 没真正独立，这时候「无穷倍改善」是危险假信号。

## 已知局限

- 三篇笔记里的性能数字**全部来自原论文**，本仓库没有自己的实测。这是最大的短板，也是下一步要做的事（见下）。
- `attention_numpy.py` 是教学实现，不追求性能，不能当推理内核用。
- `evalkit.stats` 的 3σ 只是幅度门槛；n=3 时它只提供「强/弱信号」，不是统计证明。
- `evalkit.lint` 只覆盖静态可判定的一类问题，不检查「任务的方法空间是否真的成立」。

## 接下来

- [ ] 用 0.5B 级小模型实测 KV Cache 显存曲线，对比 `kv_budget.py` 的估算误差
- [ ] 实测投机解码的接受率—加速比曲线，包括「draft 太小反而变慢」的负结果
- [ ] 视觉 token 削减改用小 VLM（如 SmolVLM）做保留比例—精度曲线，避开 7B 在本机的显存瓶颈
- [ ] `evalkit` 增加「可优化面是否成立」的检查

> 本仓库只记录公开论文与公开工具链的学习理解，不含任何未公开的任务数据、答案、锚点或委托方材料。`evalkit` 的测试数据全部是合成的。
