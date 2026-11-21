# ⚡ LLM Inference Efficiency Notes

> 三组推理加速方法的瓶颈拆解，外加一个**把「怎么判断提升是不是真的」写成代码**的工具。
> 笔记解释方法为什么可能有效；代码负责判定结论是否成立。

![tests](https://github.com/wang2067562393-dot/llm-inference-efficiency-notes/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![tests](https://img.shields.io/badge/tests-43%20passed-2EA44E)
![deps](https://img.shields.io/badge/evalkit-零依赖-16A34A)

**如果只看一个文件，看 [`notes/05-case-study.md`](notes/05-case-study.md)** —— 一次完整的「A 比 B 好」判定，
包含四个「看起来通过但不算数」的反例。其余内容都是它的支撑材料。

## 两条命令

```bash
git clone https://github.com/wang2067562393-dot/llm-inference-efficiency-notes
cd llm-inference-efficiency-notes

# 1) 成对对照判定：3σ 幅度门槛 + 归一化评分（零依赖）
python -m evalkit stats --baseline 12.4 12.1 12.7 \
                        --reference 8.9 8.7 9.1 \
                        --direction minimize --upper 0 --accept-range 0.15 0.8

# 2) 评测任务包发布前自检：13 条静态规则
python -m evalkit lint tests/fixtures/leaky_task
```

```
Δ/σ_B           11.67   （门槛 3）  通过
归一化评分      0.2823   （上锚点 U=0.0）  区间内
```

## 目录

```
notes/           5 篇笔记：KV Cache / 视觉 Token / 投机解码 / 评测方法论 / 端到端案例
evalkit/         评测判定与任务包自检（stats / lint / cli），零第三方依赖
tools/           笔记配套小工具：KV 显存估算、NumPy 手写 Attention（需要 numpy）
tests/           43 个单元测试，一条命令跑完
```

```bash
python -m unittest discover -s tests -v
```

## 笔记索引

1. [KV Cache 压缩](notes/01-kv-cache-compression.md)：KV 为什么贵、按召回/按观测窗口的淘汰路线
2. [视觉 Token 削减](notes/02-visual-token-reduction.md)：LLaVA 架构与 patch 重要性选择
3. [投机解码与 EAGLE](notes/03-speculative-decoding.md)：接受率为什么决定墙钟收益
4. [实验与评测方法论](notes/04-evaluation-methodology.md)：配对多 seed、3σ、归一化锚点、Dev/Private 隔离
5. [**端到端案例**](notes/05-case-study.md)：一次完整判定 + 四个假提升反例

## evalkit 是什么

把评测纪律写成会抛错的代码，因为规则写在文档里会被人情绕过。

| 模块 | 覆盖的规则 | 测试 |
|---|---|---|
| `evalkit.stats` | 3σ 用 Baseline 样本标准差（n−1）；上锚点须事先声明；单位混用检测；σ=0 返回「无定义」而非无穷大 | 21 |
| `evalkit.lint` | 答案泄漏、oracle 泄漏（`solution/` 内容被复制进 `environment/`）、manifest 哈希不符、超时缺失等 13 条 | 16 |

`lint` 只做静态读取，不执行包内代码，可以安全跑在不可信的提交上；有 high 级问题时退出码为 1，可直接接 CI。

## 这个仓库是什么 / 不是什么

**是**：方法瓶颈的拆解、评测判定的可运行实现、一个端到端案例。

**不是**：不是论文复现。**笔记里的性能数字全部来自原论文，本仓库没有自己的实测。**
这条写在最前面，因为这类仓库最容易犯的错就是把读论文的理解写得像自己做过的实验。

## 已知局限

- 三篇方法笔记没有自己的实测（下一步：用 0.5B 模型测 KV Cache 显存曲线与投机解码接受率）
- `attention_numpy.py` 是教学实现，不追求性能
- `evalkit.stats` 的 3σ 只是幅度门槛，n=3 时只提供「强/弱信号」，不是统计证明
- `evalkit.lint` 只覆盖静态可判定的一类问题，不检查「任务的方法空间是否真的成立」

## 接下来

- [ ] 0.5B 模型实测 KV Cache 显存曲线，对比 `tools/kv_budget.py` 的估算误差
- [ ] 实测投机解码的接受率—加速比曲线，包含「draft 太小反而变慢」的负结果
- [ ] 视觉 token 削减改用小 VLM 做保留比例—精度曲线
- [ ] `evalkit` 增加「可优化面是否成立」的检查

> 本仓库只含公开论文的理解与合成数据，不含任何未公开的任务数据、答案、锚点或委托方材料。
