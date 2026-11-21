# 参考资料（公开论文与开源项目）

## KV Cache / 长上下文

- **H2O**: *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models* (NeurIPS 2023)
- **StreamingLLM**: *Efficient Streaming Language Models with Attention Sinks* (ICLR 2024)
- **SnapKV**: *SnapKV: LLM Knows What You are Looking for Before Generation* (2024) — <https://github.com/FasterDecoding/SnapKV>
- **PyramidKV**: *PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling* (2024)
- KVPress：开源 KV 压缩统一框架 — <https://github.com/IBM/kvpress>
- GQA：*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (2023)

## 多模态视觉 Token 削减

- **FastV**: *Vision Language Models Chew and Choke: On the Unaligned Visual Tokens and Efficiency Trade-offs* (2024)
- **VisionZip**: *VisionZip: Longer is Better but Not Necessary in Vision Language Models* (2024) — <https://github.com/SYuan03/VisionZip>
- **LLaVA-1.5**: *Improved Baselines with Visual Instruction Tuning* (CVPR 2024) — <https://github.com/haotian-liu/LLaVA>
- SparseVLM、PruMerge 等 token 剪枝/合并后续工作

## 投机解码

- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* (ICML 2023)
- **EAGLE**: *EAGLE: Speculative Sampling via Feature Fusion* (ICML 2024) — <https://github.com/SafeAILab/EAGLE>
- **EAGLE-2 / EAGLE-3**: <https://github.com/SafeAILab/EAGLE> 官方仓库
- **Medusa**: *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads* (2024)
- vLLM  speculative decoding 文档 — <https://docs.vllm.ai/>

## Agent / 研究能力评测（方法论对照）

- **SWE-bench**: *Can LLMs Resolve Real-World GitHub Issues?* (ICLR 2024)
- **MLE-Bench**: *MLE-Bench: Evaluating Machine Learning Agents on Machine Learning Engineering* (2024)
- **Harbor**：agentic 任务的格式与运行框架 — <https://github.com/harbor-framework/harbor>

## 工具链

- PyTorch、HuggingFace Transformers、vLLM、llama.cpp、FlashAttention
