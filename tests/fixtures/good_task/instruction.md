# 任务：降低自回归解码的 ms/token

在固定的小模型上，修改 `method.py` 中的解码策略，让每 token 平均耗时下降，
同时保持输出质量不低于质量门。

## 评分

主分为相对 Baseline 的耗时下降：
`score = (baseline_ms_per_token - candidate_ms_per_token) / baseline_ms_per_token`

质量门：候选输出的接受长度不得低于 Baseline 的 95%。
