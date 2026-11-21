"""evalkit：agentic 评测任务包的统计判定与发布前自检。"""

from evalkit.stats import (
    Comparison,
    StatsError,
    improvement,
    mean,
    normalized_score,
    pairwise_compare,
    sample_std,
    sigma_multiple,
)

__version__ = "0.1.0"

__all__ = [
    "Comparison",
    "StatsError",
    "improvement",
    "mean",
    "normalized_score",
    "pairwise_compare",
    "sample_std",
    "sigma_multiple",
    "__version__",
]
