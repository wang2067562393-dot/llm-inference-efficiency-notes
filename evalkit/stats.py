"""评测统计：3σ 幅度门槛与归一化评分。

这两条规则看起来很朴素，但每一条都有若干个「差一点就错」的约定，
而且一旦写错，结论会静默地偏向「看起来通过了」。所以这里把约定
写成代码和异常，而不是注释：

1. 3σ 用的是 **Baseline 的样本标准差（分母 n-1）**。
   不是标准误、不是 Reference 的标准差、也不是配对差值的标准差。
2. 归一化评分的上锚点 U 必须**事先**声明。
   不能用 Reference 的结果反推 U，否则分数恒等于 1，判定失去意义。
3. 小数与百分数不能混用。混用会让 σ 和 Δ 差 100 倍，而程序不会报错。
4. 样本量小于 3 时，3σ 只能是启发式，不能当作统计证明。

所有函数对输入做显式校验，宁可抛错也不返回一个「看起来合理」的数字。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

Direction = Literal["maximize", "minimize"]

__all__ = [
    "Comparison",
    "StatsError",
    "improvement",
    "mean",
    "normalized_score",
    "pairwise_compare",
    "sample_std",
    "sigma_multiple",
]


class StatsError(ValueError):
    """输入不满足统计约定时抛出（样本量、单位、锚点、方向）。"""


def _floats(values: Iterable[float], name: str) -> list[float]:
    out = [float(v) for v in values]
    if not out:
        raise StatsError(f"{name} 为空，无法计算")
    if any(not math.isfinite(v) for v in out):
        raise StatsError(f"{name} 含非有限值（nan/inf）")
    return out


def _check_direction(direction: str) -> Direction:
    if direction not in ("maximize", "minimize"):
        raise StatsError(f"direction 必须是 'maximize' 或 'minimize'，收到 {direction!r}")
    return direction  # type: ignore[return-value]


def mean(values: Iterable[float]) -> float:
    """算术平均。"""
    xs = _floats(values, "values")
    return sum(xs) / len(xs)


def sample_std(values: Iterable[float]) -> float:
    """样本标准差（分母 n-1）。

    3σ 门槛里用的就是它。用总体标准差（分母 n）会系统性低估波动，
    让改善看起来更显著。
    """
    xs = _floats(values, "values")
    if len(xs) < 2:
        raise StatsError("样本标准差至少需要 2 个观测（分母 n-1）")
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def improvement(baseline_mean: float, reference_mean: float, direction: Direction) -> float:
    """Reference 相对 Baseline 的改善量，恒为「越大越好」。

    越高越好的指标：Δ = R̄ − B̄
    越低越好的指标：Δ = B̄ − R̄
    """
    _check_direction(direction)
    if direction == "maximize":
        return reference_mean - baseline_mean
    return baseline_mean - reference_mean


def sigma_multiple(
    baseline: Sequence[float],
    reference: Sequence[float],
    direction: Direction,
) -> float | None:
    """Δ / σ_B，即改善量相当于多少个 Baseline 样本标准差。

    返回 None 表示 Baseline 标准差为 0（所有 seed 完全相同），
    此时「改善倍数」在数学上无定义，不能当成无穷大通过门槛。
    """
    b = _floats(baseline, "baseline")
    r = _floats(reference, "reference")
    if len(b) != len(r):
        raise StatsError(f"成对比较要求样本数一致：baseline={len(b)} reference={len(r)}")
    sigma = sample_std(b)
    if sigma == 0:
        return None
    return improvement(mean(b), mean(r), direction) / sigma


def normalized_score(
    value: float,
    baseline: float,
    upper: float,
    direction: Direction,
) -> float:
    """归一化评分：完成 Baseline → 上锚点 U 之间多少比例的改进。

    越高越好：score = (value − B) / (U − B)
    越低越好：score = (B − value) / (B − U)

    U 必须位于更优的一侧，且不能等于 B（否则分母为 0）。
    """
    _check_direction(direction)
    if baseline == upper:
        raise StatsError("上锚点 U 与 Baseline 相等，归一化分母为 0")
    if direction == "maximize" and upper < baseline:
        raise StatsError("越高的指标，上锚点必须大于 Baseline")
    if direction == "minimize" and upper > baseline:
        raise StatsError("越低的指标，上锚点必须小于 Baseline")
    if direction == "maximize":
        return (value - baseline) / (upper - baseline)
    return (baseline - value) / (baseline - upper)


@dataclass(frozen=True)
class Comparison:
    """一次 Baseline / Reference 成对对照的完整判定结果。"""

    direction: Direction
    n: int
    baseline_mean: float
    baseline_std: float
    reference_mean: float
    reference_std: float
    improvement: float
    sigma_multiple: float | None
    required_sigma: float
    score: float | None
    upper: float | None
    passed_sigma: bool
    passed_score_range: bool | None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "n": self.n,
            "baseline_mean": self.baseline_mean,
            "baseline_std": self.baseline_std,
            "reference_mean": self.reference_mean,
            "reference_std": self.reference_std,
            "improvement": self.improvement,
            "sigma_multiple": self.sigma_multiple,
            "required_sigma": self.required_sigma,
            "score": self.score,
            "upper": self.upper,
            "passed_sigma": self.passed_sigma,
            "passed_score_range": self.passed_score_range,
            "warnings": list(self.warnings),
        }


def pairwise_compare(
    baseline: Sequence[float],
    reference: Sequence[float],
    *,
    direction: Direction = "maximize",
    upper: float | None = None,
    required_sigma: float = 3.0,
    accept_range: tuple[float, float] | None = None,
) -> Comparison:
    """Baseline / Reference 成对对照。

    参数
    ----
    baseline, reference
        同一组 seed 上的一一配对结果，长度必须相同且 ≥ 2。
    direction
        "maximize"（准确率等）或 "minimize"（延迟、误差等）。
    upper
        预先声明的上锚点 U；给 None 时不算归一化分数。
    required_sigma
        幅度门槛，默认 3。
    accept_range
        Reference 的归一化分数应落在的区间，例如 (0.15, 0.8)。
        越界不代表方法差，而是超出校准范围，需要复核锚点。
    """
    _check_direction(direction)
    b = _floats(baseline, "baseline")
    r = _floats(reference, "reference")
    if len(b) != len(r):
        raise StatsError(f"成对比较要求样本数一致：baseline={len(b)} reference={len(r)}")
    if len(b) < 2:
        raise StatsError("至少 2 个 seed 才能估计波动；3 个及以上才谈得上幅度判断")

    b_mean = mean(b)
    r_mean = mean(r)
    b_std = sample_std(b)
    r_std = sample_std(r)
    delta = improvement(b_mean, r_mean, direction)
    sigma = None if b_std == 0 else delta / b_std
    passed_sigma = sigma is not None and sigma >= required_sigma

    score = None
    passed_range: bool | None = None
    if upper is not None:
        score = normalized_score(r_mean, b_mean, upper, direction)
        if accept_range is not None:
            lo, hi = accept_range
            passed_range = lo <= score <= hi

    warnings: list[str] = []
    if len(b) < 3:
        warnings.append("样本量 n<3：3σ 只能作为启发式，不能当作统计证明")
    if b_std == 0:
        warnings.append("Baseline 标准差为 0：改善倍数无定义，需核对 seed 是否真的独立")
    if delta <= 0:
        warnings.append("Reference 未优于 Baseline：不要通过调整锚点或挑 seed 制造达标")
    all_values = b + r
    if any(v > 1.0 for v in all_values) and any(v <= 1.0 for v in all_values):
        warnings.append("疑似小数与百分数混用：标准差与改善量必须统一单位")
    if score is not None and accept_range is not None and passed_range is False:
        warnings.append(
            f"归一化分数 {score:.4f} 超出预声明区间 {accept_range}：优先复核锚点，而不是改结论"
        )

    return Comparison(
        direction=direction,
        n=len(b),
        baseline_mean=b_mean,
        baseline_std=b_std,
        reference_mean=r_mean,
        reference_std=r_std,
        improvement=delta,
        sigma_multiple=sigma,
        required_sigma=required_sigma,
        score=score,
        upper=upper,
        passed_sigma=passed_sigma,
        passed_score_range=passed_range,
        warnings=tuple(warnings),
    )
