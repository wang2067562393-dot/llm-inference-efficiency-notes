#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paired_eval.py —— 配对多 seed 对照的统计工具。

实现 notes/04-evaluation-methodology.md 里的两个量：
  1) 改善倍数 k = Δ / σ_B（σ_B 为 Baseline 样本标准差，ddof=1）
  2) 归一化分 score（自动处理越高/越低越好两种方向）

纯标准库。
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass


def sample_std(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        raise ValueError("样本标准差至少需要 2 个观测")
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


@dataclass
class PairedResult:
    baseline_mean: float
    reference_mean: float
    delta: float
    baseline_std: float
    sigma_multiple: float | None
    normalized_score: float


def paired_evaluate(
    baseline: list[float],
    reference: list[float],
    upper_anchor: float,
    higher_is_better: bool = True,
) -> PairedResult:
    if len(baseline) != len(reference) or not baseline:
        raise ValueError("B/R 必须非空且等长（一一配对）")
    b_mean = sum(baseline) / len(baseline)
    r_mean = sum(reference) / len(reference)
    sigma = sample_std(baseline)
    delta = (r_mean - b_mean) if higher_is_better else (b_mean - r_mean)
    if higher_is_better:
        score = (r_mean - b_mean) / (upper_anchor - b_mean)
    else:
        score = (b_mean - r_mean) / (b_mean - upper_anchor)
    k = (delta / sigma) if sigma > 0 else None
    return PairedResult(b_mean, r_mean, delta, sigma, k, score)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", nargs="+", type=float, required=True, help="如 8.57 8.45 8.47")
    ap.add_argument("--reference", nargs="+", type=float, required=True, help="如 4.85 4.83 4.85")
    ap.add_argument("--upper-anchor", type=float, required=True, help="准确率可取 1.0；延迟可取 0")
    ap.add_argument("--lower-is-better", action="store_true", help="延迟/成本类指标")
    args = ap.parse_args()

    res = paired_evaluate(
        args.baseline, args.reference, args.upper_anchor, higher_is_better=not args.lower_is_better
    )
    print(json.dumps(res.__dict__, indent=2, ensure_ascii=False))
    print("\n判定提示：Δ/σ≥3 过项目 3σ 门槛（≥5 为强信号，非假设检验）；"
          "Reference 归一化分通常要求落在 0.15–0.8。")


if __name__ == "__main__":
    main()
