"""命令行入口。

    python -m evalkit stats --baseline .60 .61 .62 --reference .66 .67 .68 --upper 1
    python -m evalkit lint path/to/task --json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from evalkit.lint import format_report, lint_task
from evalkit.stats import Comparison, StatsError, pairwise_compare


def _cmd_stats(args: argparse.Namespace) -> int:
    try:
        result = pairwise_compare(
            args.baseline,
            args.reference,
            direction=args.direction,
            upper=args.upper,
            required_sigma=args.required_sigma,
            accept_range=tuple(args.accept_range) if args.accept_range else None,
        )
    except StatsError as exc:
        print(f"统计输入不合法：{exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0

    _print_comparison(result)
    return 0


def _print_comparison(c: Comparison) -> None:
    def fmt(value: float | None, digits: int = 4) -> str:
        return "—" if value is None else f"{value:.{digits}f}"

    print(f"方向            {c.direction}   seed 数 n={c.n}")
    print(f"Baseline 均值   {c.baseline_mean:.6f}   样本标准差 σ_B={c.baseline_std:.6f}")
    print(f"Reference 均值  {c.reference_mean:.6f}   样本标准差 {c.reference_std:.6f}")
    print(f"改善量 Δ        {c.improvement:+.6f}")
    print(f"Δ/σ_B           {fmt(c.sigma_multiple, 2)}   （门槛 {c.required_sigma:g}）"
          f"  {'通过' if c.passed_sigma else '未通过'}")
    if c.score is not None:
        print(f"归一化评分      {c.score:.4f}   （上锚点 U={c.upper}）"
              f"  {'' if c.passed_score_range is None else ('区间内' if c.passed_score_range else '超出区间')}")
    for warning in c.warnings:
        print(f"⚠ {warning}")


def _cmd_lint(args: argparse.Namespace) -> int:
    try:
        findings = lint_task(args.path)
    except ValueError as exc:
        print(f"无法检查：{exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps([f.as_dict() for f in findings], ensure_ascii=False, indent=2))
    else:
        print(format_report(findings))
    return 1 if any(f.severity == "high" for f in findings) else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalkit", description="agentic 评测任务包工具")
    sub = parser.add_subparsers(dest="command", required=True)

    stats = sub.add_parser("stats", help="Baseline / Reference 成对对照（3σ + 归一化评分）")
    stats.add_argument("--baseline", type=float, nargs="+", required=True, help="Baseline 各 seed 指标")
    stats.add_argument("--reference", type=float, nargs="+", required=True, help="Reference 各 seed 指标")
    stats.add_argument("--direction", choices=("maximize", "minimize"), default="maximize")
    stats.add_argument("--upper", type=float, default=None, help="预先声明的上锚点 U")
    stats.add_argument("--required-sigma", type=float, default=3.0, help="幅度门槛，默认 3")
    stats.add_argument("--accept-range", type=float, nargs=2, metavar=("LO", "HI"),
                       help="归一化分数的预声明区间，如 0.15 0.8")
    stats.add_argument("--json", action="store_true")
    stats.set_defaults(func=_cmd_stats)

    lint = sub.add_parser("lint", help="任务包发布前静态自检")
    lint.add_argument("path", help="任务包根目录")
    lint.add_argument("--json", action="store_true")
    lint.set_defaults(func=_cmd_lint)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
