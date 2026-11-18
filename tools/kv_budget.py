#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kv_budget.py —— KV Cache 显存量级估算。

公式（每 token、全部层）：
    bytes = 2(K+V) * bytes_per_elem * n_layers * n_kv_heads * head_dim * L * batch

纯标准库，既能 import 也能命令行跑。
"""
from __future__ import annotations

import argparse

BYTES_PER_ELEM = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}


def kv_bytes_per_token(n_layers: int, n_kv_heads: int, head_dim: int, dtype: str = "bf16") -> float:
    """单条序列每生成 1 个 token 新增的 KV 字节数。"""
    bpe = BYTES_PER_ELEM[dtype]
    return 2 * bpe * n_layers * n_kv_heads * head_dim


def kv_cache_gb(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch: int = 1,
    dtype: str = "bf16",
) -> float:
    return kv_bytes_per_token(n_layers, n_kv_heads, head_dim, dtype) * seq_len * batch / 1024**3


def gqa_ratio(n_attn_heads: int, n_kv_heads: int) -> float:
    """GQA 相对 MHA 把 KV 缩小的倍数。"""
    return n_attn_heads / n_kv_heads


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=8, help="MHA 时等于注意力头数；GQA 时更小")
    ap.add_argument("--attn-heads", type=int, default=32)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=16384)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--dtype", choices=list(BYTES_PER_ELEM), default="bf16")
    args = ap.parse_args()

    per_tok_kb = kv_bytes_per_token(args.layers, args.kv_heads, args.head_dim, args.dtype) / 1024
    total_gb = kv_cache_gb(
        args.layers, args.kv_heads, args.head_dim, args.seq_len, args.batch, args.dtype
    )
    print(f"每 token KV: {per_tok_kb:.1f} KiB")
    print(f"L={args.seq_len}, batch={args.batch}, {args.dtype}: KV Cache ≈ {total_gb:.2f} GiB")
    print(f"GQA 相对 MHA 缩小: {gqa_ratio(args.attn_heads, args.kv_heads):.1f}x")


if __name__ == "__main__":
    main()
