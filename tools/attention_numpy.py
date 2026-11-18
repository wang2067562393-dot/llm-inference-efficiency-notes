#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""attention_numpy.py —— 用 NumPy 手写单头 Scaled Dot-Product Attention。

目的：把 Transformer 最核心的一步从框架里剥出来，确认每一个张量维度都对得上。

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + mask) V

运行：
    python attention_numpy.py            # 跑内置小例子与自测
    python attention_numpy.py --selftest
"""
from __future__ import annotations

import argparse

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定 softmax：减去行内最大值。"""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    causal: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 (output, weights)。

    q,k,v 形状均为 [seq, d_k]（单头）。output 形状 [seq, d_k]，weights [seq, seq]。
    """
    if q.shape != k.shape or k.shape != v.shape:
        raise ValueError(f"形状不一致: {q.shape} {k.shape} {v.shape}")
    seq, d_k = q.shape
    scores = (q @ k.T) / np.sqrt(d_k)  # [seq, seq]
    if causal:
        mask = np.triu(np.ones((seq, seq), dtype=bool), k=1)
        scores = np.where(mask, -np.inf, scores)
    weights = softmax(scores, axis=-1)
    return weights @ v, weights


def _demo() -> None:
    rng = np.random.default_rng(0)
    seq, d = 4, 8
    q = rng.normal(size=(seq, d))
    k = rng.normal(size=(seq, d))
    v = rng.normal(size=(seq, d))
    out, w = scaled_dot_product_attention(q, k, v, causal=True)
    print("causal attention weights:\n", np.round(w, 3))
    print("output shape:", out.shape)


def _selftest() -> None:
    # 1) softmax 行和为 1
    x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 10.0]])
    s = softmax(x)
    assert np.allclose(s.sum(axis=-1), 1.0)
    # 2) 大值主导但不为 NaN
    assert s[1, 2] > 0.9999
    # 3) causal：第 i 个 query 看不到未来位置
    seq, d = 3, 4
    rng = np.random.default_rng(1)
    q, k, v = (rng.normal(size=(seq, d)) for _ in range(3))
    _, w = scaled_dot_product_attention(q, k, v, causal=True)
    for i in range(seq):
        assert np.all(w[i, i + 1 :] == 0.0)
        assert np.isclose(w[i, : i + 1].sum(), 1.0)
    # 4) 非 causal 一般不为 0
    _, w2 = scaled_dot_product_attention(q, k, v, causal=False)
    assert np.all(w2 > 0)
    # 5) 输出形状
    out, _ = scaled_dot_product_attention(q, k, v)
    assert out.shape == (seq, d)
    print("selftest OK: 5/5 assertions passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        _demo()
        _selftest()
