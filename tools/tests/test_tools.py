# -*- coding: utf-8 -*-
"""tools/ 下脚本的单元测试。"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

from kv_budget import gqa_ratio, kv_bytes_per_token, kv_cache_gb  # noqa: E402
from attention_numpy import scaled_dot_product_attention, softmax  # noqa: E402


class TestKVBudget(unittest.TestCase):
    def test_per_token_formula(self):
        # 32 层、8 KV 头、128 维、bf16：2*2*32*8*128 = 131072 B = 128 KiB
        self.assertEqual(kv_bytes_per_token(32, 8, 128, "bf16"), 131072)

    def test_gqa_and_scale(self):
        self.assertAlmostEqual(gqa_ratio(32, 8), 4.0)
        # 序列翻倍，KV 翻倍
        a = kv_cache_gb(28, 8, 128, 8192, dtype="fp16")
        b = kv_cache_gb(28, 8, 128, 16384, dtype="fp16")
        self.assertAlmostEqual(b / a, 2.0, places=6)

    def test_int4_quarter_of_fp16(self):
        a = kv_cache_gb(32, 8, 128, 4096, dtype="fp16")
        b = kv_cache_gb(32, 8, 128, 4096, dtype="int4")
        self.assertAlmostEqual(a / b, 4.0, places=6)


class TestAttention(unittest.TestCase):
    def test_softmax(self):
        s = softmax(np.array([[0.0, 0.0, 10.0]]))
        self.assertTrue(np.allclose(s.sum(axis=-1), 1.0))
        # 其余两项 logit 为 0，相对 10 仍保留极小质量，故阈值取 0.9999
        self.assertGreater(s[0, 2], 0.9999)

    def test_causal_mask(self):
        rng = np.random.default_rng(2)
        q = k = v = rng.normal(size=(5, 8))
        _, w = scaled_dot_product_attention(q, k, v, causal=True)
        for i in range(5):
            self.assertTrue(np.all(w[i, i + 1 :] == 0))
            self.assertTrue(math.isclose(w[i, : i + 1].sum(), 1.0, abs_tol=1e-9))

    def test_shape_and_order(self):
        rng = np.random.default_rng(3)
        q, k, v = (rng.normal(size=(4, 8)) for _ in range(3))
        out, _ = scaled_dot_product_attention(q, k, v)
        self.assertEqual(out.shape, (4, 8))


if __name__ == "__main__":
    unittest.main()
