import math
import unittest

from evalkit.stats import (
    StatsError,
    improvement,
    mean,
    normalized_score,
    pairwise_compare,
    sample_std,
    sigma_multiple,
)


class TestBasics(unittest.TestCase):
    def test_mean(self):
        self.assertAlmostEqual(mean([1.0, 2.0, 3.0]), 2.0)

    def test_sample_std_uses_n_minus_one(self):
        # 三个值偏离 0.01 / 0 / +0.01，样本标准差正好 0.01
        self.assertAlmostEqual(sample_std([0.60, 0.61, 0.62]), 0.01, places=12)

    def test_sample_std_needs_two_points(self):
        with self.assertRaises(StatsError):
            sample_std([1.0])

    def test_non_finite_rejected(self):
        with self.assertRaises(StatsError):
            mean([1.0, math.nan])

    def test_improvement_directions(self):
        self.assertAlmostEqual(improvement(0.61, 0.67, "maximize"), 0.06)
        # 越低越好：8.49 → 4.84 是正改善
        self.assertAlmostEqual(improvement(8.494, 4.842, "minimize"), 3.652)


class TestSigmaMultiple(unittest.TestCase):
    def test_documented_example_is_6x(self):
        k = sigma_multiple([0.60, 0.61, 0.62], [0.66, 0.67, 0.68], "maximize")
        self.assertIsNotNone(k)
        self.assertAlmostEqual(k, 6.0, places=6)

    def test_real_case_minimize(self):
        # 真实交付里的形状：ms/token 从 ~8.49 降到 ~4.84，Baseline 波动极小
        baseline = [8.568, 8.447, 8.466]
        reference = [4.846, 4.829, 4.851]
        k = sigma_multiple(baseline, reference, "minimize")
        self.assertIsNotNone(k)
        self.assertGreater(k, 50.0)  # 审稿人一眼能看出这是强证据
        self.assertAlmostEqual(k, improvement(mean(baseline), mean(reference), "minimize")
                               / sample_std(baseline), places=9)

    def test_zero_sigma_returns_none_not_infinity(self):
        self.assertIsNone(sigma_multiple([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], "maximize"))

    def test_length_mismatch_rejected(self):
        with self.assertRaises(StatsError):
            sigma_multiple([1.0, 2.0, 3.0], [1.0, 2.0], "maximize")


class TestNormalizedScore(unittest.TestCase):
    def test_documented_example(self):
        score = normalized_score(0.67, 0.61, 1.0, "maximize")
        self.assertAlmostEqual(score, 0.06 / 0.39, places=12)

    def test_minimize_with_zero_upper(self):
        score = normalized_score(4.842, 8.494, 0.0, "minimize")
        self.assertAlmostEqual(score, 3.652 / 8.494, places=6)

    def test_upper_equal_baseline_rejected(self):
        with self.assertRaises(StatsError):
            normalized_score(0.7, 0.6, 0.6, "maximize")

    def test_upper_on_wrong_side_rejected(self):
        with self.assertRaises(StatsError):
            normalized_score(0.7, 0.6, 0.5, "maximize")
        with self.assertRaises(StatsError):
            normalized_score(0.5, 0.6, 0.7, "minimize")

    def test_score_can_be_negative(self):
        self.assertLess(normalized_score(0.55, 0.60, 1.0, "maximize"), 0)


class TestPairwiseCompare(unittest.TestCase):
    def test_full_pass(self):
        c = pairwise_compare(
            [0.60, 0.61, 0.62],
            [0.66, 0.67, 0.68],
            upper=1.0,
            accept_range=(0.15, 0.8),
        )
        self.assertTrue(c.passed_sigma)
        self.assertTrue(c.passed_score_range)
        self.assertAlmostEqual(c.score or 0.0, 0.1538, places=3)
        self.assertEqual(c.warnings, ())

    def test_insufficient_improvement_fails_gate(self):
        c = pairwise_compare(
            [0.60, 0.62, 0.58],
            [0.605, 0.625, 0.585],
            upper=1.0,
        )
        self.assertFalse(c.passed_sigma)

    def test_small_n_warns(self):
        c = pairwise_compare([0.60, 0.62], [0.70, 0.72], upper=1.0)
        self.assertTrue(any("n<3" in w for w in c.warnings))

    def test_mixed_units_warn(self):
        c = pairwise_compare([0.60, 0.61, 0.62], [66.0, 67.0, 68.0], upper=100.0)
        self.assertTrue(any("小数与百分数" in w for w in c.warnings))

    def test_reference_worse_warns(self):
        c = pairwise_compare([0.70, 0.71, 0.72], [0.60, 0.61, 0.62], upper=1.0)
        self.assertFalse(c.passed_sigma)
        self.assertTrue(any("未优于" in w for w in c.warnings))

    def test_accept_range_violation_warns(self):
        c = pairwise_compare(
            [0.50, 0.51, 0.52], [0.95, 0.96, 0.97], upper=1.0, accept_range=(0.15, 0.8)
        )
        self.assertFalse(c.passed_score_range)
        self.assertTrue(any("超出预声明区间" in w for w in c.warnings))

    def test_to_dict_is_json_safe(self):
        import json

        c = pairwise_compare([0.6, 0.61, 0.62], [0.66, 0.67, 0.68], upper=1.0)
        json.dumps(c.to_dict())


if __name__ == "__main__":
    unittest.main()
