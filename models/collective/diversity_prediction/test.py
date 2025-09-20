#!/usr/bin/env python3
"""
多樣性預測定理測試套件
Diversity Prediction Theorem - Test Suite

全面的單元測試和集成測試

Author: Claude Code Assistant
License: MIT
"""

import unittest
import math
import sys
import os

# 添加當前目錄到路徑，以便導入模塊
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diversity_simple import DiversityPredictionSimple

try:
    from diversity_prediction import DiversityPredictionTheorem
    FULL_VERSION_AVAILABLE = True
except ImportError:
    FULL_VERSION_AVAILABLE = False


class TestDiversityPredictionSimple(unittest.TestCase):
    """測試 DiversityPredictionSimple 類的功能"""

    def setUp(self):
        """設置測試環境"""
        self.theorem = DiversityPredictionSimple()

    def test_collective_error_basic(self):
        """測試集體誤差計算"""
        predictions = [2, 8]
        true_value = 4
        expected = 1.0  # (5-4)² = 1

        result = self.theorem.collective_error(predictions, true_value)
        self.assertAlmostEqual(result, expected, places=6)

    def test_collective_error_perfect_prediction(self):
        """測試完美預測的集體誤差"""
        predictions = [10, 10, 10]
        true_value = 10
        expected = 0.0

        result = self.theorem.collective_error(predictions, true_value)
        self.assertAlmostEqual(result, expected, places=6)

    def test_average_individual_error_basic(self):
        """測試平均個體誤差計算"""
        predictions = [2, 8]
        true_value = 4
        # 個體誤差: (2-4)² = 4, (8-4)² = 16
        # 平均: (4+16)/2 = 10
        expected = 10.0

        result = self.theorem.average_individual_error(predictions, true_value)
        self.assertAlmostEqual(result, expected, places=6)

    def test_prediction_diversity_basic(self):
        """測試預測多樣性計算"""
        predictions = [2, 8]
        # 平均預測: 5
        # 多樣性: ((2-5)² + (8-5)²)/2 = (9+9)/2 = 9
        expected = 9.0

        result = self.theorem.prediction_diversity(predictions)
        self.assertAlmostEqual(result, expected, places=6)

    def test_prediction_diversity_no_diversity(self):
        """測試無多樣性情況"""
        predictions = [5, 5, 5]
        expected = 0.0

        result = self.theorem.prediction_diversity(predictions)
        self.assertAlmostEqual(result, expected, places=6)

    def test_verify_theorem_basic(self):
        """測試基本定理驗證"""
        predictions = [2, 8]
        true_value = 4

        result = self.theorem.verify_theorem(predictions, true_value)

        # 檢查定理恆等式
        self.assertTrue(result['theorem_holds'])
        self.assertAlmostEqual(result['difference'], 0.0, places=10)

        # 檢查各項計算
        self.assertAlmostEqual(result['collective_error'], 1.0, places=6)
        self.assertAlmostEqual(result['average_individual_error'], 10.0, places=6)
        self.assertAlmostEqual(result['prediction_diversity'], 9.0, places=6)

    def test_verify_theorem_multiple_models(self):
        """測試多模型定理驗證"""
        predictions = [10, 15, 20, 25]
        true_value = 18

        result = self.theorem.verify_theorem(predictions, true_value)

        # 定理應該始終成立
        self.assertTrue(result['theorem_holds'])
        self.assertLess(result['difference'], 1e-10)

        # 檢查基本屬性
        self.assertGreaterEqual(result['prediction_diversity'], 0)
        self.assertGreaterEqual(result['average_individual_error'], 0)
        self.assertGreaterEqual(result['collective_error'], 0)

    def test_empty_predictions_error(self):
        """測試空預測列表的錯誤處理"""
        with self.assertRaises(ValueError):
            self.theorem.collective_error([], 5)

        with self.assertRaises(ValueError):
            self.theorem.average_individual_error([], 5)

        with self.assertRaises(ValueError):
            self.theorem.prediction_diversity([])

    def test_single_prediction(self):
        """測試單個預測的情況"""
        predictions = [7]
        true_value = 5

        collective_error = self.theorem.collective_error(predictions, true_value)
        avg_individual_error = self.theorem.average_individual_error(predictions, true_value)
        diversity = self.theorem.prediction_diversity(predictions)

        # 單個預測時，集體誤差等於個體誤差
        self.assertAlmostEqual(collective_error, avg_individual_error, places=6)
        # 單個預測時，多樣性為0
        self.assertAlmostEqual(diversity, 0.0, places=6)

    def test_analyze_diversity_impact(self):
        """測試多樣性影響分析"""
        predictions = [10, 20, 30]
        true_value = 25

        analysis = self.theorem.analyze_diversity_impact(predictions, true_value)

        # 檢查返回結構
        self.assertIn('basic_metrics', analysis)
        self.assertIn('diversity_benefit', analysis)
        self.assertIn('improvement_ratio', analysis)
        self.assertIn('quality_assessment', analysis)

        # 檢查改善比例合理性
        self.assertGreaterEqual(analysis['improvement_ratio'], 0)
        self.assertLessEqual(analysis['improvement_ratio'], 1)

    def test_compare_scenarios(self):
        """測試場景比較功能"""
        scenarios = [
            ([10, 10, 10], 10, "無多樣性"),
            ([5, 10, 15], 10, "中等多樣性"),
            ([0, 10, 20], 10, "高多樣性")
        ]

        comparison = self.theorem.compare_scenarios(scenarios)

        # 檢查返回結構
        self.assertIn('scenarios', comparison)
        self.assertIn('best_scenario', comparison)
        self.assertIn('best_error', comparison)

        # 檢查場景數量
        self.assertEqual(len(comparison['scenarios']), 3)

        # 最佳場景應該是誤差最小的
        best_name = comparison['best_scenario']
        best_error = comparison['best_error']
        self.assertEqual(best_error, comparison['scenarios'][best_name]['collective_error'])

    def test_generate_diversity_examples(self):
        """測試範例生成功能"""
        examples = self.theorem.generate_diversity_examples()

        # 檢查範例數量和結構
        self.assertGreater(len(examples), 0)

        for example in examples:
            self.assertIn('name', example)
            self.assertIn('predictions', example)
            self.assertIn('true_value', example)
            self.assertIn('results', example)

            # 驗證每個範例的定理
            self.assertTrue(example['results']['theorem_holds'])

    def test_simulate_bias_effect(self):
        """測試偏差效應模擬"""
        base_predictions = [8, 10, 12]
        true_value = 10
        bias = 3

        bias_analysis = self.theorem.simulate_bias_effect(base_predictions, true_value, bias)

        # 檢查返回結構
        self.assertIn('unbiased_results', bias_analysis)
        self.assertIn('biased_results', bias_analysis)
        self.assertIn('bias_impact', bias_analysis)

        # 檢查偏差不改變多樣性
        self.assertTrue(bias_analysis['bias_impact']['diversity_unchanged'])

        # 檢查偏差對集體誤差的影響
        bias_impact = bias_analysis['bias_impact']['collective_error_change']
        self.assertGreater(bias_impact, 0)  # 偏差應該增加誤差

    def test_mathematical_properties(self):
        """測試數學性質"""
        # 測試不同的預測組合
        test_cases = [
            ([1, 2, 3], 2),
            ([0, 5, 10], 5),
            ([-5, 0, 5], 0),
            ([100, 200, 300], 200),
            ([0.1, 0.2, 0.3], 0.2)
        ]

        for predictions, true_value in test_cases:
            with self.subTest(predictions=predictions, true_value=true_value):
                result = self.theorem.verify_theorem(predictions, true_value)

                # 定理應該始終成立
                self.assertTrue(result['theorem_holds'])

                # 多樣性應該非負
                self.assertGreaterEqual(result['prediction_diversity'], 0)

                # 誤差應該非負
                self.assertGreaterEqual(result['collective_error'], 0)
                self.assertGreaterEqual(result['average_individual_error'], 0)


class TestDiversityPredictionAdvanced(unittest.TestCase):
    """測試進階功能（如果可用）"""

    def setUp(self):
        """設置測試環境"""
        if FULL_VERSION_AVAILABLE:
            self.theorem = DiversityPredictionTheorem(enable_visualization=False)
        else:
            self.skipTest("完整版本不可用")

    def test_ensemble_analysis(self):
        """測試集成分析功能"""
        if not FULL_VERSION_AVAILABLE:
            self.skipTest("完整版本不可用")

        models_data = [
            {'name': '模型A', 'predictions': [10, 20, 30]},
            {'name': '模型B', 'predictions': [12, 18, 32]},
            {'name': '模型C', 'predictions': [8, 22, 28]}
        ]
        true_values = [10, 20, 30]

        analysis = self.theorem.ensemble_analysis(models_data, true_values)

        # 檢查返回結構
        self.assertIn('ensemble_metrics', analysis)
        self.assertIn('sample_analyses', analysis)
        self.assertIn('model_performances', analysis)

        # 檢查樣本分析數量
        self.assertEqual(len(analysis['sample_analyses']), len(true_values))

        # 檢查模型表現分析
        self.assertEqual(len(analysis['model_performances']), len(models_data))

    def test_diversity_vs_accuracy_study(self):
        """測試多樣性與準確性關係研究"""
        if not FULL_VERSION_AVAILABLE:
            self.skipTest("完整版本不可用")

        study_results = self.theorem.diversity_vs_accuracy_study(
            min_models=2, max_models=4, diversity_levels=[0.2, 0.5, 0.8]
        )

        # 檢查返回結構
        self.assertIn('study_results', study_results)
        self.assertIn('diversity_levels', study_results)
        self.assertIn('model_range', study_results)

        # 檢查結果數量
        expected_count = (4-2+1) * 3  # 3 models * 3 diversity levels
        self.assertEqual(len(study_results['study_results']), expected_count)

    def test_generate_comprehensive_report(self):
        """測試綜合報告生成"""
        if not FULL_VERSION_AVAILABLE:
            self.skipTest("完整版本不可用")

        predictions = [85, 90, 95]
        true_value = 88

        report = self.theorem.generate_comprehensive_report(predictions, true_value)

        # 檢查報告包含關鍵信息
        self.assertIn("多樣性預測定理分析報告", report)
        self.assertIn("基本信息", report)
        self.assertIn("核心指標", report)
        self.assertIn("定理驗證", report)
        self.assertIn("結論", report)


class TestTheoremMathematicalProperties(unittest.TestCase):
    """測試定理的數學性質"""

    def setUp(self):
        """設置測試環境"""
        self.theorem = DiversityPredictionSimple()

    def test_theorem_identity_property(self):
        """測試定理恆等式性質"""
        # 使用多種不同的預測組合
        test_cases = [
            ([1], 1),           # 單個準確預測
            ([1], 2),           # 單個不準確預測
            ([1, 1], 1),        # 兩個相同的準確預測
            ([1, 3], 2),        # 兩個對稱預測
            ([1, 2, 3], 2),     # 三個等差預測
            ([0, 10], 3),       # 大範圍預測
            ([-5, 5], 0),       # 包含負數的預測
            ([1.1, 1.2, 1.3], 1.2),  # 小數預測
        ]

        for predictions, true_value in test_cases:
            with self.subTest(predictions=predictions, true_value=true_value):
                collective = self.theorem.collective_error(predictions, true_value)
                avg_individual = self.theorem.average_individual_error(predictions, true_value)
                diversity = self.theorem.prediction_diversity(predictions)

                # 驗證恆等式：集體誤差 = 平均個體誤差 - 預測多樣性
                left_side = collective
                right_side = avg_individual - diversity

                self.assertAlmostEqual(left_side, right_side, places=10,
                                     msg=f"定理不成立: {predictions}, {true_value}")

    def test_diversity_non_negative(self):
        """測試多樣性非負性"""
        test_predictions = [
            [1, 2, 3],
            [0, 0, 0],
            [-1, 0, 1],
            [100, 200, 300],
            [0.1, 0.1, 0.1]
        ]

        for predictions in test_predictions:
            with self.subTest(predictions=predictions):
                diversity = self.theorem.prediction_diversity(predictions)
                self.assertGreaterEqual(diversity, 0,
                                      msg=f"多樣性為負: {predictions}")

    def test_diversity_zero_iff_identical(self):
        """測試多樣性為零當且僅當所有預測相同"""
        # 多樣性為零的情況
        identical_cases = [
            [5, 5, 5],
            [0, 0, 0],
            [-3, -3, -3],
            [1.5, 1.5, 1.5]
        ]

        for predictions in identical_cases:
            with self.subTest(predictions=predictions, case="identical"):
                diversity = self.theorem.prediction_diversity(predictions)
                self.assertAlmostEqual(diversity, 0, places=10)

        # 多樣性大於零的情況
        diverse_cases = [
            [1, 2],
            [0, 1, 2],
            [-1, 0, 1],
            [1.0, 1.1]
        ]

        for predictions in diverse_cases:
            with self.subTest(predictions=predictions, case="diverse"):
                diversity = self.theorem.prediction_diversity(predictions)
                self.assertGreater(diversity, 0)

    def test_collective_error_properties(self):
        """測試集體誤差的性質"""
        true_value = 10

        # 測試1: 完美平均預測
        perfect_avg_predictions = [9, 10, 11]  # 平均=10
        collective_error = self.theorem.collective_error(perfect_avg_predictions, true_value)
        self.assertAlmostEqual(collective_error, 0, places=10)

        # 測試2: 集體誤差與個體數量無關（對於相同平均值）
        predictions_3 = [8, 10, 12]  # 平均=10
        predictions_5 = [6, 8, 10, 12, 14]  # 平均=10

        error_3 = self.theorem.collective_error(predictions_3, true_value)
        error_5 = self.theorem.collective_error(predictions_5, true_value)

        self.assertAlmostEqual(error_3, error_5, places=10)

    def test_symmetry_properties(self):
        """測試對稱性性質"""
        true_value = 10

        # 對稱預測應該產生相同的多樣性
        symmetric_cases = [
            ([8, 12], [12, 8]),  # 順序對稱
            ([5, 10, 15], [15, 10, 5]),  # 順序對稱
            ([7, 10, 13], [13, 10, 7])   # 順序對稱
        ]

        for pred1, pred2 in symmetric_cases:
            with self.subTest(pred1=pred1, pred2=pred2):
                diversity1 = self.theorem.prediction_diversity(pred1)
                diversity2 = self.theorem.prediction_diversity(pred2)
                self.assertAlmostEqual(diversity1, diversity2, places=10)

                # 集體誤差也應該相同
                error1 = self.theorem.collective_error(pred1, true_value)
                error2 = self.theorem.collective_error(pred2, true_value)
                self.assertAlmostEqual(error1, error2, places=10)

    def test_scaling_properties(self):
        """測試縮放性質"""
        base_predictions = [8, 10, 12]
        base_true_value = 10
        scale_factor = 2.5

        # 縮放預測和真實值
        scaled_predictions = [p * scale_factor for p in base_predictions]
        scaled_true_value = base_true_value * scale_factor

        # 計算原始和縮放後的結果
        base_result = self.theorem.verify_theorem(base_predictions, base_true_value)
        scaled_result = self.theorem.verify_theorem(scaled_predictions, scaled_true_value)

        # 誤差應該按比例縮放（平方關係）
        scale_factor_squared = scale_factor ** 2

        self.assertAlmostEqual(
            scaled_result['collective_error'],
            base_result['collective_error'] * scale_factor_squared,
            places=8
        )

        self.assertAlmostEqual(
            scaled_result['prediction_diversity'],
            base_result['prediction_diversity'] * scale_factor_squared,
            places=8
        )


def run_performance_tests():
    """運行性能測試"""
    import time

    print("\n" + "="*50)
    print("🚀 性能測試")
    print("="*50)

    theorem = DiversityPredictionSimple()

    # 測試大量模型的性能
    model_counts = [10, 50, 100, 500, 1000]

    for n_models in model_counts:
        # 生成測試數據
        predictions = list(range(n_models))
        true_value = n_models // 2

        # 測量執行時間
        start_time = time.time()
        result = theorem.verify_theorem(predictions, true_value)
        end_time = time.time()

        elapsed = (end_time - start_time) * 1000  # 轉換為毫秒

        print(f"{n_models:4d} 模型: {elapsed:6.2f}ms, 定理成立: {result['theorem_holds']}")

    print("\n✅ 性能測試完成")


def run_edge_case_tests():
    """運行邊緣情況測試"""
    print("\n" + "="*50)
    print("🧪 邊緣情況測試")
    print("="*50)

    theorem = DiversityPredictionSimple()

    # 極端值測試
    edge_cases = [
        ("極大值", [1e6, 1e6+1, 1e6+2], 1e6+1),
        ("極小值", [1e-6, 2e-6, 3e-6], 2e-6),
        ("零值", [0, 0, 0], 0),
        ("負值", [-10, -5, 0], -5),
        ("混合符號", [-10, 0, 10], 0),
    ]

    for name, predictions, true_value in edge_cases:
        try:
            result = theorem.verify_theorem(predictions, true_value)
            status = "✅ 通過" if result['theorem_holds'] else "❌ 失敗"
            print(f"{name:<12}: {status}, 誤差差異: {result['difference']:.2e}")
        except Exception as e:
            print(f"{name:<12}: ❌ 錯誤 - {e}")

    print("\n✅ 邊緣情況測試完成")


def main():
    """主測試函數"""
    print("🧪 多樣性預測定理測試套件")
    print("="*60)

    # 運行單元測試
    print("\n📋 運行單元測試...")

    # 創建測試套件
    test_suite = unittest.TestSuite()

    # 添加測試類
    test_classes = [
        TestDiversityPredictionSimple,
        TestDiversityPredictionAdvanced,
        TestTheoremMathematicalProperties
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # 運行測試
    runner = unittest.TextTestRunner(verbosity=2)
    test_result = runner.run(test_suite)

    # 運行額外測試
    run_performance_tests()
    run_edge_case_tests()

    # 總結
    print("\n" + "="*60)
    print("📊 測試總結")
    print("="*60)

    total_tests = test_result.testsRun
    failures = len(test_result.failures)
    errors = len(test_result.errors)
    success_rate = ((total_tests - failures - errors) / total_tests * 100) if total_tests > 0 else 0

    print(f"總測試數: {total_tests}")
    print(f"成功: {total_tests - failures - errors}")
    print(f"失敗: {failures}")
    print(f"錯誤: {errors}")
    print(f"成功率: {success_rate:.1f}%")

    if failures == 0 and errors == 0:
        print("\n🎉 所有測試通過！")
        return 0
    else:
        print("\n⚠️ 部分測試失敗，請檢查代碼")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)