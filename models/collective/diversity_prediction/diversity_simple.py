"""
多樣性預測定理的簡單實現
Diversity Prediction Theorem - Simple Implementation

不需要外部依賴，純Python實現多樣性預測定理的核心計算

Author: Claude Code Assistant
License: MIT
"""

import math
from typing import List, Union, Tuple, Dict, Any


class DiversityPredictionSimple:
    """
    多樣性預測定理的簡單實現類

    實現多樣性預測定理的核心計算：
    集體誤差 = 平均個體誤差 - 預測多樣性
    """

    def __init__(self):
        """初始化多樣性預測定理計算器"""
        self.calculations_history = []

    def collective_error(self, predictions: List[float], true_value: float) -> float:
        """
        計算集體誤差（多模型誤差）

        Args:
            predictions: 各模型的預測值列表
            true_value: 真實值

        Returns:
            float: 集體誤差 = (平均預測 - 真實值)²
        """
        if not predictions:
            raise ValueError("預測值列表不能為空")

        average_prediction = sum(predictions) / len(predictions)
        return (average_prediction - true_value) ** 2

    def average_individual_error(self, predictions: List[float], true_value: float) -> float:
        """
        計算平均個體誤差

        Args:
            predictions: 各模型的預測值列表
            true_value: 真實值

        Returns:
            float: 平均個體誤差 = (1/n) × Σ(預測值 - 真實值)²
        """
        if not predictions:
            raise ValueError("預測值列表不能為空")

        individual_errors = [(pred - true_value) ** 2 for pred in predictions]
        return sum(individual_errors) / len(individual_errors)

    def prediction_diversity(self, predictions: List[float]) -> float:
        """
        計算預測多樣性

        Args:
            predictions: 各模型的預測值列表

        Returns:
            float: 預測多樣性 = (1/n) × Σ(預測值 - 平均預測)²
        """
        if not predictions:
            raise ValueError("預測值列表不能為空")

        average_prediction = sum(predictions) / len(predictions)
        diversity_terms = [(pred - average_prediction) ** 2 for pred in predictions]
        return sum(diversity_terms) / len(diversity_terms)

    def verify_theorem(self, predictions: List[float], true_value: float) -> Dict[str, float]:
        """
        驗證多樣性預測定理恆等式

        Args:
            predictions: 各模型的預測值列表
            true_value: 真實值

        Returns:
            dict: 包含所有計算結果和驗證信息
        """
        collective = self.collective_error(predictions, true_value)
        avg_individual = self.average_individual_error(predictions, true_value)
        diversity = self.prediction_diversity(predictions)

        # 驗證恆等式：集體誤差 = 平均個體誤差 - 預測多樣性
        theorem_verification = avg_individual - diversity
        is_valid = abs(collective - theorem_verification) < 1e-10

        result = {
            'collective_error': collective,
            'average_individual_error': avg_individual,
            'prediction_diversity': diversity,
            'theorem_left_side': collective,
            'theorem_right_side': theorem_verification,
            'theorem_holds': is_valid,
            'difference': abs(collective - theorem_verification),
            'average_prediction': sum(predictions) / len(predictions),
            'num_models': len(predictions)
        }

        # 保存計算歷史
        calculation = {
            'predictions': predictions.copy(),
            'true_value': true_value,
            'results': result
        }
        self.calculations_history.append(calculation)

        return result

    def analyze_diversity_impact(self, predictions: List[float], true_value: float) -> Dict[str, Any]:
        """
        分析多樣性對預測準確性的影響

        Args:
            predictions: 各模型的預測值列表
            true_value: 真實值

        Returns:
            dict: 詳細的分析結果
        """
        result = self.verify_theorem(predictions, true_value)

        analysis = {
            'basic_metrics': result,
            'diversity_benefit': result['prediction_diversity'],
            'improvement_ratio': result['prediction_diversity'] / result['average_individual_error'] if result['average_individual_error'] > 0 else 0,
            'error_reduction_percent': (result['prediction_diversity'] / result['average_individual_error'] * 100) if result['average_individual_error'] > 0 else 0
        }

        # 預測質量評估
        if result['collective_error'] < result['average_individual_error']:
            analysis['quality_assessment'] = "多樣性帶來正面效果"
        elif result['collective_error'] == result['average_individual_error']:
            analysis['quality_assessment'] = "無多樣性，集體等於平均"
        else:
            analysis['quality_assessment'] = "理論上不可能的情況"

        return analysis

    def compare_scenarios(self, scenarios: List[Tuple[List[float], float, str]]) -> Dict[str, Any]:
        """
        比較多個預測場景

        Args:
            scenarios: [(預測值列表, 真實值, 場景名稱), ...]

        Returns:
            dict: 場景比較結果
        """
        comparisons = {}

        for predictions, true_value, name in scenarios:
            analysis = self.analyze_diversity_impact(predictions, true_value)
            comparisons[name] = {
                'predictions': predictions,
                'true_value': true_value,
                'collective_error': analysis['basic_metrics']['collective_error'],
                'diversity': analysis['basic_metrics']['prediction_diversity'],
                'improvement': analysis['diversity_benefit'],
                'assessment': analysis['quality_assessment']
            }

        # 找出最佳場景
        best_scenario = min(comparisons.keys(),
                          key=lambda x: comparisons[x]['collective_error'])

        return {
            'scenarios': comparisons,
            'best_scenario': best_scenario,
            'best_error': comparisons[best_scenario]['collective_error']
        }

    def generate_diversity_examples(self) -> List[Dict[str, Any]]:
        """
        生成經典的多樣性預測定理示例

        Returns:
            list: 包含多個示例的列表
        """
        examples = []

        # 經典奧斯卡獎例子
        oscar_example = {
            'name': '奧斯卡獎預測',
            'description': '兩個模型預測電影獲獎數量',
            'predictions': [2, 8],
            'true_value': 4,
            'context': '一個模型保守預測2項，另一個樂觀預測8項'
        }
        oscar_result = self.verify_theorem(oscar_example['predictions'], oscar_example['true_value'])
        oscar_example['results'] = oscar_result
        examples.append(oscar_example)

        # 天氣預測例子
        weather_example = {
            'name': '天氣溫度預測',
            'description': '四個氣象模型預測明日最高溫',
            'predictions': [25, 28, 22, 30],
            'true_value': 26,
            'context': '不同氣象模型基於不同數據源'
        }
        weather_result = self.verify_theorem(weather_example['predictions'], weather_example['true_value'])
        weather_example['results'] = weather_result
        examples.append(weather_example)

        # 股價預測例子
        stock_example = {
            'name': '股價變化預測',
            'description': '技術面與基本面分析師預測',
            'predictions': [5, -2, 8, 0, 3],
            'true_value': 3,
            'context': '混合技術分析和基本面分析的預測'
        }
        stock_result = self.verify_theorem(stock_example['predictions'], stock_example['true_value'])
        stock_example['results'] = stock_result
        examples.append(stock_example)

        return examples

    def simulate_bias_effect(self, base_predictions: List[float], true_value: float, bias: float) -> Dict[str, Any]:
        """
        模擬共同偏差對定理的影響

        Args:
            base_predictions: 基礎預測值
            true_value: 真實值
            bias: 共同偏差值

        Returns:
            dict: 偏差影響分析
        """
        # 無偏差情況
        unbiased_result = self.verify_theorem(base_predictions, true_value)

        # 有偏差情況
        biased_predictions = [pred + bias for pred in base_predictions]
        biased_result = self.verify_theorem(biased_predictions, true_value)

        return {
            'original_predictions': base_predictions,
            'biased_predictions': biased_predictions,
            'bias_amount': bias,
            'true_value': true_value,
            'unbiased_results': unbiased_result,
            'biased_results': biased_result,
            'bias_impact': {
                'collective_error_change': biased_result['collective_error'] - unbiased_result['collective_error'],
                'diversity_unchanged': abs(biased_result['prediction_diversity'] - unbiased_result['prediction_diversity']) < 1e-10,
                'avg_error_change': biased_result['average_individual_error'] - unbiased_result['average_individual_error']
            }
        }

    def print_theorem_explanation(self, predictions: List[float], true_value: float):
        """
        打印定理的詳細解釋

        Args:
            predictions: 預測值列表
            true_value: 真實值
        """
        result = self.verify_theorem(predictions, true_value)

        print("=" * 60)
        print("多樣性預測定理 (Diversity Prediction Theorem)")
        print("=" * 60)
        print()
        print(f"模型預測: {predictions}")
        print(f"真實值: {true_value}")
        print(f"平均預測: {result['average_prediction']:.2f}")
        print()
        print("計算過程:")
        print("-" * 40)
        print(f"1. 集體誤差 = (平均預測 - 真實值)²")
        print(f"   = ({result['average_prediction']:.2f} - {true_value})²")
        print(f"   = {result['collective_error']:.2f}")
        print()
        print(f"2. 平均個體誤差 = 各模型誤差的平均")
        for i, pred in enumerate(predictions):
            error = (pred - true_value) ** 2
            print(f"   模型{i+1}誤差: ({pred} - {true_value})² = {error}")
        print(f"   平均 = {result['average_individual_error']:.2f}")
        print()
        print(f"3. 預測多樣性 = 各模型與平均預測差異的平方和")
        for i, pred in enumerate(predictions):
            diversity = (pred - result['average_prediction']) ** 2
            print(f"   模型{i+1}多樣性: ({pred} - {result['average_prediction']:.2f})² = {diversity:.2f}")
        print(f"   平均 = {result['prediction_diversity']:.2f}")
        print()
        print("定理驗證:")
        print("-" * 40)
        print(f"集體誤差 = 平均個體誤差 - 預測多樣性")
        print(f"{result['collective_error']:.2f} = {result['average_individual_error']:.2f} - {result['prediction_diversity']:.2f}")
        print(f"{result['collective_error']:.2f} = {result['theorem_right_side']:.2f}")
        print(f"定理成立: {result['theorem_holds']}")
        print()
        print("結論:")
        print("-" * 40)
        if result['prediction_diversity'] > 0:
            improvement = result['prediction_diversity']
            percentage = (improvement / result['average_individual_error']) * 100
            print(f"多樣性帶來了 {improvement:.2f} 的誤差改善 ({percentage:.1f}%)")
        else:
            print("沒有預測多樣性，集體預測等於個體平均")
        print()


def main():
    """演示多樣性預測定理的主要功能"""
    print("多樣性預測定理演示")
    print("=" * 50)

    # 創建實例
    theorem = DiversityPredictionSimple()

    # 經典奧斯卡獎例子
    print("\n1. 經典奧斯卡獎預測例子")
    print("-" * 30)
    theorem.print_theorem_explanation([2, 8], 4)

    # 生成更多例子
    print("\n2. 更多應用例子")
    print("-" * 30)
    examples = theorem.generate_diversity_examples()

    for example in examples[1:]:  # 跳過第一個（已經展示過）
        print(f"\n{example['name']}:")
        print(f"場景: {example['context']}")
        result = example['results']
        print(f"預測: {example['predictions']}")
        print(f"真實值: {example['true_value']}")
        print(f"集體誤差: {result['collective_error']:.2f}")
        print(f"多樣性改善: {result['prediction_diversity']:.2f}")

    # 比較不同多樣性場景
    print("\n3. 多樣性影響比較")
    print("-" * 30)
    scenarios = [
        ([30, 30, 30], 30, "無多樣性"),
        ([29, 30, 31], 30, "低多樣性"),
        ([25, 30, 35], 30, "中多樣性"),
        ([10, 30, 50], 30, "高多樣性")
    ]

    comparison = theorem.compare_scenarios(scenarios)

    for name, data in comparison['scenarios'].items():
        print(f"{name}: 集體誤差={data['collective_error']:.2f}, 多樣性={data['diversity']:.2f}")

    print(f"\n最佳場景: {comparison['best_scenario']}")

    # 偏差影響演示
    print("\n4. 共同偏差的影響")
    print("-" * 30)
    bias_analysis = theorem.simulate_bias_effect([25, 30, 35], 30, 5)
    print(f"原始預測: {bias_analysis['original_predictions']}")
    print(f"偏差後預測: {bias_analysis['biased_predictions']}")
    print(f"集體誤差變化: {bias_analysis['bias_impact']['collective_error_change']:.2f}")
    print(f"多樣性是否改變: {'否' if bias_analysis['bias_impact']['diversity_unchanged'] else '是'}")

    print("\n演示完成！")


if __name__ == "__main__":
    main()