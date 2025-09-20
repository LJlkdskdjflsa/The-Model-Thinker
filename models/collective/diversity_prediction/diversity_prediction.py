"""
多樣性預測定理的完整實現
Diversity Prediction Theorem - Complete Implementation with Visualization

包含視覺化功能的多樣性預測定理實現

Author: Claude Code Assistant
License: MIT
"""

import math
import statistics
from typing import List, Union, Tuple, Dict, Any, Optional
from diversity_simple import DiversityPredictionSimple

try:
    import matplotlib.pyplot as plt
    import numpy as np
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("警告: matplotlib 和 numpy 未安裝，視覺化功能不可用")
    print("請運行: pip install matplotlib numpy")


class DiversityPredictionTheorem(DiversityPredictionSimple):
    """
    多樣性預測定理的完整實現類

    繼承自 DiversityPredictionSimple，添加了視覺化和高級分析功能
    """

    def __init__(self, enable_visualization: bool = True):
        """
        初始化多樣性預測定理分析器

        Args:
            enable_visualization: 是否啟用視覺化功能
        """
        super().__init__()
        self.enable_visualization = enable_visualization and VISUALIZATION_AVAILABLE

        if not VISUALIZATION_AVAILABLE and enable_visualization:
            print("警告: 視覺化庫不可用，僅使用基本功能")

    def ensemble_analysis(self, models_data: List[Dict[str, Any]], true_values: List[float]) -> Dict[str, Any]:
        """
        集成模型分析

        Args:
            models_data: 模型數據，格式為 [{'name': str, 'predictions': List[float]}, ...]
            true_values: 對應的真實值列表

        Returns:
            dict: 詳細的集成分析結果
        """
        if len(models_data[0]['predictions']) != len(true_values):
            raise ValueError("預測數量與真實值數量不匹配")

        n_samples = len(true_values)
        n_models = len(models_data)

        # 為每個樣本計算多樣性預測定理
        sample_analyses = []
        total_collective_error = 0
        total_avg_individual_error = 0
        total_diversity = 0

        for i in range(n_samples):
            sample_predictions = [model['predictions'][i] for model in models_data]
            true_val = true_values[i]

            analysis = self.analyze_diversity_impact(sample_predictions, true_val)
            analysis['sample_index'] = i
            analysis['predictions'] = sample_predictions
            analysis['true_value'] = true_val

            sample_analyses.append(analysis)

            total_collective_error += analysis['basic_metrics']['collective_error']
            total_avg_individual_error += analysis['basic_metrics']['average_individual_error']
            total_diversity += analysis['basic_metrics']['prediction_diversity']

        # 計算整體統計
        avg_collective_error = total_collective_error / n_samples
        avg_individual_error = total_avg_individual_error / n_samples
        avg_diversity = total_diversity / n_samples

        # 模型個別表現分析
        model_performances = {}
        for j, model in enumerate(models_data):
            model_name = model['name']
            model_predictions = model['predictions']

            individual_errors = [(pred - true) ** 2 for pred, true in zip(model_predictions, true_values)]
            avg_error = sum(individual_errors) / len(individual_errors)

            model_performances[model_name] = {
                'average_error': avg_error,
                'predictions': model_predictions,
                'individual_errors': individual_errors
            }

        return {
            'ensemble_metrics': {
                'average_collective_error': avg_collective_error,
                'average_individual_error': avg_individual_error,
                'average_diversity': avg_diversity,
                'diversity_benefit': avg_diversity,
                'improvement_ratio': avg_diversity / avg_individual_error if avg_individual_error > 0 else 0
            },
            'sample_analyses': sample_analyses,
            'model_performances': model_performances,
            'n_samples': n_samples,
            'n_models': n_models
        }

    def diversity_vs_accuracy_study(self, min_models: int = 2, max_models: int = 10,
                                  base_accuracy: float = 0.8, diversity_levels: List[float] = None) -> Dict[str, Any]:
        """
        研究多樣性與準確性的關係

        Args:
            min_models: 最少模型數量
            max_models: 最多模型數量
            base_accuracy: 基礎準確性（用於生成模擬數據）
            diversity_levels: 多樣性水平列表

        Returns:
            dict: 多樣性與準確性關係的研究結果
        """
        if diversity_levels is None:
            diversity_levels = [0.1, 0.3, 0.5, 0.7, 0.9]

        study_results = []

        for n_models in range(min_models, max_models + 1):
            for diversity_level in diversity_levels:
                # 生成模擬預測數據
                true_value = 10.0  # 假設真實值為10
                predictions = self._generate_diverse_predictions(n_models, true_value, diversity_level, base_accuracy)

                analysis = self.analyze_diversity_impact(predictions, true_value)

                study_results.append({
                    'n_models': n_models,
                    'diversity_level': diversity_level,
                    'actual_diversity': analysis['basic_metrics']['prediction_diversity'],
                    'collective_error': analysis['basic_metrics']['collective_error'],
                    'improvement_ratio': analysis['improvement_ratio'],
                    'predictions': predictions
                })

        return {
            'study_results': study_results,
            'diversity_levels': diversity_levels,
            'model_range': (min_models, max_models)
        }

    def _generate_diverse_predictions(self, n_models: int, true_value: float,
                                    diversity_level: float, base_accuracy: float) -> List[float]:
        """
        生成具有指定多樣性水平的模擬預測

        Args:
            n_models: 模型數量
            true_value: 真實值
            diversity_level: 多樣性水平 (0-1)
            base_accuracy: 基礎準確性

        Returns:
            list: 生成的預測值列表
        """
        import random
        random.seed(42)  # 確保可重現性

        # 基礎誤差範圍
        base_error_range = true_value * (1 - base_accuracy)

        predictions = []
        for i in range(n_models):
            # 基於多樣性水平生成不同的預測
            error_magnitude = base_error_range * (1 + diversity_level * random.uniform(-1, 1))
            direction = 1 if random.random() > 0.5 else -1
            prediction = true_value + direction * error_magnitude

            predictions.append(max(0, prediction))  # 確保預測值非負

        return predictions

    def plot_theorem_demonstration(self, predictions: List[float], true_value: float,
                                 title: str = "多樣性預測定理演示") -> None:
        """
        繪製定理演示圖

        Args:
            predictions: 預測值列表
            true_value: 真實值
            title: 圖表標題
        """
        if not self.enable_visualization:
            print("視覺化功能不可用")
            return

        analysis = self.analyze_diversity_impact(predictions, true_value)
        result = analysis['basic_metrics']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # 1. 預測值分布
        ax1.bar(range(len(predictions)), predictions, alpha=0.7, color='skyblue', label='模型預測')
        ax1.axhline(y=true_value, color='red', linestyle='--', linewidth=2, label=f'真實值 ({true_value})')
        ax1.axhline(y=result['average_prediction'], color='green', linestyle='-', linewidth=2,
                   label=f'平均預測 ({result["average_prediction"]:.2f})')
        ax1.set_xlabel('模型編號')
        ax1.set_ylabel('預測值')
        ax1.set_title('各模型預測值分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 誤差分解
        labels = ['集體誤差', '平均個體誤差', '預測多樣性']
        values = [result['collective_error'], result['average_individual_error'], result['prediction_diversity']]
        colors = ['lightcoral', 'lightsalmon', 'lightgreen']

        bars = ax2.bar(labels, values, color=colors, alpha=0.8)
        ax2.set_ylabel('誤差值')
        ax2.set_title('誤差分解分析')
        ax2.grid(True, alpha=0.3)

        # 在柱狀圖上標註數值
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.2f}', ha='center', va='bottom')

        # 3. 定理驗證可視化
        theorem_data = [result['collective_error'], result['average_individual_error'], -result['prediction_diversity']]
        theorem_labels = ['集體誤差', '平均個體誤差', '-預測多樣性']
        colors_theorem = ['red', 'blue', 'green']

        bars_theorem = ax3.bar(theorem_labels, theorem_data, color=colors_theorem, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.set_ylabel('誤差值')
        ax3.set_title('定理恆等式驗證\n集體誤差 = 平均個體誤差 - 預測多樣性')
        ax3.grid(True, alpha=0.3)

        # 標註驗證結果
        verification_text = f"定理驗證: {result['theorem_holds']}\n差異: {result['difference']:.6f}"
        ax3.text(0.5, 0.95, verification_text, transform=ax3.transAxes,
                ha='center', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 4. 改善效果分析
        improvement_categories = ['個體模型\n平均表現', '集體模型\n表現']
        improvement_values = [result['average_individual_error'], result['collective_error']]
        improvement_colors = ['lightblue', 'darkgreen']

        bars_improvement = ax4.bar(improvement_categories, improvement_values, color=improvement_colors, alpha=0.8)
        ax4.set_ylabel('誤差值')
        ax4.set_title('集體 vs 個體表現比較')
        ax4.grid(True, alpha=0.3)

        # 標註改善程度
        improvement = result['prediction_diversity']
        improvement_percent = (improvement / result['average_individual_error'] * 100) if result['average_individual_error'] > 0 else 0
        improvement_text = f"多樣性改善: {improvement:.2f}\n改善比例: {improvement_percent:.1f}%"
        ax4.text(0.5, 0.95, improvement_text, transform=ax4.transAxes,
                ha='center', va='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        plt.tight_layout()
        plt.show()

    def plot_diversity_impact_heatmap(self, study_results: Dict[str, Any]) -> None:
        """
        繪製多樣性影響熱力圖

        Args:
            study_results: diversity_vs_accuracy_study 的結果
        """
        if not self.enable_visualization:
            print("視覺化功能不可用")
            return

        results = study_results['study_results']
        diversity_levels = study_results['diversity_levels']
        min_models, max_models = study_results['model_range']

        # 準備數據矩陣
        n_models_range = list(range(min_models, max_models + 1))
        improvement_matrix = np.zeros((len(n_models_range), len(diversity_levels)))

        for result in results:
            i = n_models_range.index(result['n_models'])
            j = diversity_levels.index(result['diversity_level'])
            improvement_matrix[i, j] = result['improvement_ratio']

        # 繪製熱力圖
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(improvement_matrix, cmap='RdYlGn', aspect='auto')

        # 設置軸標籤
        ax.set_xticks(range(len(diversity_levels)))
        ax.set_xticklabels([f'{d:.1f}' for d in diversity_levels])
        ax.set_yticks(range(len(n_models_range)))
        ax.set_yticklabels(n_models_range)

        ax.set_xlabel('多樣性水平')
        ax.set_ylabel('模型數量')
        ax.set_title('多樣性對預測改善的影響\n(顏色越綠表示改善越大)')

        # 添加數值標註
        for i in range(len(n_models_range)):
            for j in range(len(diversity_levels)):
                text = ax.text(j, i, f'{improvement_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)

        # 添加顏色條
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('改善比例', rotation=270, labelpad=15)

        plt.tight_layout()
        plt.show()

    def plot_ensemble_performance(self, ensemble_analysis: Dict[str, Any]) -> None:
        """
        繪製集成模型表現分析圖

        Args:
            ensemble_analysis: ensemble_analysis 方法的結果
        """
        if not self.enable_visualization:
            print("視覺化功能不可用")
            return

        model_performances = ensemble_analysis['model_performances']
        ensemble_metrics = ensemble_analysis['ensemble_metrics']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('集成模型表現分析', fontsize=16, fontweight='bold')

        # 1. 各模型平均誤差比較
        model_names = list(model_performances.keys())
        avg_errors = [model_performances[name]['average_error'] for name in model_names]
        ensemble_error = ensemble_metrics['average_collective_error']

        x_pos = range(len(model_names))
        bars = ax1.bar(x_pos, avg_errors, alpha=0.7, color='lightblue', label='個別模型')
        ax1.axhline(y=ensemble_error, color='red', linestyle='--', linewidth=2,
                   label=f'集成模型 ({ensemble_error:.2f})')

        ax1.set_xlabel('模型')
        ax1.set_ylabel('平均誤差')
        ax1.set_title('個別模型 vs 集成模型誤差比較')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(model_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 多樣性收益分析
        diversity_metrics = ['平均個體誤差', '集體誤差', '多樣性收益']
        diversity_values = [
            ensemble_metrics['average_individual_error'],
            ensemble_metrics['average_collective_error'],
            ensemble_metrics['average_diversity']
        ]
        colors = ['lightcoral', 'lightblue', 'lightgreen']

        bars2 = ax2.bar(diversity_metrics, diversity_values, color=colors, alpha=0.8)
        ax2.set_ylabel('誤差值')
        ax2.set_title('多樣性預測定理分解')
        ax2.grid(True, alpha=0.3)

        for bar, value in zip(bars2, diversity_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.2f}', ha='center', va='bottom')

        # 3. 樣本級別的改善分布
        sample_analyses = ensemble_analysis['sample_analyses']
        improvements = [analysis['diversity_benefit'] for analysis in sample_analyses]

        ax3.hist(improvements, bins=min(20, len(improvements)//2), alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(x=np.mean(improvements), color='red', linestyle='--', linewidth=2,
                   label=f'平均改善 ({np.mean(improvements):.2f})')
        ax3.set_xlabel('多樣性改善')
        ax3.set_ylabel('樣本數量')
        ax3.set_title('樣本級別的多樣性改善分布')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 改善比例趨勢
        sample_indices = [analysis['sample_index'] for analysis in sample_analyses]
        improvement_ratios = [analysis['improvement_ratio'] for analysis in sample_analyses]

        ax4.plot(sample_indices, improvement_ratios, 'o-', color='green', alpha=0.7, markersize=4)
        ax4.axhline(y=np.mean(improvement_ratios), color='red', linestyle='--', linewidth=2,
                   label=f'平均比例 ({np.mean(improvement_ratios):.3f})')
        ax4.set_xlabel('樣本編號')
        ax4.set_ylabel('改善比例')
        ax4.set_title('各樣本的改善比例趨勢')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def generate_comprehensive_report(self, predictions: List[float], true_value: float) -> str:
        """
        生成綜合分析報告

        Args:
            predictions: 預測值列表
            true_value: 真實值

        Returns:
            str: 格式化的分析報告
        """
        analysis = self.analyze_diversity_impact(predictions, true_value)
        result = analysis['basic_metrics']

        report = f"""
多樣性預測定理分析報告
{'='*50}

基本信息:
  預測模型數量: {len(predictions)}
  模型預測值: {predictions}
  真實值: {true_value}
  集體預測: {result['average_prediction']:.3f}

核心指標:
  集體誤差: {result['collective_error']:.6f}
  平均個體誤差: {result['average_individual_error']:.6f}
  預測多樣性: {result['prediction_diversity']:.6f}

定理驗證:
  左邊 (集體誤差): {result['theorem_left_side']:.6f}
  右邊 (平均誤差 - 多樣性): {result['theorem_right_side']:.6f}
  定理成立: {result['theorem_holds']}
  數值差異: {result['difference']:.10f}

多樣性分析:
  多樣性收益: {analysis['diversity_benefit']:.6f}
  改善比例: {analysis['improvement_ratio']:.4f} ({analysis['error_reduction_percent']:.2f}%)
  質量評估: {analysis['quality_assessment']}

詳細計算:
  各模型個體誤差:
"""
        for i, pred in enumerate(predictions):
            individual_error = (pred - true_value) ** 2
            report += f"    模型 {i+1}: ({pred:.2f} - {true_value})² = {individual_error:.6f}\n"

        report += f"\n  各模型多樣性貢獻:\n"
        avg_pred = result['average_prediction']
        for i, pred in enumerate(predictions):
            diversity_contrib = (pred - avg_pred) ** 2
            report += f"    模型 {i+1}: ({pred:.2f} - {avg_pred:.3f})² = {diversity_contrib:.6f}\n"

        if result['prediction_diversity'] > 0:
            report += f"\n結論:\n"
            report += f"  多樣性使集體預測比平均個體預測提升了 {analysis['diversity_benefit']:.6f} 的誤差改善。\n"
            report += f"  這相當於 {analysis['error_reduction_percent']:.2f}% 的相對改善。\n"
        else:
            report += f"\n結論:\n"
            report += f"  所有模型預測相同，沒有多樣性收益。\n"

        report += f"\n應用建議:\n"
        if analysis['improvement_ratio'] > 0.3:
            report += f"  - 高多樣性帶來顯著改善，建議保持模型多樣性\n"
        elif analysis['improvement_ratio'] > 0.1:
            report += f"  - 中等多樣性帶來適度改善，可考慮增加模型多樣性\n"
        else:
            report += f"  - 多樣性改善有限，建議關注提升個體模型質量\n"

        return report


def main():
    """演示多樣性預測定理的完整功能"""
    print("多樣性預測定理完整演示")
    print("=" * 60)

    # 創建實例
    theorem = DiversityPredictionTheorem()

    # 1. 基本演示
    print("\n1. 基本定理演示")
    print("-" * 40)
    predictions = [2, 8]
    true_value = 4

    if theorem.enable_visualization:
        theorem.plot_theorem_demonstration(predictions, true_value, "經典奧斯卡獎預測例子")

    # 生成報告
    report = theorem.generate_comprehensive_report(predictions, true_value)
    print(report)

    # 2. 集成分析演示
    print("\n2. 集成模型分析")
    print("-" * 40)

    # 模擬多個模型在多個樣本上的預測
    models_data = [
        {'name': '線性回歸', 'predictions': [10.2, 15.1, 8.9, 12.3, 9.8]},
        {'name': '隨機森林', 'predictions': [9.8, 14.8, 9.2, 11.9, 10.1]},
        {'name': '神經網絡', 'predictions': [10.5, 15.3, 8.7, 12.1, 9.9]},
        {'name': 'SVM', 'predictions': [9.9, 14.9, 9.0, 12.0, 10.0]}
    ]
    true_values = [10.0, 15.0, 9.0, 12.0, 10.0]

    ensemble_analysis = theorem.ensemble_analysis(models_data, true_values)

    print(f"集成分析結果:")
    print(f"  平均集體誤差: {ensemble_analysis['ensemble_metrics']['average_collective_error']:.4f}")
    print(f"  平均個體誤差: {ensemble_analysis['ensemble_metrics']['average_individual_error']:.4f}")
    print(f"  平均多樣性收益: {ensemble_analysis['ensemble_metrics']['average_diversity']:.4f}")
    print(f"  改善比例: {ensemble_analysis['ensemble_metrics']['improvement_ratio']:.4f}")

    if theorem.enable_visualization:
        theorem.plot_ensemble_performance(ensemble_analysis)

    # 3. 多樣性與準確性關係研究
    print("\n3. 多樣性影響研究")
    print("-" * 40)

    study_results = theorem.diversity_vs_accuracy_study(min_models=2, max_models=8)

    print(f"研究了 {len(study_results['study_results'])} 種不同配置")
    print(f"模型數量範圍: {study_results['model_range'][0]} - {study_results['model_range'][1]}")
    print(f"多樣性水平: {study_results['diversity_levels']}")

    if theorem.enable_visualization:
        theorem.plot_diversity_impact_heatmap(study_results)

    # 4. 實際應用案例
    print("\n4. 實際應用案例")
    print("-" * 40)

    # 股票預測案例
    stock_predictions = [105.2, 98.7, 102.1, 99.8, 103.5]  # 五個分析師預測
    actual_price = 101.0

    stock_analysis = theorem.analyze_diversity_impact(stock_predictions, actual_price)
    print(f"股票預測案例:")
    print(f"  預測值: {stock_predictions}")
    print(f"  實際價格: {actual_price}")
    print(f"  集體預測: {stock_analysis['basic_metrics']['average_prediction']:.2f}")
    print(f"  多樣性收益: {stock_analysis['diversity_benefit']:.4f}")
    print(f"  改善比例: {stock_analysis['improvement_ratio']:.4f}")

    print("\n演示完成！")

    if not theorem.enable_visualization:
        print("\n提示: 安裝 matplotlib 和 numpy 以啟用視覺化功能:")
        print("pip install matplotlib numpy")


if __name__ == "__main__":
    main()