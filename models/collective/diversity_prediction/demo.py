#!/usr/bin/env python3
"""
多樣性預測定理互動式演示
Diversity Prediction Theorem - Interactive Demo

提供命令行界面的互動式演示程序

Author: Claude Code Assistant
License: MIT
"""

import sys
import json
from typing import List, Dict, Any
from diversity_simple import DiversityPredictionSimple

try:
    from diversity_prediction import DiversityPredictionTheorem
    FULL_VERSION_AVAILABLE = True
except ImportError:
    FULL_VERSION_AVAILABLE = False
    print("註意: 完整版功能不可用，使用簡化版")


class DiversityPredictionDemo:
    """多樣性預測定理互動式演示類"""

    def __init__(self):
        """初始化演示器"""
        if FULL_VERSION_AVAILABLE:
            self.theorem = DiversityPredictionTheorem()
            print("✓ 已載入完整版 (包含視覺化)")
        else:
            self.theorem = DiversityPredictionSimple()
            print("✓ 已載入簡化版")

        self.examples_database = self._load_examples()

    def _load_examples(self) -> Dict[str, Dict[str, Any]]:
        """載入預設範例"""
        return {
            "1": {
                "name": "經典奧斯卡獎預測",
                "description": "兩個模型預測電影獲獎數量",
                "predictions": [2, 8],
                "true_value": 4,
                "context": "一個模型保守預測2項，另一個樂觀預測8項"
            },
            "2": {
                "name": "天氣溫度預測",
                "description": "四個氣象模型預測明日最高溫",
                "predictions": [25, 28, 22, 30],
                "true_value": 26,
                "context": "不同氣象模型基於不同數據源"
            },
            "3": {
                "name": "股價變化預測",
                "description": "五位分析師預測股價變化(%)",
                "predictions": [5, -2, 8, 0, 3],
                "true_value": 3,
                "context": "混合技術分析和基本面分析的預測"
            },
            "4": {
                "name": "房價估值",
                "description": "三個估價模型預測房價(萬元)",
                "predictions": [520, 480, 540],
                "true_value": 500,
                "context": "不同的房價評估方法"
            },
            "5": {
                "name": "醫療診斷信心",
                "description": "四位醫生的診斷信心分數(1-10)",
                "predictions": [7, 4, 6, 8],
                "true_value": 6,
                "context": "不同專科醫生對病情嚴重程度的評估"
            }
        }

    def show_main_menu(self):
        """顯示主選單"""
        print("\n" + "="*60)
        print("🎯 多樣性預測定理互動式演示")
        print("="*60)
        print("請選擇功能:")
        print("1. 📊 基本定理演示")
        print("2. 📈 預設範例展示")
        print("3. ✏️  自定義預測分析")
        print("4. 🔍 多樣性影響研究")
        print("5. 📋 批量場景比較")
        if FULL_VERSION_AVAILABLE:
            print("6. 📊 視覺化分析")
            print("7. 📊 集成模型分析")
        print("8. 📖 理論說明")
        print("9. 🧪 進階實驗")
        print("0. 🚪 退出")
        print("-"*60)

    def basic_demonstration(self):
        """基本定理演示"""
        print("\n🎯 基本定理演示")
        print("="*50)

        # 經典例子
        predictions = [2, 8]
        true_value = 4

        print(f"使用經典奧斯卡獎預測例子:")
        print(f"模型預測: {predictions}")
        print(f"真實值: {true_value}")

        self.theorem.print_theorem_explanation(predictions, true_value)

        input("\n按 Enter 鍵繼續...")

    def show_examples(self):
        """顯示預設範例"""
        print("\n📈 預設範例展示")
        print("="*50)

        print("可用範例:")
        for key, example in self.examples_database.items():
            print(f"{key}. {example['name']} - {example['description']}")

        print("\n請選擇範例編號 (1-5) 或按 Enter 顯示全部:")
        choice = input("選擇: ").strip()

        if choice in self.examples_database:
            self._analyze_single_example(choice)
        else:
            self._analyze_all_examples()

    def _analyze_single_example(self, choice: str):
        """分析單個範例"""
        example = self.examples_database[choice]

        print(f"\n📊 {example['name']}")
        print("-"*40)
        print(f"場景: {example['context']}")

        self.theorem.print_theorem_explanation(
            example['predictions'],
            example['true_value']
        )

        if FULL_VERSION_AVAILABLE:
            print("\n是否顯示視覺化圖表? (y/N)")
            if input().lower() == 'y':
                self.theorem.plot_theorem_demonstration(
                    example['predictions'],
                    example['true_value'],
                    example['name']
                )

    def _analyze_all_examples(self):
        """分析所有範例"""
        print("\n📊 所有範例分析結果")
        print("="*60)

        results = []
        for key, example in self.examples_database.items():
            analysis = self.theorem.analyze_diversity_impact(
                example['predictions'],
                example['true_value']
            )
            results.append({
                'name': example['name'],
                'analysis': analysis
            })

        # 顯示比較表格
        print(f"{'範例名稱':<20} {'集體誤差':<10} {'多樣性':<10} {'改善%':<10}")
        print("-"*60)

        for result in results:
            name = result['name'][:18]
            metrics = result['analysis']['basic_metrics']
            collective_error = metrics['collective_error']
            diversity = metrics['prediction_diversity']
            improvement = result['analysis']['error_reduction_percent']

            print(f"{name:<20} {collective_error:<10.3f} {diversity:<10.3f} {improvement:<10.1f}")

        input("\n按 Enter 鍵繼續...")

    def custom_analysis(self):
        """自定義預測分析"""
        print("\n✏️ 自定義預測分析")
        print("="*50)

        try:
            # 輸入預測值
            print("請輸入模型預測值 (用空格分隔):")
            predictions_input = input("預測值: ").strip()
            predictions = [float(x) for x in predictions_input.split()]

            if len(predictions) < 2:
                print("❌ 至少需要2個預測值")
                return

            # 輸入真實值
            true_value = float(input("真實值: "))

            # 分析
            print(f"\n📊 分析結果:")
            self.theorem.print_theorem_explanation(predictions, true_value)

            # 生成報告
            if FULL_VERSION_AVAILABLE:
                print("\n是否生成詳細報告? (y/N)")
                if input().lower() == 'y':
                    report = self.theorem.generate_comprehensive_report(predictions, true_value)
                    print(report)

                print("\n是否顯示視覺化圖表? (y/N)")
                if input().lower() == 'y':
                    self.theorem.plot_theorem_demonstration(
                        predictions, true_value, "自定義分析"
                    )

        except ValueError:
            print("❌ 輸入格式錯誤，請輸入數字")
        except Exception as e:
            print(f"❌ 發生錯誤: {e}")

        input("\n按 Enter 鍵繼續...")

    def diversity_impact_study(self):
        """多樣性影響研究"""
        print("\n🔍 多樣性影響研究")
        print("="*50)

        if not FULL_VERSION_AVAILABLE:
            print("❌ 此功能需要完整版本，請安裝 matplotlib 和 numpy")
            input("按 Enter 鍵繼續...")
            return

        print("正在生成多樣性影響研究...")

        # 進行研究
        study_results = self.theorem.diversity_vs_accuracy_study(
            min_models=2, max_models=8
        )

        print(f"✓ 已分析 {len(study_results['study_results'])} 種配置")

        # 顯示摘要
        results = study_results['study_results']
        max_improvement = max(r['improvement_ratio'] for r in results)
        best_config = next(r for r in results if r['improvement_ratio'] == max_improvement)

        print(f"\n📈 研究摘要:")
        print(f"最大改善比例: {max_improvement:.4f}")
        print(f"最佳配置: {best_config['n_models']}個模型，多樣性水平{best_config['diversity_level']}")

        print("\n是否顯示熱力圖? (y/N)")
        if input().lower() == 'y':
            self.theorem.plot_diversity_impact_heatmap(study_results)

        input("\n按 Enter 鍵繼續...")

    def batch_scenario_comparison(self):
        """批量場景比較"""
        print("\n📋 批量場景比較")
        print("="*50)

        scenarios = [
            ([30, 30, 30], 30, "無多樣性"),
            ([29, 30, 31], 30, "低多樣性"),
            ([25, 30, 35], 30, "中多樣性"),
            ([10, 30, 50], 30, "高多樣性"),
            ([20, 25, 30, 35, 40], 30, "五模型中多樣性")
        ]

        comparison = self.theorem.compare_scenarios(scenarios)

        print("場景比較結果:")
        print(f"{'場景名稱':<15} {'集體誤差':<10} {'多樣性':<10} {'評估':<15}")
        print("-"*60)

        for name, data in comparison['scenarios'].items():
            print(f"{name:<15} {data['collective_error']:<10.2f} {data['diversity']:<10.2f} {data['assessment']:<15}")

        print(f"\n🏆 最佳場景: {comparison['best_scenario']}")
        print(f"最小誤差: {comparison['best_error']:.2f}")

        input("\n按 Enter 鍵繼續...")

    def visualization_analysis(self):
        """視覺化分析"""
        if not FULL_VERSION_AVAILABLE:
            print("❌ 視覺化功能不可用")
            input("按 Enter 鍵繼續...")
            return

        print("\n📊 視覺化分析")
        print("="*50)

        print("選擇視覺化類型:")
        print("1. 基本定理演示圖")
        print("2. 多樣性影響熱力圖")
        print("3. 自定義數據視覺化")

        choice = input("選擇 (1-3): ").strip()

        if choice == "1":
            # 使用經典例子
            self.theorem.plot_theorem_demonstration([2, 8], 4, "經典奧斯卡獎預測")

        elif choice == "2":
            study_results = self.theorem.diversity_vs_accuracy_study()
            self.theorem.plot_diversity_impact_heatmap(study_results)

        elif choice == "3":
            self.custom_analysis()

        input("\n按 Enter 鍵繼續...")

    def ensemble_analysis(self):
        """集成模型分析"""
        if not FULL_VERSION_AVAILABLE:
            print("❌ 集成分析功能不可用")
            input("按 Enter 鍵繼續...")
            return

        print("\n📊 集成模型分析")
        print("="*50)

        # 使用預設的集成數據
        models_data = [
            {'name': '線性回歸', 'predictions': [10.2, 15.1, 8.9, 12.3, 9.8]},
            {'name': '隨機森林', 'predictions': [9.8, 14.8, 9.2, 11.9, 10.1]},
            {'name': '神經網絡', 'predictions': [10.5, 15.3, 8.7, 12.1, 9.9]},
            {'name': 'SVM', 'predictions': [9.9, 14.9, 9.0, 12.0, 10.0]}
        ]
        true_values = [10.0, 15.0, 9.0, 12.0, 10.0]

        print("正在分析集成模型...")
        analysis = self.theorem.ensemble_analysis(models_data, true_values)

        metrics = analysis['ensemble_metrics']
        print(f"\n📈 集成分析結果:")
        print(f"平均集體誤差: {metrics['average_collective_error']:.4f}")
        print(f"平均個體誤差: {metrics['average_individual_error']:.4f}")
        print(f"平均多樣性收益: {metrics['average_diversity']:.4f}")
        print(f"改善比例: {metrics['improvement_ratio']:.4f}")

        print("\n是否顯示詳細視覺化分析? (y/N)")
        if input().lower() == 'y':
            self.theorem.plot_ensemble_performance(analysis)

        input("\n按 Enter 鍵繼續...")

    def theory_explanation(self):
        """理論說明"""
        print("\n📖 多樣性預測定理理論說明")
        print("="*60)

        explanation = """
📚 定理內容:
   多模型誤差 = 平均模型誤差 - 模型預測的多樣性

🧮 數學表達:
   集體誤差 = (M̄ - V)²
   平均個體誤差 = (1/N) × Σ(Mᵢ - V)²
   預測多樣性 = (1/N) × Σ(Mᵢ - M̄)²

   其中：
   - Mᵢ: 模型 i 的預測值
   - M̄: 所有模型預測的平均值
   - V: 真實值
   - N: 模型數量

🔑 核心洞察:
   1. 這是一個數學恆等式，總是成立
   2. 多樣性越大，集體誤差相對於個體平均誤差的改善越大
   3. 相反類型的誤差(正負)會相互抵消

⚠️ 重要限制:
   1. 無法消除所有模型共有的系統性偏差
   2. 需要模型間保持相對獨立性
   3. 多樣性必須是有意義的，而非隨機噪音

🎯 實際應用:
   - 機器學習中的集成方法
   - 金融預測模型組合
   - 醫療診斷的多專家會診
   - 氣象預報的多模型集成
   - 民意調查的多機構平均

💡 優化策略:
   - 使用不同的算法或方法
   - 基於不同的特徵或數據源
   - 採用不同的模型假設
   - 在不同的子樣本上訓練
        """

        print(explanation)
        input("\n按 Enter 鍵繼續...")

    def advanced_experiments(self):
        """進階實驗"""
        print("\n🧪 進階實驗")
        print("="*50)

        print("選擇實驗類型:")
        print("1. 🔬 偏差影響實驗")
        print("2. 📊 模型數量影響")
        print("3. 🎲 隨機性與多樣性")
        print("4. ⚖️ 加權集成 vs 等權集成")

        choice = input("選擇實驗 (1-4): ").strip()

        if choice == "1":
            self._bias_experiment()
        elif choice == "2":
            self._model_count_experiment()
        elif choice == "3":
            self._randomness_experiment()
        elif choice == "4":
            self._weighted_ensemble_experiment()
        else:
            print("無效選擇")

        input("\n按 Enter 鍵繼續...")

    def _bias_experiment(self):
        """偏差影響實驗"""
        print("\n🔬 偏差影響實驗")
        print("-"*30)

        base_predictions = [25, 30, 35]
        true_value = 30
        biases = [0, 2, 5, 10]

        print("實驗: 觀察共同偏差對定理的影響")
        print(f"基礎預測: {base_predictions}")
        print(f"真實值: {true_value}")

        print(f"\n{'偏差':<6} {'集體誤差':<10} {'多樣性':<10} {'多樣性變化':<12}")
        print("-"*45)

        base_diversity = self.theorem.prediction_diversity(base_predictions)

        for bias in biases:
            bias_analysis = self.theorem.simulate_bias_effect(base_predictions, true_value, bias)
            biased_error = bias_analysis['biased_results']['collective_error']
            biased_diversity = bias_analysis['biased_results']['prediction_diversity']
            diversity_change = "unchanged" if abs(biased_diversity - base_diversity) < 1e-10 else "changed"

            print(f"{bias:<6} {biased_error:<10.2f} {biased_diversity:<10.2f} {diversity_change:<12}")

        print("\n💡 結論: 共同偏差不影響多樣性，但會增加集體誤差")

    def _model_count_experiment(self):
        """模型數量影響實驗"""
        print("\n📊 模型數量影響實驗")
        print("-"*30)

        true_value = 100
        base_models = [95, 105]  # 兩個基礎模型

        print("實驗: 增加更多模型對多樣性的影響")
        print(f"真實值: {true_value}")

        for n in range(2, 8):
            # 生成 n 個模型的預測
            predictions = base_models[:2]  # 保持前兩個
            for i in range(2, n):
                # 添加新的預測值，保持一定多樣性
                new_pred = true_value + (i-1) * 10 - 20
                predictions.append(new_pred)

            analysis = self.theorem.analyze_diversity_impact(predictions, true_value)
            metrics = analysis['basic_metrics']

            print(f"{n}模型: 集體誤差={metrics['collective_error']:.2f}, "
                  f"多樣性={metrics['prediction_diversity']:.2f}, "
                  f"改善={analysis['error_reduction_percent']:.1f}%")

    def _randomness_experiment(self):
        """隨機性與多樣性實驗"""
        print("\n🎲 隨機性與多樣性實驗")
        print("-"*30)

        import random
        random.seed(42)

        true_value = 50
        base_prediction = 50

        print("實驗: 比較有意義的多樣性 vs 純隨機噪音")

        # 有意義的多樣性 (系統性的不同方法)
        meaningful_predictions = [45, 50, 55]  # 保守、中性、樂觀
        meaningful_analysis = self.theorem.analyze_diversity_impact(meaningful_predictions, true_value)

        # 隨機噪音
        random_predictions = [base_prediction + random.uniform(-10, 10) for _ in range(3)]
        random_analysis = self.theorem.analyze_diversity_impact(random_predictions, true_value)

        print(f"\n有意義多樣性: {meaningful_predictions}")
        print(f"集體誤差: {meaningful_analysis['basic_metrics']['collective_error']:.2f}")
        print(f"多樣性: {meaningful_analysis['basic_metrics']['prediction_diversity']:.2f}")

        print(f"\n隨機噪音: {[f'{p:.1f}' for p in random_predictions]}")
        print(f"集體誤差: {random_analysis['basic_metrics']['collective_error']:.2f}")
        print(f"多樣性: {random_analysis['basic_metrics']['prediction_diversity']:.2f}")

    def _weighted_ensemble_experiment(self):
        """加權集成實驗"""
        print("\n⚖️ 加權集成 vs 等權集成實驗")
        print("-"*30)

        predictions = [90, 100, 110]  # 三個模型預測
        true_value = 95
        weights_scenarios = [
            ([1/3, 1/3, 1/3], "等權重"),
            ([0.5, 0.3, 0.2], "偏向模型1"),
            ([0.2, 0.6, 0.2], "偏向模型2"),
            ([0.1, 0.1, 0.8], "偏向模型3")
        ]

        print(f"模型預測: {predictions}")
        print(f"真實值: {true_value}")
        print(f"\n{'權重策略':<15} {'加權預測':<10} {'誤差':<8}")
        print("-"*35)

        for weights, name in weights_scenarios:
            weighted_pred = sum(p * w for p, w in zip(predictions, weights))
            weighted_error = (weighted_pred - true_value) ** 2

            print(f"{name:<15} {weighted_pred:<10.1f} {weighted_error:<8.1f}")

        print("\n💡 註: 等權重集成等同於多樣性預測定理的情況")

    def run(self):
        """運行演示程序"""
        print("🎉 歡迎使用多樣性預測定理互動式演示！")

        while True:
            try:
                self.show_main_menu()
                choice = input("請選擇功能 (0-9): ").strip()

                if choice == "0":
                    print("\n👋 感謝使用！再見！")
                    break
                elif choice == "1":
                    self.basic_demonstration()
                elif choice == "2":
                    self.show_examples()
                elif choice == "3":
                    self.custom_analysis()
                elif choice == "4":
                    self.diversity_impact_study()
                elif choice == "5":
                    self.batch_scenario_comparison()
                elif choice == "6":
                    self.visualization_analysis()
                elif choice == "7":
                    self.ensemble_analysis()
                elif choice == "8":
                    self.theory_explanation()
                elif choice == "9":
                    self.advanced_experiments()
                else:
                    print("❌ 無效選擇，請重新輸入")

            except KeyboardInterrupt:
                print("\n\n👋 程序已中斷，再見！")
                break
            except Exception as e:
                print(f"\n❌ 發生錯誤: {e}")
                print("請重試或選擇其他功能")


def main():
    """主程序入口"""
    try:
        demo = DiversityPredictionDemo()
        demo.run()
    except Exception as e:
        print(f"❌ 程序啟動失敗: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()