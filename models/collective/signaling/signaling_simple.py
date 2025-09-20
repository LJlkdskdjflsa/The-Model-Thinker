"""
信號傳遞模型的簡單實現
Signaling Models - Simple Implementation

基於斯彭斯(Spence)信號理論的數學模型

Author: Claude Code Assistant
License: MIT
"""

import math
from typing import List, Dict, Tuple, Any, Optional, Union
import itertools


class SignalingModelSimple:
    """
    信號傳遞模型的簡單實現類

    實現斯彭斯的信號理論，包括：
    - 教育信號模型
    - 分離均衡和混同均衡
    - 信號成本分析
    - 福利效應評估
    """

    def __init__(self):
        """初始化信號傳遞模型分析器"""
        self.analysis_history = []

    def signal_cost(self, signal_level: float, ability: float, cost_parameter: float = 1.0) -> float:
        """
        計算信號成本

        Args:
            signal_level: 信號水準（如教育年數）
            ability: 個體能力
            cost_parameter: 成本參數

        Returns:
            float: 信號成本
        """
        # 標準假設：成本與能力負相關
        # c(e|θ) = α * e^2 / θ
        if ability <= 0:
            raise ValueError("能力必須為正數")
        return cost_parameter * (signal_level ** 2) / ability

    def productivity_function(self, ability: float, signal_level: float = 0.0,
                            human_capital_effect: float = 0.0) -> float:
        """
        生產力函數

        Args:
            ability: 個體能力
            signal_level: 信號水準
            human_capital_effect: 人力資本效應（信號對生產力的直接影響）

        Returns:
            float: 實際生產力
        """
        # 基礎生產力 + 人力資本效應
        return ability + human_capital_effect * signal_level

    def wage_function(self, expected_ability: float, signal_level: float = 0.0,
                     human_capital_effect: float = 0.0) -> float:
        """
        工資函數（基於期望生產力）

        Args:
            expected_ability: 雇主對能力的期望
            signal_level: 觀察到的信號水準
            human_capital_effect: 人力資本效應

        Returns:
            float: 工資水準
        """
        return expected_ability + human_capital_effect * signal_level

    def individual_utility(self, wage: float, signal_cost: float) -> float:
        """
        個體效用函數

        Args:
            wage: 工資
            signal_cost: 信號成本

        Returns:
            float: 淨效用
        """
        return wage - signal_cost

    def separating_equilibrium_analysis(self, high_ability: float, low_ability: float,
                                      population_ratio: float = 0.5, cost_parameter: float = 1.0,
                                      human_capital_effect: float = 0.0) -> Dict[str, Any]:
        """
        分離均衡分析

        Args:
            high_ability: 高能力個體的能力水準
            low_ability: 低能力個體的能力水準
            population_ratio: 高能力個體在人口中的比例
            cost_parameter: 成本參數
            human_capital_effect: 人力資本效應

        Returns:
            dict: 分離均衡分析結果
        """
        # 在分離均衡中，低能力個體選擇零信號
        low_signal = 0.0
        low_wage = self.wage_function(low_ability, low_signal, human_capital_effect)
        low_cost = self.signal_cost(low_signal, low_ability, cost_parameter)
        low_utility = self.individual_utility(low_wage, low_cost)

        # 高能力個體需要發送足夠的信號來區分自己
        # 激勵相容約束：高能力個體不願意模仿低能力個體
        # 參與約束：低能力個體不願意模仿高能力個體

        # 對於高能力個體，最優信號水準通過最大化效用得到
        # 但必須滿足激勵相容約束
        # IC約束：低能力個體模仿高能力個體的效用 ≤ 選擇零信號的效用

        # 尋找最小的高信號水準，使得低能力個體不願意模仿
        high_signal_candidates = [x * 0.1 for x in range(1, 51)]  # 0.1 到 5.0
        feasible_high_signals = []

        for hs in high_signal_candidates:
            # 檢查激勵相容約束
            high_wage = self.wage_function(high_ability, hs, human_capital_effect)

            # 低能力個體模仿的效用
            low_mimic_cost = self.signal_cost(hs, low_ability, cost_parameter)
            low_mimic_utility = self.individual_utility(high_wage, low_mimic_cost)

            # 激勵相容：低能力個體不願意模仿
            if low_mimic_utility <= low_utility:
                feasible_high_signals.append(hs)

        if not feasible_high_signals:
            return {
                'equilibrium_type': 'no_separating_equilibrium',
                'reason': '無可行的分離均衡'
            }

        # 選擇最小的可行信號（最有效率的分離均衡）
        optimal_high_signal = min(feasible_high_signals)
        high_wage = self.wage_function(high_ability, optimal_high_signal, human_capital_effect)
        high_cost = self.signal_cost(optimal_high_signal, high_ability, cost_parameter)
        high_utility = self.individual_utility(high_wage, high_cost)

        # 計算社會福利
        total_productivity = (population_ratio * self.productivity_function(high_ability, optimal_high_signal, human_capital_effect) +
                            (1 - population_ratio) * self.productivity_function(low_ability, low_signal, human_capital_effect))

        total_signal_cost = (population_ratio * high_cost + (1 - population_ratio) * low_cost)
        social_welfare = total_productivity - total_signal_cost

        return {
            'equilibrium_type': 'separating',
            'high_ability_strategy': {
                'signal_level': optimal_high_signal,
                'wage': high_wage,
                'cost': high_cost,
                'utility': high_utility
            },
            'low_ability_strategy': {
                'signal_level': low_signal,
                'wage': low_wage,
                'cost': low_cost,
                'utility': low_utility
            },
            'social_welfare': social_welfare,
            'total_productivity': total_productivity,
            'total_signal_cost': total_signal_cost,
            'population_ratio': population_ratio
        }

    def pooling_equilibrium_analysis(self, high_ability: float, low_ability: float,
                                   population_ratio: float = 0.5, cost_parameter: float = 1.0,
                                   human_capital_effect: float = 0.0) -> Dict[str, Any]:
        """
        混同均衡分析

        Args:
            high_ability: 高能力個體的能力水準
            low_ability: 低能力個體的能力水準
            population_ratio: 高能力個體在人口中的比例
            cost_parameter: 成本參數
            human_capital_effect: 人力資本效應

        Returns:
            dict: 混同均衡分析結果
        """
        # 在混同均衡中，所有個體選擇相同的信號水準
        # 雇主支付期望生產力對應的工資

        average_ability = population_ratio * high_ability + (1 - population_ratio) * low_ability

        # 檢查不同的混同信號水準
        signal_candidates = [x * 0.1 for x in range(0, 31)]  # 0 到 3.0
        feasible_pooling = []

        for signal in signal_candidates:
            pooling_wage = self.wage_function(average_ability, signal, human_capital_effect)

            # 檢查兩種類型是否都願意選擇這個信號水準
            high_cost = self.signal_cost(signal, high_ability, cost_parameter)
            low_cost = self.signal_cost(signal, low_ability, cost_parameter)

            high_utility = self.individual_utility(pooling_wage, high_cost)
            low_utility = self.individual_utility(pooling_wage, low_cost)

            # 檢查是否存在有利可圖的偏離
            # 對於每種類型，檢查偏離到其他信號水準是否更有利

            high_no_deviation = True
            low_no_deviation = True

            # 簡化檢查：只檢查偏離到0和較高信號水準
            deviation_signals = [0.0, signal + 0.1, signal + 0.5]

            for dev_signal in deviation_signals:
                if dev_signal == signal:
                    continue

                # 如果偏離，雇主會怎麼更新信念？
                # 簡化假設：偏離者被認為是平均類型
                dev_wage = self.wage_function(average_ability, dev_signal, human_capital_effect)

                dev_high_cost = self.signal_cost(dev_signal, high_ability, cost_parameter)
                dev_low_cost = self.signal_cost(dev_signal, low_ability, cost_parameter)

                dev_high_utility = self.individual_utility(dev_wage, dev_high_cost)
                dev_low_utility = self.individual_utility(dev_wage, dev_low_cost)

                if dev_high_utility > high_utility:
                    high_no_deviation = False
                if dev_low_utility > low_utility:
                    low_no_deviation = False

            if high_no_deviation and low_no_deviation:
                feasible_pooling.append({
                    'signal_level': signal,
                    'wage': pooling_wage,
                    'high_utility': high_utility,
                    'low_utility': low_utility,
                    'average_utility': population_ratio * high_utility + (1 - population_ratio) * low_utility
                })

        if not feasible_pooling:
            return {
                'equilibrium_type': 'no_pooling_equilibrium',
                'reason': '無可行的混同均衡'
            }

        # 選擇社會最優的混同均衡（通常是信號水準最低的）
        optimal_pooling = min(feasible_pooling, key=lambda x: x['signal_level'])

        signal_level = optimal_pooling['signal_level']
        wage = optimal_pooling['wage']

        high_cost = self.signal_cost(signal_level, high_ability, cost_parameter)
        low_cost = self.signal_cost(signal_level, low_ability, cost_parameter)

        # 計算社會福利
        total_productivity = (population_ratio * self.productivity_function(high_ability, signal_level, human_capital_effect) +
                            (1 - population_ratio) * self.productivity_function(low_ability, signal_level, human_capital_effect))

        total_signal_cost = population_ratio * high_cost + (1 - population_ratio) * low_cost
        social_welfare = total_productivity - total_signal_cost

        return {
            'equilibrium_type': 'pooling',
            'signal_level': signal_level,
            'wage': wage,
            'high_ability_outcome': {
                'cost': high_cost,
                'utility': optimal_pooling['high_utility']
            },
            'low_ability_outcome': {
                'cost': low_cost,
                'utility': optimal_pooling['low_utility']
            },
            'social_welfare': social_welfare,
            'total_productivity': total_productivity,
            'total_signal_cost': total_signal_cost,
            'population_ratio': population_ratio
        }

    def welfare_comparison(self, high_ability: float, low_ability: float,
                         population_ratio: float = 0.5, cost_parameter: float = 1.0,
                         human_capital_effect: float = 0.0) -> Dict[str, Any]:
        """
        福利比較分析

        Args:
            high_ability: 高能力水準
            low_ability: 低能力水準
            population_ratio: 高能力個體比例
            cost_parameter: 成本參數
            human_capital_effect: 人力資本效應

        Returns:
            dict: 福利比較結果
        """
        # 分析分離均衡
        separating = self.separating_equilibrium_analysis(
            high_ability, low_ability, population_ratio, cost_parameter, human_capital_effect
        )

        # 分析混同均衡
        pooling = self.pooling_equilibrium_analysis(
            high_ability, low_ability, population_ratio, cost_parameter, human_capital_effect
        )

        # 計算第一優解（無信息不對稱的最優解）
        first_best_productivity = (population_ratio * high_ability +
                                 (1 - population_ratio) * low_ability)
        first_best_welfare = first_best_productivity  # 無信號成本

        results = {
            'first_best_welfare': first_best_welfare,
            'separating_equilibrium': separating,
            'pooling_equilibrium': pooling,
            'welfare_ranking': []
        }

        # 福利排序
        welfare_scenarios = []

        if separating['equilibrium_type'] == 'separating':
            welfare_scenarios.append(('separating', separating['social_welfare']))

        if pooling['equilibrium_type'] == 'pooling':
            welfare_scenarios.append(('pooling', pooling['social_welfare']))

        welfare_scenarios.append(('first_best', first_best_welfare))
        welfare_scenarios.sort(key=lambda x: x[1], reverse=True)

        results['welfare_ranking'] = welfare_scenarios

        # 信號的社會價值分析
        if separating['equilibrium_type'] == 'separating' and pooling['equilibrium_type'] == 'pooling':
            signaling_value = separating['social_welfare'] - pooling['social_welfare']
            results['signaling_value'] = {
                'welfare_difference': signaling_value,
                'is_signaling_beneficial': signaling_value > 0,
                'efficiency_loss_from_signaling': first_best_welfare - max(separating['social_welfare'], pooling['social_welfare'])
            }

        return results

    def human_capital_vs_signaling_analysis(self, high_ability: float, low_ability: float,
                                          human_capital_effects: List[float] = [0.0, 0.1, 0.5, 1.0]) -> Dict[str, Any]:
        """
        人力資本效應 vs 純信號效應分析

        Args:
            high_ability: 高能力水準
            low_ability: 低能力水準
            human_capital_effects: 不同的人力資本效應水準

        Returns:
            dict: 人力資本效應分析結果
        """
        results = {}

        for hc_effect in human_capital_effects:
            welfare_analysis = self.welfare_comparison(
                high_ability, low_ability, human_capital_effect=hc_effect
            )

            results[hc_effect] = {
                'human_capital_effect': hc_effect,
                'welfare_analysis': welfare_analysis,
                'signaling_interpretation': 'pure_signaling' if hc_effect == 0 else 'mixed_effect'
            }

        return {
            'human_capital_effects': human_capital_effects,
            'analyses': results,
            'interpretation': {
                0.0: '純信號效應：教育不提升生產力',
                0.1: '微弱人力資本效應',
                0.5: '中等人力資本效應',
                1.0: '強人力資本效應：教育完全提升生產力'
            }
        }

    def signal_distortion_analysis(self, high_ability: float, low_ability: float,
                                 cost_parameters: List[float] = [0.5, 1.0, 2.0, 5.0]) -> Dict[str, Any]:
        """
        信號扭曲程度分析

        Args:
            high_ability: 高能力水準
            low_ability: 低能力水準
            cost_parameters: 不同的成本參數

        Returns:
            dict: 信號扭曲分析結果
        """
        results = {}

        for cost_param in cost_parameters:
            separating = self.separating_equilibrium_analysis(
                high_ability, low_ability, cost_parameter=cost_param
            )

            if separating['equilibrium_type'] == 'separating':
                high_signal = separating['high_ability_strategy']['signal_level']
                signal_cost = separating['high_ability_strategy']['cost']
                social_welfare = separating['social_welfare']

                results[cost_param] = {
                    'cost_parameter': cost_param,
                    'high_ability_signal': high_signal,
                    'signal_cost': signal_cost,
                    'social_welfare': social_welfare,
                    'distortion_level': 'low' if cost_param > 2.0 else 'high'
                }

        return {
            'cost_parameters': cost_parameters,
            'results': results,
            'insight': '成本參數越高，信號扭曲越小（高能力個體需要更少的信號來區分自己）'
        }

    def print_signaling_analysis(self, high_ability: float = 2.0, low_ability: float = 1.0,
                               scenario_name: str = "教育信號模型"):
        """
        打印信號傳遞模型的詳細分析

        Args:
            high_ability: 高能力水準
            low_ability: 低能力水準
            scenario_name: 場景名稱
        """
        print("=" * 60)
        print(f"信號傳遞模型分析: {scenario_name}")
        print("=" * 60)
        print(f"高能力個體: {high_ability}")
        print(f"低能力個體: {low_ability}")
        print()

        # 1. 分離均衡分析
        print("1. 分離均衡分析")
        print("-" * 40)
        separating = self.separating_equilibrium_analysis(high_ability, low_ability)

        if separating['equilibrium_type'] == 'separating':
            print("分離均衡存在：")
            high_strat = separating['high_ability_strategy']
            low_strat = separating['low_ability_strategy']

            print(f"  高能力個體:")
            print(f"    信號水準: {high_strat['signal_level']:.3f}")
            print(f"    工資: {high_strat['wage']:.3f}")
            print(f"    成本: {high_strat['cost']:.3f}")
            print(f"    效用: {high_strat['utility']:.3f}")

            print(f"  低能力個體:")
            print(f"    信號水準: {low_strat['signal_level']:.3f}")
            print(f"    工資: {low_strat['wage']:.3f}")
            print(f"    成本: {low_strat['cost']:.3f}")
            print(f"    效用: {low_strat['utility']:.3f}")

            print(f"  社會福利: {separating['social_welfare']:.3f}")
        else:
            print("不存在分離均衡")
        print()

        # 2. 混同均衡分析
        print("2. 混同均衡分析")
        print("-" * 40)
        pooling = self.pooling_equilibrium_analysis(high_ability, low_ability)

        if pooling['equilibrium_type'] == 'pooling':
            print("混同均衡存在：")
            print(f"  共同信號水準: {pooling['signal_level']:.3f}")
            print(f"  共同工資: {pooling['wage']:.3f}")
            print(f"  高能力個體效用: {pooling['high_ability_outcome']['utility']:.3f}")
            print(f"  低能力個體效用: {pooling['low_ability_outcome']['utility']:.3f}")
            print(f"  社會福利: {pooling['social_welfare']:.3f}")
        else:
            print("不存在混同均衡")
        print()

        # 3. 福利比較
        print("3. 福利比較分析")
        print("-" * 40)
        welfare_comp = self.welfare_comparison(high_ability, low_ability)

        print("福利排序（從高到低）:")
        for i, (scenario, welfare) in enumerate(welfare_comp['welfare_ranking'], 1):
            scenario_names = {
                'first_best': '第一優解（無信息不對稱）',
                'separating': '分離均衡',
                'pooling': '混同均衡'
            }
            print(f"  {i}. {scenario_names[scenario]}: {welfare:.3f}")

        if 'signaling_value' in welfare_comp:
            sv = welfare_comp['signaling_value']
            print(f"\n信號的社會價值:")
            print(f"  福利差異: {sv['welfare_difference']:.3f}")
            print(f"  信號是否有益: {'是' if sv['is_signaling_beneficial'] else '否'}")
            print(f"  因信號導致的效率損失: {sv['efficiency_loss_from_signaling']:.3f}")

        print("\n分析完成！")


def main():
    """演示信號傳遞模型的主要功能"""
    print("信號傳遞模型演示")
    print("=" * 50)

    # 創建實例
    model = SignalingModelSimple()

    # 1. 基本演示
    print("\n1. 基本信號傳遞模型分析")
    print("-" * 30)
    model.print_signaling_analysis(2.0, 1.0, "標準教育信號模型")

    # 2. 人力資本效應分析
    print("\n2. 人力資本 vs 純信號效應分析")
    print("-" * 30)
    hc_analysis = model.human_capital_vs_signaling_analysis(2.0, 1.0)

    print("不同人力資本效應下的福利比較:")
    print(f"{'HC效應':<8} {'分離福利':<10} {'混同福利':<10} {'最優選擇':<12}")
    print("-" * 45)

    for hc_effect, analysis in hc_analysis['analyses'].items():
        welfare_analysis = analysis['welfare_analysis']
        separating_welfare = welfare_analysis['separating_equilibrium'].get('social_welfare', 0)
        pooling_welfare = welfare_analysis['pooling_equilibrium'].get('social_welfare', 0)

        if separating_welfare > pooling_welfare:
            optimal = "分離"
        elif pooling_welfare > separating_welfare:
            optimal = "混同"
        else:
            optimal = "相等"

        print(f"{hc_effect:<8.1f} {separating_welfare:<10.3f} {pooling_welfare:<10.3f} {optimal:<12}")

    # 3. 信號扭曲分析
    print("\n3. 信號扭曲程度分析")
    print("-" * 30)
    distortion_analysis = model.signal_distortion_analysis(2.0, 1.0)

    print("不同成本參數下的信號水準:")
    print(f"{'成本參數':<10} {'高能力信號':<12} {'信號成本':<10} {'扭曲程度':<10}")
    print("-" * 45)

    for cost_param, result in distortion_analysis['results'].items():
        print(f"{cost_param:<10.1f} {result['high_ability_signal']:<12.3f} "
              f"{result['signal_cost']:<10.3f} {result['distortion_level']:<10}")

    # 4. 不同能力差距的影響
    print("\n4. 能力差距對均衡的影響")
    print("-" * 30)
    ability_gaps = [(2.0, 1.8), (2.0, 1.5), (2.0, 1.0), (3.0, 1.0)]

    print("不同能力差距下的均衡比較:")
    print(f"{'能力差距':<10} {'分離信號':<10} {'分離福利':<10} {'混同福利':<10}")
    print("-" * 45)

    for high_ab, low_ab in ability_gaps:
        separating = model.separating_equilibrium_analysis(high_ab, low_ab)
        pooling = model.pooling_equilibrium_analysis(high_ab, low_ab)

        gap = high_ab - low_ab
        sep_signal = separating.get('high_ability_strategy', {}).get('signal_level', 0)
        sep_welfare = separating.get('social_welfare', 0)
        pool_welfare = pooling.get('social_welfare', 0)

        print(f"{gap:<10.1f} {sep_signal:<10.3f} {sep_welfare:<10.3f} {pool_welfare:<10.3f}")

    print("\n演示完成！")


if __name__ == "__main__":
    main()