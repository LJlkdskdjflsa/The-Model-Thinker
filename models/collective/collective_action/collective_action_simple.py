"""
集體行動問題模型的簡單實現
Collective Action Problem Models - Simple Implementation

基於奧爾森(Olson)理論的集體行動問題數學模型

Author: Claude Code Assistant
License: MIT
"""

import math
from typing import List, Dict, Tuple, Any, Optional, Union


class CollectiveActionSimple:
    """
    集體行動問題的簡單實現類

    實現奧爾森的集體行動理論，包括：
    - 公共物品提供問題
    - 搭便車問題
    - 群體規模效應
    - 激勵機制分析
    """

    def __init__(self):
        """初始化集體行動問題分析器"""
        self.analysis_history = []

    def individual_benefit(self, public_good_level: float, benefit_coefficient: float) -> float:
        """
        計算個人從公共物品中獲得的收益

        Args:
            public_good_level: 公共物品的提供水準
            benefit_coefficient: 個人收益係數

        Returns:
            float: 個人收益
        """
        return benefit_coefficient * public_good_level

    def individual_cost(self, contribution: float, cost_coefficient: float = 1.0) -> float:
        """
        計算個人貢獻的成本

        Args:
            contribution: 個人貢獻量
            cost_coefficient: 成本係數

        Returns:
            float: 個人成本
        """
        return cost_coefficient * contribution

    def public_good_production(self, contributions: List[float],
                             production_function: str = "linear",
                             efficiency: float = 1.0) -> float:
        """
        公共物品生產函數

        Args:
            contributions: 各個體的貢獻列表
            production_function: 生產函數類型 ("linear", "threshold", "accelerating")
            efficiency: 生產效率

        Returns:
            float: 公共物品水準
        """
        total_contribution = sum(contributions)

        if production_function == "linear":
            return efficiency * total_contribution
        elif production_function == "threshold":
            # 閾值生產函數：需要達到一定水準才能產生公共物品
            threshold = len(contributions) * 0.5  # 假設閾值為人數的一半
            if total_contribution >= threshold:
                return efficiency * total_contribution
            else:
                return 0
        elif production_function == "accelerating":
            # 加速生產函數：邊際收益遞增
            return efficiency * (total_contribution ** 1.5)
        elif production_function == "diminishing":
            # 邊際收益遞減
            return efficiency * math.sqrt(total_contribution)
        else:
            raise ValueError(f"未知的生產函數類型: {production_function}")

    def individual_utility(self, contribution: float, public_good_level: float,
                         benefit_coefficient: float, cost_coefficient: float = 1.0) -> float:
        """
        計算個人效用 (收益 - 成本)

        Args:
            contribution: 個人貢獻
            public_good_level: 公共物品水準
            benefit_coefficient: 收益係數
            cost_coefficient: 成本係數

        Returns:
            float: 個人淨效用
        """
        benefit = self.individual_benefit(public_good_level, benefit_coefficient)
        cost = self.individual_cost(contribution, cost_coefficient)
        return benefit - cost

    def nash_equilibrium_analysis(self, n_players: int, benefit_coefficients: List[float],
                                cost_coefficients: Optional[List[float]] = None,
                                production_function: str = "linear",
                                efficiency: float = 1.0) -> Dict[str, Any]:
        """
        納什均衡分析

        Args:
            n_players: 玩家數量
            benefit_coefficients: 各玩家的收益係數
            cost_coefficients: 各玩家的成本係數
            production_function: 生產函數類型
            efficiency: 生產效率

        Returns:
            dict: 納什均衡分析結果
        """
        if cost_coefficients is None:
            cost_coefficients = [1.0] * n_players

        if len(benefit_coefficients) != n_players:
            raise ValueError("收益係數數量必須等於玩家數量")
        if len(cost_coefficients) != n_players:
            raise ValueError("成本係數數量必須等於玩家數量")

        # 對於線性生產函數，計算納什均衡
        if production_function == "linear":
            # 在線性情況下，個人的最優貢獻取決於邊際收益和邊際成本
            optimal_contributions = []
            for i in range(n_players):
                # 邊際收益 = benefit_coefficient * efficiency
                # 邊際成本 = cost_coefficient
                marginal_benefit = benefit_coefficients[i] * efficiency
                marginal_cost = cost_coefficients[i]

                # 如果邊際收益 > 邊際成本，則有貢獻動機
                if marginal_benefit > marginal_cost:
                    # 簡化假設：最優貢獻 = (邊際收益 - 邊際成本) / 邊際成本
                    optimal_contribution = max(0, (marginal_benefit - marginal_cost) / marginal_cost)
                else:
                    optimal_contribution = 0.0

                optimal_contributions.append(optimal_contribution)

            # 計算總公共物品水準
            public_good_level = self.public_good_production(
                optimal_contributions, production_function, efficiency
            )

            # 計算各個體的效用
            utilities = []
            for i in range(n_players):
                utility = self.individual_utility(
                    optimal_contributions[i], public_good_level,
                    benefit_coefficients[i], cost_coefficients[i]
                )
                utilities.append(utility)

            return {
                'equilibrium_contributions': optimal_contributions,
                'public_good_level': public_good_level,
                'individual_utilities': utilities,
                'total_utility': sum(utilities),
                'total_contribution': sum(optimal_contributions),
                'production_function': production_function,
                'efficiency': efficiency
            }

        else:
            # 對於非線性生產函數，使用迭代方法尋找均衡
            return self._iterative_equilibrium_search(
                n_players, benefit_coefficients, cost_coefficients,
                production_function, efficiency
            )

    def _iterative_equilibrium_search(self, n_players: int, benefit_coefficients: List[float],
                                    cost_coefficients: List[float], production_function: str,
                                    efficiency: float, max_iterations: int = 100) -> Dict[str, Any]:
        """
        迭代搜索納什均衡（用於非線性情況）

        Args:
            n_players: 玩家數量
            benefit_coefficients: 收益係數
            cost_coefficients: 成本係數
            production_function: 生產函數類型
            efficiency: 效率
            max_iterations: 最大迭代次數

        Returns:
            dict: 均衡結果
        """
        # 初始化貢獻（隨機起點）
        contributions = [0.5] * n_players
        tolerance = 1e-6

        for iteration in range(max_iterations):
            new_contributions = []

            for i in range(n_players):
                # 計算其他人的貢獻
                others_contributions = sum(contributions[:i] + contributions[i+1:])

                # 尋找個人最優反應
                best_contribution = 0
                best_utility = float('-inf')

                # 在合理範圍內搜索最優貢獻
                for contrib in [x * 0.1 for x in range(0, 21)]:  # 0 到 2.0，步長 0.1
                    total_contrib = others_contributions + contrib
                    public_good = self.public_good_production(
                        [total_contrib], production_function, efficiency
                    )

                    utility = self.individual_utility(
                        contrib, public_good,
                        benefit_coefficients[i], cost_coefficients[i]
                    )

                    if utility > best_utility:
                        best_utility = utility
                        best_contribution = contrib

                new_contributions.append(best_contribution)

            # 檢查收斂
            max_change = max(abs(new_contributions[i] - contributions[i])
                           for i in range(n_players))

            if max_change < tolerance:
                break

            contributions = new_contributions

        # 計算最終結果
        public_good_level = self.public_good_production(
            contributions, production_function, efficiency
        )

        utilities = []
        for i in range(n_players):
            utility = self.individual_utility(
                contributions[i], public_good_level,
                benefit_coefficients[i], cost_coefficients[i]
            )
            utilities.append(utility)

        return {
            'equilibrium_contributions': contributions,
            'public_good_level': public_good_level,
            'individual_utilities': utilities,
            'total_utility': sum(utilities),
            'total_contribution': sum(contributions),
            'production_function': production_function,
            'efficiency': efficiency,
            'iterations': iteration + 1,
            'converged': iteration < max_iterations - 1
        }

    def free_rider_analysis(self, n_players: int, benefit_coefficient: float = 1.0,
                          cost_coefficient: float = 1.0, efficiency: float = 1.0) -> Dict[str, Any]:
        """
        搭便車問題分析

        Args:
            n_players: 玩家數量
            benefit_coefficient: 收益係數
            cost_coefficient: 成本係數
            efficiency: 效率

        Returns:
            dict: 搭便車分析結果
        """
        # 情況1：所有人都貢獻
        all_contribute_contributions = [1.0] * n_players
        all_contribute_public_good = self.public_good_production(
            all_contribute_contributions, "linear", efficiency
        )

        all_contribute_utilities = []
        for i in range(n_players):
            utility = self.individual_utility(
                1.0, all_contribute_public_good, benefit_coefficient, cost_coefficient
            )
            all_contribute_utilities.append(utility)

        # 情況2：只有一個人貢獻，其他人搭便車
        one_contribute_contributions = [1.0] + [0.0] * (n_players - 1)
        one_contribute_public_good = self.public_good_production(
            one_contribute_contributions, "linear", efficiency
        )

        contributor_utility = self.individual_utility(
            1.0, one_contribute_public_good, benefit_coefficient, cost_coefficient
        )
        free_rider_utility = self.individual_utility(
            0.0, one_contribute_public_good, benefit_coefficient, cost_coefficient
        )

        # 情況3：沒有人貢獻
        no_contribute_public_good = 0.0
        no_contribute_utility = 0.0

        # 計算搭便車的誘因
        free_rider_incentive = free_rider_utility - all_contribute_utilities[0]

        return {
            'scenarios': {
                'all_contribute': {
                    'contributions': all_contribute_contributions,
                    'public_good_level': all_contribute_public_good,
                    'individual_utilities': all_contribute_utilities,
                    'total_utility': sum(all_contribute_utilities)
                },
                'one_contribute': {
                    'contributions': one_contribute_contributions,
                    'public_good_level': one_contribute_public_good,
                    'contributor_utility': contributor_utility,
                    'free_rider_utility': free_rider_utility,
                    'total_utility': contributor_utility + free_rider_utility * (n_players - 1)
                },
                'no_contribute': {
                    'contributions': [0.0] * n_players,
                    'public_good_level': no_contribute_public_good,
                    'individual_utilities': [no_contribute_utility] * n_players,
                    'total_utility': 0.0
                }
            },
            'free_rider_incentive': free_rider_incentive,
            'is_free_riding_beneficial': free_rider_incentive > 0,
            'n_players': n_players
        }

    def group_size_effect_analysis(self, group_sizes: List[int],
                                 benefit_coefficient: float = 1.0,
                                 cost_coefficient: float = 1.0,
                                 efficiency: float = 1.0) -> Dict[str, Any]:
        """
        群體規模效應分析（奧爾森效應）

        Args:
            group_sizes: 要分析的群體規模列表
            benefit_coefficient: 收益係數
            cost_coefficient: 成本係數
            efficiency: 效率

        Returns:
            dict: 群體規模效應分析結果
        """
        results = {}

        for size in group_sizes:
            # 計算該規模下的納什均衡
            benefit_coefficients = [benefit_coefficient] * size
            cost_coefficients = [cost_coefficient] * size

            equilibrium = self.nash_equilibrium_analysis(
                size, benefit_coefficients, cost_coefficients, "linear", efficiency
            )

            # 計算每人平均貢獻和效用
            avg_contribution = equilibrium['total_contribution'] / size
            avg_utility = equilibrium['total_utility'] / size

            # 計算提供率（實際提供/最優提供）
            # 假設最優提供是所有人都充分貢獻時的水準
            optimal_total_contribution = size * max(0, (benefit_coefficient * efficiency - cost_coefficient))
            if optimal_total_contribution > 0:
                provision_rate = equilibrium['total_contribution'] / optimal_total_contribution
            else:
                provision_rate = 0.0

            results[size] = {
                'equilibrium': equilibrium,
                'average_contribution': avg_contribution,
                'average_utility': avg_utility,
                'provision_rate': provision_rate,
                'free_rider_ratio': sum(1 for c in equilibrium['equilibrium_contributions'] if c == 0) / size
            }

        return {
            'group_size_results': results,
            'group_sizes': group_sizes,
            'analysis_parameters': {
                'benefit_coefficient': benefit_coefficient,
                'cost_coefficient': cost_coefficient,
                'efficiency': efficiency
            }
        }

    def olson_logic_demonstration(self, small_group_size: int = 3, large_group_size: int = 20) -> Dict[str, Any]:
        """
        演示奧爾森邏輯：小群體 vs 大群體

        Args:
            small_group_size: 小群體規模
            large_group_size: 大群體規模

        Returns:
            dict: 奧爾森邏輯演示結果
        """
        # 分析小群體
        small_group_analysis = self.nash_equilibrium_analysis(
            small_group_size, [1.0] * small_group_size
        )

        # 分析大群體
        large_group_analysis = self.nash_equilibrium_analysis(
            large_group_size, [1.0] * large_group_size
        )

        # 計算比較指標
        small_avg_contribution = small_group_analysis['total_contribution'] / small_group_size
        large_avg_contribution = large_group_analysis['total_contribution'] / large_group_size

        small_avg_utility = small_group_analysis['total_utility'] / small_group_size
        large_avg_utility = large_group_analysis['total_utility'] / large_group_size

        return {
            'small_group': {
                'size': small_group_size,
                'analysis': small_group_analysis,
                'average_contribution': small_avg_contribution,
                'average_utility': small_avg_utility
            },
            'large_group': {
                'size': large_group_size,
                'analysis': large_group_analysis,
                'average_contribution': large_avg_contribution,
                'average_utility': large_avg_utility
            },
            'comparison': {
                'contribution_ratio': small_avg_contribution / large_avg_contribution if large_avg_contribution > 0 else float('inf'),
                'utility_ratio': small_avg_utility / large_avg_utility if large_avg_utility > 0 else float('inf'),
                'olson_effect_confirmed': small_avg_contribution > large_avg_contribution
            }
        }

    def incentive_mechanism_analysis(self, n_players: int, mechanisms: List[str],
                                   base_benefit: float = 1.0, base_cost: float = 1.0) -> Dict[str, Any]:
        """
        激勵機制分析

        Args:
            n_players: 玩家數量
            mechanisms: 激勵機制列表 ["baseline", "subsidy", "penalty", "matching"]
            base_benefit: 基礎收益
            base_cost: 基礎成本

        Returns:
            dict: 激勵機制分析結果
        """
        results = {}

        for mechanism in mechanisms:
            if mechanism == "baseline":
                # 基線情況：無激勵
                benefit_coeffs = [base_benefit] * n_players
                cost_coeffs = [base_cost] * n_players

            elif mechanism == "subsidy":
                # 補貼機制：降低貢獻成本
                benefit_coeffs = [base_benefit] * n_players
                cost_coeffs = [base_cost * 0.5] * n_players

            elif mechanism == "penalty":
                # 懲罰機制：不貢獻者承擔額外成本
                benefit_coeffs = [base_benefit] * n_players
                cost_coeffs = [base_cost] * n_players
                # 這裡簡化處理，實際應該在效用函數中加入懲罰項

            elif mechanism == "matching":
                # 配對機制：政府按比例配對個人貢獻
                benefit_coeffs = [base_benefit * 1.5] * n_players  # 相當於提高收益
                cost_coeffs = [base_cost] * n_players

            else:
                raise ValueError(f"未知的激勵機制: {mechanism}")

            # 計算該機制下的均衡
            equilibrium = self.nash_equilibrium_analysis(
                n_players, benefit_coeffs, cost_coeffs
            )

            results[mechanism] = {
                'equilibrium': equilibrium,
                'average_contribution': equilibrium['total_contribution'] / n_players,
                'average_utility': equilibrium['total_utility'] / n_players,
                'total_welfare': equilibrium['total_utility']
            }

        return {
            'mechanism_results': results,
            'mechanisms': mechanisms,
            'n_players': n_players,
            'ranking_by_welfare': sorted(
                mechanisms,
                key=lambda m: results[m]['total_welfare'],
                reverse=True
            )
        }

    def print_collective_action_analysis(self, n_players: int, scenario_name: str = "基本場景"):
        """
        打印集體行動問題的詳細分析

        Args:
            n_players: 玩家數量
            scenario_name: 場景名稱
        """
        print("=" * 60)
        print(f"集體行動問題分析: {scenario_name}")
        print("=" * 60)
        print(f"參與者數量: {n_players}")
        print()

        # 1. 納什均衡分析
        print("1. 納什均衡分析")
        print("-" * 40)
        equilibrium = self.nash_equilibrium_analysis(n_players, [1.0] * n_players)

        print(f"均衡貢獻: {[f'{c:.3f}' for c in equilibrium['equilibrium_contributions']]}")
        print(f"公共物品水準: {equilibrium['public_good_level']:.3f}")
        print(f"總貢獻: {equilibrium['total_contribution']:.3f}")
        print(f"平均效用: {equilibrium['total_utility'] / n_players:.3f}")
        print()

        # 2. 搭便車分析
        print("2. 搭便車問題分析")
        print("-" * 40)
        free_rider = self.free_rider_analysis(n_players)

        print("場景比較:")
        scenarios = free_rider['scenarios']
        for name, scenario in scenarios.items():
            scenario_names = {
                'all_contribute': '全員貢獻',
                'one_contribute': '僅一人貢獻',
                'no_contribute': '無人貢獻'
            }
            print(f"  {scenario_names[name]}:")
            if name == 'one_contribute':
                print(f"    貢獻者效用: {scenario['contributor_utility']:.3f}")
                print(f"    搭便車者效用: {scenario['free_rider_utility']:.3f}")
            else:
                avg_utility = scenario['total_utility'] / n_players
                print(f"    平均效用: {avg_utility:.3f}")
            print(f"    總福利: {scenario['total_utility']:.3f}")

        print(f"\n搭便車誘因: {free_rider['free_rider_incentive']:.3f}")
        print(f"搭便車是否有利: {'是' if free_rider['is_free_riding_beneficial'] else '否'}")
        print()

        # 3. 奧爾森邏輯演示
        print("3. 奧爾森邏輯演示（群體規模效應）")
        print("-" * 40)
        if n_players <= 10:
            # 與更大群體比較
            olson_demo = self.olson_logic_demonstration(n_players, 20)
            small_data = olson_demo['small_group']
            large_data = olson_demo['large_group']

            print(f"小群體 ({small_data['size']}人):")
            print(f"  平均貢獻: {small_data['average_contribution']:.3f}")
            print(f"  平均效用: {small_data['average_utility']:.3f}")

            print(f"大群體 ({large_data['size']}人):")
            print(f"  平均貢獻: {large_data['average_contribution']:.3f}")
            print(f"  平均效用: {large_data['average_utility']:.3f}")

            comparison = olson_demo['comparison']
            print(f"\n奧爾森效應確認: {'是' if comparison['olson_effect_confirmed'] else '否'}")
            print(f"貢獻比率 (小/大): {comparison['contribution_ratio']:.2f}")

        else:
            print(f"當前群體規模較大 ({n_players}人)，符合奧爾森大群體特徵")

        print("\n分析完成！")


def main():
    """演示集體行動問題模型的主要功能"""
    print("集體行動問題模型演示")
    print("=" * 50)

    # 創建實例
    model = CollectiveActionSimple()

    # 1. 基本演示
    print("\n1. 基本集體行動問題分析")
    print("-" * 30)
    model.print_collective_action_analysis(5, "5人群體")

    # 2. 群體規模效應分析
    print("\n2. 群體規模效應分析")
    print("-" * 30)
    group_sizes = [2, 5, 10, 20, 50]
    size_analysis = model.group_size_effect_analysis(group_sizes)

    print("群體規模效應結果:")
    print(f"{'規模':<6} {'平均貢獻':<10} {'平均效用':<10} {'提供率':<10} {'搭便車比例':<12}")
    print("-" * 55)

    for size in group_sizes:
        result = size_analysis['group_size_results'][size]
        print(f"{size:<6} {result['average_contribution']:<10.3f} "
              f"{result['average_utility']:<10.3f} {result['provision_rate']:<10.3f} "
              f"{result['free_rider_ratio']:<12.2%}")

    # 3. 激勵機制比較
    print("\n3. 激勵機制效果比較")
    print("-" * 30)
    mechanisms = ["baseline", "subsidy", "matching"]
    incentive_analysis = model.incentive_mechanism_analysis(10, mechanisms)

    print("激勵機制效果:")
    print(f"{'機制':<12} {'平均貢獻':<10} {'平均效用':<10} {'總福利':<10}")
    print("-" * 45)

    mechanism_names = {
        "baseline": "基線",
        "subsidy": "補貼",
        "matching": "配對"
    }

    for mechanism in mechanisms:
        result = incentive_analysis['mechanism_results'][mechanism]
        name = mechanism_names.get(mechanism, mechanism)
        print(f"{name:<12} {result['average_contribution']:<10.3f} "
              f"{result['average_utility']:<10.3f} {result['total_welfare']:<10.3f}")

    best_mechanism = incentive_analysis['ranking_by_welfare'][0]
    print(f"\n最佳機制: {mechanism_names.get(best_mechanism, best_mechanism)}")

    # 4. 生產函數比較
    print("\n4. 不同生產函數效果比較")
    print("-" * 30)
    production_functions = ["linear", "threshold", "accelerating", "diminishing"]

    print("生產函數比較:")
    print(f"{'函數類型':<12} {'總貢獻':<10} {'公共物品':<10} {'總效用':<10}")
    print("-" * 45)

    for func_type in production_functions:
        try:
            analysis = model.nash_equilibrium_analysis(
                5, [1.0] * 5, production_function=func_type
            )
            print(f"{func_type:<12} {analysis['total_contribution']:<10.3f} "
                  f"{analysis['public_good_level']:<10.3f} {analysis['total_utility']:<10.3f}")
        except Exception as e:
            print(f"{func_type:<12} 計算錯誤: {e}")

    print("\n演示完成！")


if __name__ == "__main__":
    main()