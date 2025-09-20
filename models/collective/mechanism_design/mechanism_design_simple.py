"""
機制設計模型的簡單實現
Mechanism Design Models - Simple Implementation

基於VCG機制和收入等價定理的數學模型

Author: Claude Code Assistant
License: MIT
"""

import math
import random
from typing import List, Dict, Tuple, Any, Optional, Union, Callable


class MechanismDesignSimple:
    """
    機制設計模型的簡單實現類

    實現機制設計理論的核心概念，包括：
    - VCG機制（維克瑞-克拉克-格羅夫斯）
    - 各種拍賣機制
    - 收入等價定理
    - 激勵相容性分析
    """

    def __init__(self):
        """初始化機制設計分析器"""
        self.analysis_history = []

    def vickrey_auction(self, bids: List[float], num_items: int = 1) -> Dict[str, Any]:
        """
        維克瑞拍賣（第二價格密封拍賣）

        Args:
            bids: 各參與者的出價列表
            num_items: 拍賣物品數量

        Returns:
            dict: 拍賣結果
        """
        if len(bids) < 2:
            raise ValueError("至少需要2個出價者")

        n_bidders = len(bids)

        # 創建(出價, 出價者ID)的列表並排序
        bid_pairs = [(bids[i], i) for i in range(n_bidders)]
        bid_pairs.sort(reverse=True)  # 按出價從高到低排序

        winners = []
        payments = []

        # 分配前num_items個最高出價者
        for i in range(min(num_items, n_bidders)):
            winner_bid, winner_id = bid_pairs[i]

            if i + 1 < len(bid_pairs):
                # 支付第二高價（或下一個最高價）
                payment = bid_pairs[i + 1][0]
            else:
                # 如果沒有更低的出價，支付0（保留價格）
                payment = 0.0

            winners.append(winner_id)
            payments.append(payment)

        # 計算各參與者的效用
        utilities = [0.0] * n_bidders
        for i, winner_id in enumerate(winners):
            utilities[winner_id] = bids[winner_id] - payments[i]

        total_revenue = sum(payments)

        return {
            'mechanism_type': 'vickrey',
            'winners': winners,
            'payments': payments,
            'utilities': utilities,
            'total_revenue': total_revenue,
            'allocation_efficient': True,  # 維克瑞拍賣是有效率的
            'truthful': True  # 維克瑞拍賣是真實偏好顯示的
        }

    def first_price_auction(self, bids: List[float], num_items: int = 1) -> Dict[str, Any]:
        """
        第一價格密封拍賣

        Args:
            bids: 各參與者的出價列表
            num_items: 拍賣物品數量

        Returns:
            dict: 拍賣結果
        """
        if len(bids) < 2:
            raise ValueError("至少需要2個出價者")

        n_bidders = len(bids)

        # 創建(出價, 出價者ID)的列表並排序
        bid_pairs = [(bids[i], i) for i in range(n_bidders)]
        bid_pairs.sort(reverse=True)

        winners = []
        payments = []

        # 分配前num_items個最高出價者
        for i in range(min(num_items, n_bidders)):
            winner_bid, winner_id = bid_pairs[i]
            payment = winner_bid  # 支付自己的出價

            winners.append(winner_id)
            payments.append(payment)

        # 計算各參與者的效用
        utilities = [0.0] * n_bidders
        for i, winner_id in enumerate(winners):
            utilities[winner_id] = bids[winner_id] - payments[i]

        total_revenue = sum(payments)

        return {
            'mechanism_type': 'first_price',
            'winners': winners,
            'payments': payments,
            'utilities': utilities,
            'total_revenue': total_revenue,
            'allocation_efficient': True,  # 假設出價反映真實價值
            'truthful': False  # 第一價格拍賣不是真實偏好顯示的
        }

    def english_auction_simulation(self, valuations: List[float], increment: float = 0.1) -> Dict[str, Any]:
        """
        英式拍賣模擬

        Args:
            valuations: 各參與者的真實估值
            increment: 價格增量

        Returns:
            dict: 拍賣結果
        """
        if len(valuations) < 2:
            raise ValueError("至少需要2個參與者")

        n_bidders = len(valuations)
        current_price = 0.0
        active_bidders = list(range(n_bidders))

        auction_history = []

        while len(active_bidders) > 1:
            # 移除估值低於當前價格的出價者
            active_bidders = [i for i in active_bidders if valuations[i] >= current_price + increment]

            auction_history.append({
                'price': current_price,
                'active_bidders': active_bidders.copy()
            })

            if len(active_bidders) <= 1:
                break

            current_price += increment

        # 確定獲勜者和支付價格
        if active_bidders:
            winner = active_bidders[0]
            payment = current_price
        else:
            # 如果沒有人願意出價，選擇估值最高的
            winner = max(range(n_bidders), key=lambda i: valuations[i])
            payment = 0.0

        utilities = [0.0] * n_bidders
        utilities[winner] = valuations[winner] - payment

        return {
            'mechanism_type': 'english',
            'winner': winner,
            'payment': payment,
            'utilities': utilities,
            'total_revenue': payment,
            'auction_history': auction_history,
            'final_price': current_price,
            'truthful': True  # 英式拍賣中真實出價是占優策略
        }

    def vcg_mechanism(self, valuations: List[List[float]], allocation_function: Optional[Callable] = None) -> Dict[str, Any]:
        """
        VCG機制的一般實現

        Args:
            valuations: 二維列表，valuations[i][j]表示參與者i對結果j的估值
            allocation_function: 分配函數，如果為None則使用社會福利最大化

        Returns:
            dict: VCG機制結果
        """
        n_agents = len(valuations)
        n_outcomes = len(valuations[0]) if valuations else 0

        if not all(len(val) == n_outcomes for val in valuations):
            raise ValueError("所有參與者的估值向量長度必須相同")

        # 如果沒有提供分配函數，使用社會福利最大化
        if allocation_function is None:
            def social_welfare_maximization(vals):
                total_values = [sum(vals[i][j] for i in range(len(vals))) for j in range(len(vals[0]))]
                return max(range(len(total_values)), key=lambda x: total_values[x])
            allocation_function = social_welfare_maximization

        # 確定社會最優分配
        optimal_outcome = allocation_function(valuations)

        # 計算VCG支付
        payments = []

        for i in range(n_agents):
            # 計算沒有參與者i時的最優社會福利
            other_valuations = [valuations[j] for j in range(n_agents) if j != i]
            if other_valuations:
                other_optimal = allocation_function(other_valuations)
                other_welfare = sum(other_valuations[j][other_optimal] for j in range(len(other_valuations)))
            else:
                other_welfare = 0.0

            # 計算有參與者i時其他人的福利
            others_welfare_with_i = sum(valuations[j][optimal_outcome] for j in range(n_agents) if j != i)

            # VCG支付 = 參與者i對其他人造成的外部性
            payment = other_welfare - others_welfare_with_i
            payments.append(payment)

        # 計算效用
        utilities = []
        for i in range(n_agents):
            utility = valuations[i][optimal_outcome] - payments[i]
            utilities.append(utility)

        total_welfare = sum(valuations[i][optimal_outcome] for i in range(n_agents))
        total_payment = sum(payments)

        return {
            'mechanism_type': 'vcg',
            'optimal_outcome': optimal_outcome,
            'payments': payments,
            'utilities': utilities,
            'total_welfare': total_welfare,
            'total_payment': total_payment,
            'budget_balanced': abs(total_payment) < 1e-10,  # 檢查是否預算平衡
            'truthful': True,
            'efficient': True
        }

    def revenue_equivalence_theorem_demo(self, valuations: List[float],
                                       mechanisms: List[str] = ['vickrey', 'first_price']) -> Dict[str, Any]:
        """
        收入等價定理演示

        Args:
            valuations: 參與者的真實估值
            mechanisms: 要比較的機制列表

        Returns:
            dict: 收入等價性分析結果
        """
        if not all(v >= 0 for v in valuations):
            raise ValueError("估值必須非負")

        results = {}

        for mechanism in mechanisms:
            if mechanism == 'vickrey':
                # 維克瑞拍賣中，理性參與者會報真實估值
                result = self.vickrey_auction(valuations)
            elif mechanism == 'first_price':
                # 第一價格拍賣中需要計算均衡出價策略
                # 簡化：假設對稱均衡下的出價函數
                equilibrium_bids = self._first_price_equilibrium_bids(valuations)
                result = self.first_price_auction(equilibrium_bids)
            elif mechanism == 'english':
                result = self.english_auction_simulation(valuations)
            else:
                continue

            results[mechanism] = result

        # 檢查收入等價性
        revenues = {mech: results[mech]['total_revenue'] for mech in results}
        revenue_values = list(revenues.values())

        # 檢查是否所有收入都相等（在容忍誤差內）
        revenue_equivalent = all(abs(r - revenue_values[0]) < 1e-6 for r in revenue_values)

        return {
            'mechanisms': mechanisms,
            'results': results,
            'revenues': revenues,
            'revenue_equivalent': revenue_equivalent,
            'theorem_holds': revenue_equivalent
        }

    def _first_price_equilibrium_bids(self, valuations: List[float]) -> List[float]:
        """
        計算第一價格拍賣的對稱均衡出價

        Args:
            valuations: 真實估值

        Returns:
            list: 均衡出價策略
        """
        # 簡化的均衡出價函數：對於均勻分布的估值
        # b(v) = (n-1)/n * v，其中n是參與者數量
        n = len(valuations)
        if n <= 1:
            return valuations

        bid_factor = (n - 1) / n
        return [v * bid_factor for v in valuations]

    def mechanism_comparison(self, valuations: List[float],
                           criteria: List[str] = ['revenue', 'efficiency', 'truthfulness']) -> Dict[str, Any]:
        """
        多機制比較分析

        Args:
            valuations: 參與者估值
            criteria: 比較標準

        Returns:
            dict: 機制比較結果
        """
        mechanisms = ['vickrey', 'first_price', 'english']
        comparison_results = {}

        for mechanism in mechanisms:
            if mechanism == 'vickrey':
                result = self.vickrey_auction(valuations)
            elif mechanism == 'first_price':
                equilibrium_bids = self._first_price_equilibrium_bids(valuations)
                result = self.first_price_auction(equilibrium_bids)
            elif mechanism == 'english':
                result = self.english_auction_simulation(valuations)

            comparison_results[mechanism] = result

        # 按各標準進行比較
        analysis = {}

        if 'revenue' in criteria:
            revenues = {mech: comparison_results[mech]['total_revenue'] for mech in mechanisms}
            best_revenue_mechanism = max(revenues.keys(), key=lambda x: revenues[x])
            analysis['revenue'] = {
                'revenues': revenues,
                'best_mechanism': best_revenue_mechanism,
                'ranking': sorted(revenues.keys(), key=lambda x: revenues[x], reverse=True)
            }

        if 'efficiency' in criteria:
            # 效率：最高估值者是否獲勝
            max_valuation_holder = max(range(len(valuations)), key=lambda i: valuations[i])
            efficiency_scores = {}

            for mech in mechanisms:
                result = comparison_results[mech]
                if 'winners' in result:
                    efficient = max_valuation_holder in result['winners']
                elif 'winner' in result:
                    efficient = result['winner'] == max_valuation_holder
                else:
                    efficient = False
                efficiency_scores[mech] = efficient

            analysis['efficiency'] = efficiency_scores

        if 'truthfulness' in criteria:
            truthfulness_scores = {mech: comparison_results[mech].get('truthful', False) for mech in mechanisms}
            analysis['truthfulness'] = truthfulness_scores

        return {
            'comparison_results': comparison_results,
            'analysis': analysis,
            'valuations': valuations
        }

    def public_goods_provision_vcg(self, valuations: List[float], cost: float) -> Dict[str, Any]:
        """
        公共物品提供的VCG機制

        Args:
            valuations: 各個體對公共物品的估值
            cost: 公共物品的提供成本

        Returns:
            dict: 公共物品提供分析結果
        """
        n_agents = len(valuations)
        total_valuation = sum(valuations)

        # 決定是否提供公共物品（社會福利最大化）
        provide_good = total_valuation >= cost

        if provide_good:
            # 計算VCG支付
            payments = []
            for i in range(n_agents):
                # 計算沒有參與者i時的決策
                others_valuation = total_valuation - valuations[i]
                would_provide_without_i = others_valuation >= cost

                if would_provide_without_i:
                    # 如果沒有i也會提供，i的支付為0
                    payment = 0.0
                else:
                    # 如果沒有i就不會提供，i需要補償其他人的損失
                    payment = cost - others_valuation

                payments.append(payment)

            # 計算效用
            utilities = [valuations[i] - payments[i] for i in range(n_agents)]
            net_social_welfare = total_valuation - cost
        else:
            payments = [0.0] * n_agents
            utilities = [0.0] * n_agents
            net_social_welfare = 0.0

        budget_balance = sum(payments) - (cost if provide_good else 0)

        return {
            'provide_good': provide_good,
            'total_valuation': total_valuation,
            'cost': cost,
            'payments': payments,
            'utilities': utilities,
            'net_social_welfare': net_social_welfare,
            'budget_surplus': budget_balance,
            'budget_balanced': abs(budget_balance) < 1e-10
        }

    def print_mechanism_analysis(self, valuations: List[float], scenario_name: str = "基本拍賣場景"):
        """
        打印機制設計的詳細分析

        Args:
            valuations: 參與者估值
            scenario_name: 場景名稱
        """
        print("=" * 60)
        print(f"機制設計分析: {scenario_name}")
        print("=" * 60)
        print(f"參與者估值: {valuations}")
        print()

        # 1. 機制比較
        print("1. 不同拍賣機制比較")
        print("-" * 40)
        comparison = self.mechanism_comparison(valuations)

        print("收入比較:")
        revenues = comparison['analysis']['revenue']['revenues']
        for mechanism, revenue in revenues.items():
            print(f"  {mechanism}: {revenue:.3f}")

        print(f"\n最高收入機制: {comparison['analysis']['revenue']['best_mechanism']}")

        print("\n效率性比較:")
        efficiency = comparison['analysis']['efficiency']
        for mechanism, is_efficient in efficiency.items():
            print(f"  {mechanism}: {'有效率' if is_efficient else '無效率'}")

        print("\n真實偏好顯示:")
        truthfulness = comparison['analysis']['truthfulness']
        for mechanism, is_truthful in truthfulness.items():
            print(f"  {mechanism}: {'是' if is_truthful else '否'}")
        print()

        # 2. 收入等價定理驗證
        print("2. 收入等價定理驗證")
        print("-" * 40)
        revenue_equiv = self.revenue_equivalence_theorem_demo(valuations)

        print("各機制收入:")
        for mechanism, revenue in revenue_equiv['revenues'].items():
            print(f"  {mechanism}: {revenue:.6f}")

        print(f"\n收入等價性: {'成立' if revenue_equiv['theorem_holds'] else '不成立'}")
        print()

        # 3. VCG機制分析
        print("3. VCG機制分析")
        print("-" * 40)

        # 創建簡單的二元選擇問題：是否分配給估值最高的人
        n = len(valuations)
        vcg_valuations = [[valuations[i] if j == i else 0 for j in range(n)] for i in range(n)]

        vcg_result = self.vcg_mechanism(vcg_valuations)

        print(f"最優分配: 參與者 {vcg_result['optimal_outcome']}")
        print(f"VCG支付: {[f'{p:.3f}' for p in vcg_result['payments']]}")
        print(f"參與者效用: {[f'{u:.3f}' for u in vcg_result['utilities']]}")
        print(f"總社會福利: {vcg_result['total_welfare']:.3f}")
        print(f"預算平衡: {'是' if vcg_result['budget_balanced'] else '否'}")
        print()

        # 4. 公共物品提供分析
        print("4. 公共物品提供分析")
        print("-" * 40)
        cost = sum(valuations) * 0.8  # 設定成本為總估值的80%

        public_goods = self.public_goods_provision_vcg(valuations, cost)

        print(f"提供成本: {cost:.3f}")
        print(f"總估值: {public_goods['total_valuation']:.3f}")
        print(f"是否提供: {'是' if public_goods['provide_good'] else '否'}")
        print(f"VCG支付: {[f'{p:.3f}' for p in public_goods['payments']]}")
        print(f"淨社會福利: {public_goods['net_social_welfare']:.3f}")
        print(f"預算盈餘: {public_goods['budget_surplus']:.3f}")

        print("\n分析完成！")


def main():
    """演示機制設計模型的主要功能"""
    print("機制設計模型演示")
    print("=" * 50)

    # 創建實例
    model = MechanismDesignSimple()

    # 1. 基本演示
    print("\n1. 基本機制設計分析")
    print("-" * 30)
    valuations = [10, 8, 6, 4, 2]
    model.print_mechanism_analysis(valuations, "五人拍賣")

    # 2. 收入等價性驗證
    print("\n2. 收入等價定理驗證（多場景）")
    print("-" * 30)

    test_scenarios = [
        [10, 5],
        [15, 10, 5],
        [20, 15, 10, 5],
        [100, 80, 60, 40, 20]
    ]

    print("不同場景下的收入等價性:")
    print(f"{'場景':<15} {'維克瑞':<10} {'第一價格':<10} {'等價性':<8}")
    print("-" * 50)

    for i, vals in enumerate(test_scenarios):
        revenue_test = model.revenue_equivalence_theorem_demo(vals)
        vickrey_rev = revenue_test['revenues'].get('vickrey', 0)
        first_price_rev = revenue_test['revenues'].get('first_price', 0)
        equivalent = revenue_test['theorem_holds']

        print(f"場景{i+1}({len(vals)}人):<15 {vickrey_rev:<10.3f} {first_price_rev:<10.3f} {'✓' if equivalent else '✗':<8}")

    # 3. VCG機制的應用
    print("\n3. VCG機制應用示例")
    print("-" * 30)

    # 示例：分配兩個不同物品
    print("多物品分配問題:")
    # 參與者對不同物品組合的估值
    multi_item_valuations = [
        [0, 5, 3, 7],   # 參與者1: [無,物品A,物品B,兩者]
        [0, 4, 6, 9],   # 參與者2
        [0, 3, 4, 6]    # 參與者3
    ]

    vcg_multi = model.vcg_mechanism(multi_item_valuations)

    allocation_names = ["無分配", "物品A", "物品B", "兩個物品"]
    print(f"最優分配方案: {allocation_names[vcg_multi['optimal_outcome']]}")
    print(f"各參與者支付: {[f'{p:.3f}' for p in vcg_multi['payments']]}")
    print(f"各參與者效用: {[f'{u:.3f}' for u in vcg_multi['utilities']]}")

    # 4. 公共物品提供的不同成本情況
    print("\n4. 公共物品提供成本敏感性分析")
    print("-" * 30)

    public_valuations = [15, 10, 8, 5, 2]
    costs = [20, 30, 40, 50]

    print("不同成本下的公共物品提供決策:")
    print(f"{'成本':<8} {'總估值':<8} {'提供':<6} {'總支付':<8} {'盈餘':<8}")
    print("-" * 45)

    total_val = sum(public_valuations)
    for cost in costs:
        pg_result = model.public_goods_provision_vcg(public_valuations, cost)
        provide = "是" if pg_result['provide_good'] else "否"
        total_payment = sum(pg_result['payments'])
        surplus = pg_result['budget_surplus']

        print(f"{cost:<8} {total_val:<8} {provide:<6} {total_payment:<8.1f} {surplus:<8.1f}")

    print("\n演示完成！")


if __name__ == "__main__":
    main()