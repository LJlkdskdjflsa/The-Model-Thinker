"""
隨機漫步模型的簡單實現
Random Walk Models - Simple Implementation

包含各種隨機漫步模型的數學實現

Author: Claude Code Assistant
License: MIT
"""

import math
import random
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
from collections import defaultdict


class RandomWalkSimple:
    """
    隨機漫步模型的簡單實現類

    實現各種隨機漫步模型，包括：
    - 簡單隨機漫步
    - 偏移隨機漫步
    - 幾何布朗運動
    - 醉漢問題
    - 股價模型
    """

    def __init__(self, seed: Optional[int] = None):
        """
        初始化隨機漫步模型

        Args:
            seed: 隨機數種子，用於重現結果
        """
        if seed is not None:
            random.seed(seed)
        self.simulation_history = []

    def simple_random_walk_1d(self, steps: int, start_position: float = 0.0,
                             step_size: float = 1.0, probability: float = 0.5) -> Dict[str, Any]:
        """
        一維簡單隨機漫步

        Args:
            steps: 步數
            start_position: 起始位置
            step_size: 步長
            probability: 向右移動的概率

        Returns:
            dict: 隨機漫步結果
        """
        positions = [start_position]
        current_position = start_position

        for _ in range(steps):
            if random.random() < probability:
                current_position += step_size  # 向右
            else:
                current_position -= step_size  # 向左
            positions.append(current_position)

        # 計算統計量
        max_position = max(positions)
        min_position = min(positions)
        final_position = positions[-1]
        displacement = final_position - start_position

        # 計算首次到達統計
        first_positive = None
        first_negative = None
        for i, pos in enumerate(positions):
            if first_positive is None and pos > start_position:
                first_positive = i
            if first_negative is None and pos < start_position:
                first_negative = i

        return {
            'walk_type': 'simple_1d',
            'steps': steps,
            'positions': positions,
            'start_position': start_position,
            'final_position': final_position,
            'displacement': displacement,
            'max_position': max_position,
            'min_position': min_position,
            'range': max_position - min_position,
            'first_positive_time': first_positive,
            'first_negative_time': first_negative,
            'probability': probability,
            'step_size': step_size
        }

    def random_walk_2d(self, steps: int, start_position: Tuple[float, float] = (0.0, 0.0),
                      step_size: float = 1.0) -> Dict[str, Any]:
        """
        二維隨機漫步

        Args:
            steps: 步數
            start_position: 起始位置 (x, y)
            step_size: 步長

        Returns:
            dict: 二維隨機漫步結果
        """
        positions = [start_position]
        current_x, current_y = start_position

        for _ in range(steps):
            # 隨機選擇四個方向之一
            direction = random.choice(['up', 'down', 'left', 'right'])

            if direction == 'up':
                current_y += step_size
            elif direction == 'down':
                current_y -= step_size
            elif direction == 'left':
                current_x -= step_size
            else:  # right
                current_x += step_size

            positions.append((current_x, current_y))

        # 計算統計量
        distances_from_origin = [math.sqrt(x**2 + y**2) for x, y in positions]
        max_distance = max(distances_from_origin)
        final_distance = distances_from_origin[-1]

        return {
            'walk_type': '2d',
            'steps': steps,
            'positions': positions,
            'start_position': start_position,
            'final_position': positions[-1],
            'final_distance_from_origin': final_distance,
            'max_distance_from_origin': max_distance,
            'step_size': step_size
        }

    def biased_random_walk(self, steps: int, drift: float, volatility: float = 1.0,
                          start_position: float = 0.0) -> Dict[str, Any]:
        """
        有偏移的隨機漫步（連續時間近似）

        Args:
            steps: 步數
            drift: 漂移率（期望移動方向）
            volatility: 波動率
            start_position: 起始位置

        Returns:
            dict: 偏移隨機漫步結果
        """
        positions = [start_position]
        current_position = start_position
        dt = 1.0  # 時間步長

        for _ in range(steps):
            # dX = drift * dt + volatility * sqrt(dt) * Z
            # 其中 Z ~ N(0,1)
            random_shock = random.gauss(0, 1)
            change = drift * dt + volatility * math.sqrt(dt) * random_shock
            current_position += change
            positions.append(current_position)

        final_position = positions[-1]
        max_position = max(positions)
        min_position = min(positions)

        return {
            'walk_type': 'biased',
            'steps': steps,
            'positions': positions,
            'start_position': start_position,
            'final_position': final_position,
            'displacement': final_position - start_position,
            'max_position': max_position,
            'min_position': min_position,
            'drift': drift,
            'volatility': volatility
        }

    def geometric_brownian_motion(self, steps: int, initial_price: float, mu: float,
                                 sigma: float, dt: float = 1.0) -> Dict[str, Any]:
        """
        幾何布朗運動（股價模型）

        Args:
            steps: 步數
            initial_price: 初始價格
            mu: 漂移率（期望收益率）
            sigma: 波動率
            dt: 時間步長

        Returns:
            dict: 幾何布朗運動結果
        """
        prices = [initial_price]
        current_price = initial_price

        for _ in range(steps):
            # dS = mu * S * dt + sigma * S * sqrt(dt) * Z
            random_shock = random.gauss(0, 1)
            change = mu * current_price * dt + sigma * current_price * math.sqrt(dt) * random_shock
            current_price += change
            current_price = max(0, current_price)  # 價格不能為負
            prices.append(current_price)

        # 計算收益率
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)

        max_price = max(prices)
        min_price = min(prices)
        final_price = prices[-1]
        total_return = (final_price - initial_price) / initial_price

        return {
            'walk_type': 'geometric_brownian',
            'steps': steps,
            'prices': prices,
            'returns': returns,
            'initial_price': initial_price,
            'final_price': final_price,
            'max_price': max_price,
            'min_price': min_price,
            'total_return': total_return,
            'mu': mu,
            'sigma': sigma,
            'dt': dt
        }

    def drunk_man_problem(self, cliff_position: int = 0, home_position: int = 10,
                         start_position: int = 5, p_away_from_cliff: float = 0.5) -> Dict[str, Any]:
        """
        醉漢問題：在懸崖和家之間的隨機漫步

        Args:
            cliff_position: 懸崖位置
            home_position: 家的位置
            start_position: 起始位置
            p_away_from_cliff: 遠離懸崖方向移動的概率

        Returns:
            dict: 醉漢問題結果
        """
        if not cliff_position < start_position < home_position:
            raise ValueError("起始位置必須在懸崖和家之間")

        positions = [start_position]
        current_position = start_position
        steps = 0
        max_steps = 10000  # 防止無限循環

        while (current_position != cliff_position and
               current_position != home_position and
               steps < max_steps):

            if random.random() < p_away_from_cliff:
                # 遠離懸崖（向家移動）
                current_position += 1
            else:
                # 向懸崖移動
                current_position -= 1

            positions.append(current_position)
            steps += 1

        # 確定結果
        if current_position == home_position:
            outcome = 'home'
        elif current_position == cliff_position:
            outcome = 'cliff'
        else:
            outcome = 'timeout'

        # 計算到達家的理論概率（如果有解析解）
        if p_away_from_cliff != 0.5:
            # 對於非對稱隨機漫步的吸收概率
            p_towards_cliff = 1 - p_away_from_cliff
            r = p_towards_cliff / p_away_from_cliff

            if r != 1:
                # P(到達家) = (1 - r^(start-cliff)) / (1 - r^(home-cliff))
                numerator = 1 - r**(start_position - cliff_position)
                denominator = 1 - r**(home_position - cliff_position)
                theoretical_prob_home = numerator / denominator
            else:
                theoretical_prob_home = (start_position - cliff_position) / (home_position - cliff_position)
        else:
            theoretical_prob_home = (start_position - cliff_position) / (home_position - cliff_position)

        return {
            'walk_type': 'drunk_man',
            'outcome': outcome,
            'steps_taken': steps,
            'positions': positions,
            'start_position': start_position,
            'cliff_position': cliff_position,
            'home_position': home_position,
            'p_away_from_cliff': p_away_from_cliff,
            'theoretical_prob_home': theoretical_prob_home,
            'reached_home': outcome == 'home'
        }

    def first_passage_time_analysis(self, target: float, steps: int,
                                   start_position: float = 0.0) -> Dict[str, Any]:
        """
        首次通過時間分析

        Args:
            target: 目標位置
            steps: 最大步數
            start_position: 起始位置

        Returns:
            dict: 首次通過時間分析結果
        """
        walk = self.simple_random_walk_1d(steps, start_position)
        positions = walk['positions']

        first_passage_time = None
        crossed = False

        for i, position in enumerate(positions):
            if (target > start_position and position >= target) or \
               (target < start_position and position <= target):
                first_passage_time = i
                crossed = True
                break

        return {
            'target': target,
            'start_position': start_position,
            'first_passage_time': first_passage_time,
            'crossed': crossed,
            'total_steps': steps,
            'positions': positions
        }

    def multiple_walks_statistics(self, num_walks: int, steps_per_walk: int,
                                 walk_type: str = 'simple_1d', **kwargs) -> Dict[str, Any]:
        """
        多次隨機漫步的統計分析

        Args:
            num_walks: 漫步次數
            steps_per_walk: 每次漫步的步數
            walk_type: 漫步類型
            **kwargs: 其他參數

        Returns:
            dict: 統計分析結果
        """
        walks = []
        final_positions = []
        max_distances = []

        for _ in range(num_walks):
            if walk_type == 'simple_1d':
                walk = self.simple_random_walk_1d(steps_per_walk, **kwargs)
                final_positions.append(walk['final_position'])
                max_distances.append(max(abs(p) for p in walk['positions']))
            elif walk_type == '2d':
                walk = self.random_walk_2d(steps_per_walk, **kwargs)
                final_positions.append(walk['final_distance_from_origin'])
                max_distances.append(walk['max_distance_from_origin'])
            elif walk_type == 'drunk_man':
                walk = self.drunk_man_problem(**kwargs)
                final_positions.append(1 if walk['reached_home'] else 0)
                max_distances.append(walk['steps_taken'])

            walks.append(walk)

        # 計算統計量
        mean_final = sum(final_positions) / len(final_positions)
        variance_final = sum((x - mean_final)**2 for x in final_positions) / len(final_positions)
        std_final = math.sqrt(variance_final)

        mean_max_distance = sum(max_distances) / len(max_distances)

        return {
            'num_walks': num_walks,
            'steps_per_walk': steps_per_walk,
            'walk_type': walk_type,
            'walks': walks,
            'final_positions': final_positions,
            'statistics': {
                'mean_final_position': mean_final,
                'std_final_position': std_final,
                'variance_final_position': variance_final,
                'mean_max_distance': mean_max_distance,
                'min_final_position': min(final_positions),
                'max_final_position': max(final_positions)
            }
        }

    def random_walk_convergence_test(self, steps_list: List[int], num_simulations: int = 100) -> Dict[str, Any]:
        """
        測試隨機漫步的收斂性質

        Args:
            steps_list: 不同步數的列表
            num_simulations: 每個步數的模擬次數

        Returns:
            dict: 收斂性測試結果
        """
        convergence_data = {}

        for steps in steps_list:
            results = self.multiple_walks_statistics(
                num_simulations, steps, 'simple_1d'
            )

            stats = results['statistics']
            convergence_data[steps] = {
                'steps': steps,
                'mean_displacement': stats['mean_final_position'],
                'std_displacement': stats['std_final_position'],
                'theoretical_std': math.sqrt(steps),  # 理論標準差
                'normalized_std': stats['std_final_position'] / math.sqrt(steps)
            }

        return {
            'convergence_data': convergence_data,
            'steps_list': steps_list,
            'num_simulations': num_simulations
        }

    def print_random_walk_analysis(self, scenario_name: str = "基本隨機漫步"):
        """
        打印隨機漫步的詳細分析

        Args:
            scenario_name: 場景名稱
        """
        print("=" * 60)
        print(f"隨機漫步模型分析: {scenario_name}")
        print("=" * 60)

        # 1. 一維隨機漫步
        print("1. 一維簡單隨機漫步")
        print("-" * 40)
        walk_1d = self.simple_random_walk_1d(100)
        print(f"步數: {walk_1d['steps']}")
        print(f"起始位置: {walk_1d['start_position']}")
        print(f"最終位置: {walk_1d['final_position']}")
        print(f"位移: {walk_1d['displacement']}")
        print(f"最大位置: {walk_1d['max_position']}")
        print(f"最小位置: {walk_1d['min_position']}")
        print(f"範圍: {walk_1d['range']}")
        print()

        # 2. 二維隨機漫步
        print("2. 二維隨機漫步")
        print("-" * 40)
        walk_2d = self.random_walk_2d(100)
        print(f"步數: {walk_2d['steps']}")
        print(f"起始位置: {walk_2d['start_position']}")
        print(f"最終位置: {walk_2d['final_position']}")
        print(f"與原點距離: {walk_2d['final_distance_from_origin']:.3f}")
        print(f"最大距離: {walk_2d['max_distance_from_origin']:.3f}")
        print()

        # 3. 醉漢問題
        print("3. 醉漢問題")
        print("-" * 40)
        drunk_result = self.drunk_man_problem(p_away_from_cliff=0.6)
        print(f"結果: {drunk_result['outcome']}")
        print(f"步數: {drunk_result['steps_taken']}")
        print(f"理論到家概率: {drunk_result['theoretical_prob_home']:.3f}")
        print(f"實際到家: {'是' if drunk_result['reached_home'] else '否'}")
        print()

        # 4. 股價模型
        print("4. 股價模型（幾何布朗運動）")
        print("-" * 40)
        stock = self.geometric_brownian_motion(100, 100, 0.05, 0.2)
        print(f"初始價格: {stock['initial_price']}")
        print(f"最終價格: {stock['final_price']:.2f}")
        print(f"總收益率: {stock['total_return']:.2%}")
        print(f"最高價: {stock['max_price']:.2f}")
        print(f"最低價: {stock['min_price']:.2f}")
        print()

        # 5. 多次模擬統計
        print("5. 多次模擬統計分析")
        print("-" * 40)
        multi_stats = self.multiple_walks_statistics(1000, 100)
        stats = multi_stats['statistics']
        print(f"模擬次數: {multi_stats['num_walks']}")
        print(f"平均最終位置: {stats['mean_final_position']:.3f}")
        print(f"標準差: {stats['std_final_position']:.3f}")
        print(f"理論標準差: {math.sqrt(100):.3f}")
        print(f"標準化標準差: {stats['std_final_position']/math.sqrt(100):.3f}")

        print("\n分析完成！")


def main():
    """演示隨機漫步模型的主要功能"""
    print("隨機漫步模型演示")
    print("=" * 50)

    # 創建實例
    model = RandomWalkSimple(seed=42)

    # 1. 基本演示
    print("\n1. 基本隨機漫步分析")
    print("-" * 30)
    model.print_random_walk_analysis("綜合隨機漫步模型")

    # 2. 醉漢問題的概率分析
    print("\n2. 醉漢問題概率分析")
    print("-" * 30)
    print("不同偏移概率下到達家的機率:")
    print(f"{'偏移概率':<10} {'理論概率':<10} {'模擬概率':<10} {'樣本數':<8}")
    print("-" * 42)

    for p in [0.4, 0.5, 0.6, 0.7, 0.8]:
        # 理論計算
        theory_result = model.drunk_man_problem(p_away_from_cliff=p)
        theoretical_prob = theory_result['theoretical_prob_home']

        # 多次模擬
        success_count = 0
        simulations = 1000
        for _ in range(simulations):
            result = model.drunk_man_problem(p_away_from_cliff=p)
            if result['reached_home']:
                success_count += 1

        simulated_prob = success_count / simulations
        print(f"{p:<10.1f} {theoretical_prob:<10.3f} {simulated_prob:<10.3f} {simulations:<8}")

    # 3. 收斂性分析
    print("\n3. 隨機漫步收斂性分析")
    print("-" * 30)
    steps_list = [10, 50, 100, 500, 1000]
    convergence = model.random_walk_convergence_test(steps_list, 200)

    print("步數增加時的標準差收斂:")
    print(f"{'步數':<8} {'實際標準差':<12} {'理論標準差':<12} {'標準化':<10}")
    print("-" * 45)

    for steps, data in convergence['convergence_data'].items():
        print(f"{steps:<8} {data['std_displacement']:<12.3f} "
              f"{data['theoretical_std']:<12.3f} {data['normalized_std']:<10.3f}")

    # 4. 不同漂移率的影響
    print("\n4. 漂移率對隨機漫步的影響")
    print("-" * 30)
    drift_rates = [-0.1, 0.0, 0.1, 0.2, 0.5]

    print("不同漂移率下的最終位置:")
    print(f"{'漂移率':<8} {'平均位置':<10} {'標準差':<10}")
    print("-" * 30)

    for drift in drift_rates:
        positions = []
        for _ in range(100):
            walk = model.biased_random_walk(100, drift, 1.0)
            positions.append(walk['final_position'])

        mean_pos = sum(positions) / len(positions)
        variance = sum((x - mean_pos)**2 for x in positions) / len(positions)
        std_pos = math.sqrt(variance)

        print(f"{drift:<8.1f} {mean_pos:<10.3f} {std_pos:<10.3f}")

    # 5. 首次通過時間分析
    print("\n5. 首次通過時間分析")
    print("-" * 30)
    targets = [5, 10, 20, 50]

    print("首次到達不同目標的時間統計:")
    print(f"{'目標':<6} {'平均時間':<10} {'成功率':<10}")
    print("-" * 28)

    for target in targets:
        times = []
        successes = 0
        simulations = 200

        for _ in range(simulations):
            result = model.first_passage_time_analysis(target, 500)
            if result['crossed']:
                times.append(result['first_passage_time'])
                successes += 1

        if times:
            avg_time = sum(times) / len(times)
        else:
            avg_time = 0

        success_rate = successes / simulations

        print(f"{target:<6} {avg_time:<10.1f} {success_rate:<10.2%}")

    print("\n演示完成！")


if __name__ == "__main__":
    main()