"""
閾值模型的簡單實現
Threshold Models - Simple Implementation

基於Granovetter閾值理論的集體行為模型

Author: Claude Code Assistant
License: MIT
"""

import math
import random
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
from collections import defaultdict


class ThresholdModelSimple:
    """
    閾值模型的簡單實現類

    實現Granovetter的閾值理論，包括：
    - 基本閾值模型
    - 創新擴散模型
    - 臨界質量分析
    - 級聯失效模型
    - 雙閾值模型
    """

    def __init__(self, seed: Optional[int] = None):
        """
        初始化閾值模型

        Args:
            seed: 隨機數種子，用於重現結果
        """
        if seed is not None:
            random.seed(seed)
        self.simulation_history = []

    def granovetter_threshold_model(self, thresholds: List[float],
                                  initial_adopters: int = 1) -> Dict[str, Any]:
        """
        Granovetter閾值模型的經典實現

        Args:
            thresholds: 各個體的閾值列表（0到1之間）
            initial_adopters: 初始採用者數量

        Returns:
            dict: 閾值模型結果
        """
        n = len(thresholds)
        if initial_adopters > n:
            raise ValueError("初始採用者數量不能超過總人數")

        # 按閾值排序，便於分析
        sorted_thresholds = sorted(thresholds)

        # 記錄擴散過程
        adoption_history = [initial_adopters]
        current_adopters = initial_adopters

        # 模擬擴散過程
        changed = True
        while changed:
            changed = False
            new_adopters = current_adopters

            # 檢查每個尚未採用的個體
            for threshold in sorted_thresholds:
                adoption_rate = current_adopters / n
                if adoption_rate >= threshold and current_adopters < n:
                    new_adopters += 1
                    changed = True
                else:
                    break  # 由於已排序，後面的個體閾值更高

            if new_adopters > n:
                new_adopters = n
                changed = False

            current_adopters = new_adopters
            adoption_history.append(current_adopters)

            if current_adopters == n:
                break

        final_adoption_rate = current_adopters / n

        return {
            'model_type': 'granovetter',
            'total_population': n,
            'initial_adopters': initial_adopters,
            'final_adopters': current_adopters,
            'final_adoption_rate': final_adoption_rate,
            'adoption_history': adoption_history,
            'thresholds': thresholds,
            'sorted_thresholds': sorted_thresholds,
            'complete_adoption': current_adopters == n,
            'steps_to_equilibrium': len(adoption_history) - 1
        }

    def critical_mass_analysis(self, thresholds: List[float],
                             max_initial: Optional[int] = None) -> Dict[str, Any]:
        """
        臨界質量分析：找出達到完全採用所需的最小初始採用者

        Args:
            thresholds: 個體閾值列表
            max_initial: 最大初始採用者數量（默認為人口的一半）

        Returns:
            dict: 臨界質量分析結果
        """
        n = len(thresholds)
        if max_initial is None:
            max_initial = n // 2

        critical_mass = None
        results = {}

        for initial in range(1, max_initial + 1):
            result = self.granovetter_threshold_model(thresholds, initial)
            results[initial] = result

            if result['complete_adoption'] and critical_mass is None:
                critical_mass = initial

        return {
            'critical_mass': critical_mass,
            'population_size': n,
            'max_tested': max_initial,
            'results_by_initial': results,
            'critical_mass_ratio': critical_mass / n if critical_mass else None
        }

    def innovation_diffusion_model(self, innovators_ratio: float = 0.025,
                                 early_adopters_ratio: float = 0.135,
                                 early_majority_ratio: float = 0.34,
                                 late_majority_ratio: float = 0.34,
                                 laggards_ratio: float = 0.16,
                                 population_size: int = 1000) -> Dict[str, Any]:
        """
        Rogers創新擴散模型（基於閾值）

        Args:
            innovators_ratio: 創新者比例
            early_adopters_ratio: 早期採用者比例
            early_majority_ratio: 早期大眾比例
            late_majority_ratio: 晚期大眾比例
            laggards_ratio: 落後者比例
            population_size: 人口規模

        Returns:
            dict: 創新擴散結果
        """
        # 確保比例總和為1
        total_ratio = (innovators_ratio + early_adopters_ratio +
                      early_majority_ratio + late_majority_ratio + laggards_ratio)

        if abs(total_ratio - 1.0) > 1e-10:
            # 標準化比例
            innovators_ratio /= total_ratio
            early_adopters_ratio /= total_ratio
            early_majority_ratio /= total_ratio
            late_majority_ratio /= total_ratio
            laggards_ratio /= total_ratio

        # 生成閾值分布
        thresholds = []

        # 創新者：閾值接近0
        for _ in range(int(population_size * innovators_ratio)):
            thresholds.append(random.uniform(0, 0.05))

        # 早期採用者：低閾值
        for _ in range(int(population_size * early_adopters_ratio)):
            thresholds.append(random.uniform(0.05, 0.15))

        # 早期大眾：中等閾值
        for _ in range(int(population_size * early_majority_ratio)):
            thresholds.append(random.uniform(0.15, 0.5))

        # 晚期大眾：較高閾值
        for _ in range(int(population_size * late_majority_ratio)):
            thresholds.append(random.uniform(0.5, 0.8))

        # 落後者：高閾值
        remaining = population_size - len(thresholds)
        for _ in range(remaining):
            thresholds.append(random.uniform(0.8, 1.0))

        # 執行擴散模擬
        initial_adopters = int(population_size * innovators_ratio)
        result = self.granovetter_threshold_model(thresholds, initial_adopters)

        return {
            'model_type': 'rogers_diffusion',
            'population_breakdown': {
                'innovators': int(population_size * innovators_ratio),
                'early_adopters': int(population_size * early_adopters_ratio),
                'early_majority': int(population_size * early_majority_ratio),
                'late_majority': int(population_size * late_majority_ratio),
                'laggards': remaining
            },
            'diffusion_result': result,
            'thresholds': thresholds
        }

    def cascade_failure_model(self, network_structure: List[List[int]],
                             failure_thresholds: List[float],
                             initial_failures: List[int]) -> Dict[str, Any]:
        """
        級聯失效模型

        Args:
            network_structure: 網絡結構（鄰接列表）
            failure_thresholds: 各節點的失效閾值
            initial_failures: 初始失效節點列表

        Returns:
            dict: 級聯失效結果
        """
        n = len(network_structure)
        failed_nodes = set(initial_failures)
        failure_history = [len(failed_nodes)]

        changed = True
        while changed:
            changed = False
            new_failures = set()

            for node in range(n):
                if node in failed_nodes:
                    continue

                # 計算鄰居中失效的比例
                neighbors = network_structure[node]
                if not neighbors:  # 沒有鄰居的節點不會因級聯而失效
                    continue

                failed_neighbors = sum(1 for neighbor in neighbors if neighbor in failed_nodes)
                failure_ratio = failed_neighbors / len(neighbors)

                # 檢查是否超過閾值
                if failure_ratio >= failure_thresholds[node]:
                    new_failures.add(node)
                    changed = True

            failed_nodes.update(new_failures)
            failure_history.append(len(failed_nodes))

        survival_rate = (n - len(failed_nodes)) / n

        return {
            'model_type': 'cascade_failure',
            'total_nodes': n,
            'initial_failures': len(initial_failures),
            'final_failures': len(failed_nodes),
            'survival_rate': survival_rate,
            'failed_nodes': list(failed_nodes),
            'failure_history': failure_history,
            'cascade_steps': len(failure_history) - 1
        }

    def bi_threshold_model(self, adoption_thresholds: List[float],
                         abandonment_thresholds: List[float],
                         initial_adopters: int = 1,
                         max_steps: int = 100) -> Dict[str, Any]:
        """
        雙閾值模型：既有採用閾值，也有放棄閾值

        Args:
            adoption_thresholds: 採用閾值列表
            abandonment_thresholds: 放棄閾值列表
            initial_adopters: 初始採用者數量
            max_steps: 最大模擬步數

        Returns:
            dict: 雙閾值模型結果
        """
        n = len(adoption_thresholds)
        if len(abandonment_thresholds) != n:
            raise ValueError("採用閾值和放棄閾值列表長度必須相同")

        # 初始狀態：前initial_adopters個個體為採用者
        adopters = set(range(initial_adopters))
        adoption_history = [len(adopters)]

        for step in range(max_steps):
            previous_adopters = len(adopters)
            adoption_rate = len(adopters) / n

            new_adopters = set(adopters)

            # 檢查新採用
            for i in range(n):
                if i not in adopters and adoption_rate >= adoption_thresholds[i]:
                    new_adopters.add(i)

            # 檢查放棄
            for i in list(adopters):
                if adoption_rate >= abandonment_thresholds[i]:
                    new_adopters.discard(i)

            adopters = new_adopters
            adoption_history.append(len(adopters))

            # 檢查是否達到平衡
            if len(adopters) == previous_adopters:
                break

        final_adoption_rate = len(adopters) / n

        return {
            'model_type': 'bi_threshold',
            'total_population': n,
            'initial_adopters': initial_adopters,
            'final_adopters': len(adopters),
            'final_adoption_rate': final_adoption_rate,
            'adoption_history': adoption_history,
            'equilibrium_reached': step < max_steps - 1,
            'steps_to_equilibrium': step + 1 if step < max_steps - 1 else max_steps
        }

    def threshold_distribution_analysis(self, distribution_type: str,
                                      population_size: int = 1000,
                                      **params) -> Dict[str, Any]:
        """
        不同閾值分布的影響分析

        Args:
            distribution_type: 分布類型（'uniform', 'normal', 'exponential', 'bimodal'）
            population_size: 人口大小
            **params: 分布參數

        Returns:
            dict: 分布分析結果
        """
        thresholds = []

        if distribution_type == 'uniform':
            low = params.get('low', 0.0)
            high = params.get('high', 1.0)
            thresholds = [random.uniform(low, high) for _ in range(population_size)]

        elif distribution_type == 'normal':
            mean = params.get('mean', 0.5)
            std = params.get('std', 0.2)
            thresholds = [max(0, min(1, random.gauss(mean, std))) for _ in range(population_size)]

        elif distribution_type == 'exponential':
            lambd = params.get('lambda', 2.0)
            thresholds = [min(1.0, random.expovariate(lambd)) for _ in range(population_size)]

        elif distribution_type == 'bimodal':
            low_mean = params.get('low_mean', 0.2)
            high_mean = params.get('high_mean', 0.8)
            std = params.get('std', 0.1)
            mix_ratio = params.get('mix_ratio', 0.5)

            for _ in range(population_size):
                if random.random() < mix_ratio:
                    threshold = max(0, min(1, random.gauss(low_mean, std)))
                else:
                    threshold = max(0, min(1, random.gauss(high_mean, std)))
                thresholds.append(threshold)

        else:
            raise ValueError(f"未知的分布類型: {distribution_type}")

        # 執行閾值模型
        result = self.granovetter_threshold_model(thresholds, 1)

        # 計算分布統計
        mean_threshold = sum(thresholds) / len(thresholds)
        variance = sum((t - mean_threshold)**2 for t in thresholds) / len(thresholds)
        std_threshold = math.sqrt(variance)

        return {
            'distribution_type': distribution_type,
            'distribution_params': params,
            'threshold_statistics': {
                'mean': mean_threshold,
                'std': std_threshold,
                'min': min(thresholds),
                'max': max(thresholds)
            },
            'simulation_result': result,
            'thresholds': thresholds
        }

    def tipping_point_analysis(self, thresholds: List[float],
                             test_range: Tuple[int, int] = (1, 50)) -> Dict[str, Any]:
        """
        翻轉點分析：找出導致大規模採用的臨界點

        Args:
            thresholds: 閾值列表
            test_range: 測試的初始採用者範圍

        Returns:
            dict: 翻轉點分析結果
        """
        n = len(thresholds)
        start, end = test_range
        end = min(end, n)

        results = []
        tipping_point = None

        for initial in range(start, end + 1):
            result = self.granovetter_threshold_model(thresholds, initial)
            adoption_rate = result['final_adoption_rate']

            results.append({
                'initial_adopters': initial,
                'final_adoption_rate': adoption_rate,
                'complete_adoption': result['complete_adoption']
            })

            # 檢查是否找到翻轉點（定義為達到80%以上採用率）
            if tipping_point is None and adoption_rate >= 0.8:
                tipping_point = initial

        return {
            'tipping_point': tipping_point,
            'tipping_point_ratio': tipping_point / n if tipping_point else None,
            'results': results,
            'population_size': n
        }

    def print_threshold_analysis(self, scenario_name: str = "基本閾值模型"):
        """
        打印閾值模型的詳細分析

        Args:
            scenario_name: 場景名稱
        """
        print("=" * 60)
        print(f"閾值模型分析: {scenario_name}")
        print("=" * 60)

        # 1. 基本Granovetter模型
        print("1. Granovetter閾值模型")
        print("-" * 40)

        # 生成隨機閾值
        thresholds = [random.uniform(0, 1) for _ in range(100)]
        result = self.granovetter_threshold_model(thresholds, 5)

        print(f"人口規模: {result['total_population']}")
        print(f"初始採用者: {result['initial_adopters']}")
        print(f"最終採用者: {result['final_adopters']}")
        print(f"最終採用率: {result['final_adoption_rate']:.2%}")
        print(f"達到完全採用: {'是' if result['complete_adoption'] else '否'}")
        print(f"平衡步數: {result['steps_to_equilibrium']}")
        print()

        # 2. 臨界質量分析
        print("2. 臨界質量分析")
        print("-" * 40)
        critical_mass = self.critical_mass_analysis(thresholds[:50], 25)

        print(f"臨界質量: {critical_mass['critical_mass']}")
        print(f"臨界質量比例: {critical_mass['critical_mass_ratio']:.2%}" if critical_mass['critical_mass'] else "未找到臨界質量")
        print()

        # 3. 創新擴散模型
        print("3. Rogers創新擴散模型")
        print("-" * 40)
        diffusion = self.innovation_diffusion_model(population_size=200)

        breakdown = diffusion['population_breakdown']
        diffusion_result = diffusion['diffusion_result']

        print("人口分布:")
        print(f"  創新者: {breakdown['innovators']}")
        print(f"  早期採用者: {breakdown['early_adopters']}")
        print(f"  早期大眾: {breakdown['early_majority']}")
        print(f"  晚期大眾: {breakdown['late_majority']}")
        print(f"  落後者: {breakdown['laggards']}")

        print(f"\n擴散結果:")
        print(f"  最終採用率: {diffusion_result['final_adoption_rate']:.2%}")
        print(f"  擴散步數: {diffusion_result['steps_to_equilibrium']}")
        print()

        # 4. 雙閾值模型
        print("4. 雙閾值模型")
        print("-" * 40)
        adoption_thresholds = [random.uniform(0, 0.5) for _ in range(50)]
        abandonment_thresholds = [random.uniform(0.6, 1.0) for _ in range(50)]

        bi_result = self.bi_threshold_model(adoption_thresholds, abandonment_thresholds, 3)

        print(f"最終採用率: {bi_result['final_adoption_rate']:.2%}")
        print(f"達到平衡: {'是' if bi_result['equilibrium_reached'] else '否'}")
        print(f"平衡步數: {bi_result['steps_to_equilibrium']}")
        print()

        # 5. 翻轉點分析
        print("5. 翻轉點分析")
        print("-" * 40)
        tipping = self.tipping_point_analysis(thresholds[:30], (1, 15))

        print(f"翻轉點: {tipping['tipping_point']}")
        print(f"翻轉點比例: {tipping['tipping_point_ratio']:.2%}" if tipping['tipping_point'] else "未找到翻轉點")

        print("\n分析完成！")


def main():
    """演示閾值模型的主要功能"""
    print("閾值模型演示")
    print("=" * 50)

    # 創建實例
    model = ThresholdModelSimple(seed=42)

    # 1. 基本演示
    print("\n1. 基本閾值模型分析")
    print("-" * 30)
    model.print_threshold_analysis("綜合閾值模型")

    # 2. 不同閾值分布的比較
    print("\n2. 不同閾值分布的影響")
    print("-" * 30)

    distributions = [
        ('uniform', {'low': 0.0, 'high': 1.0}),
        ('normal', {'mean': 0.5, 'std': 0.2}),
        ('bimodal', {'low_mean': 0.2, 'high_mean': 0.8, 'std': 0.1, 'mix_ratio': 0.5})
    ]

    print(f"{'分布類型':<12} {'平均閾值':<10} {'最終採用率':<12} {'完全採用':<8}")
    print("-" * 50)

    for dist_type, params in distributions:
        analysis = model.threshold_distribution_analysis(dist_type, 200, **params)
        stats = analysis['threshold_statistics']
        result = analysis['simulation_result']

        complete = "是" if result['complete_adoption'] else "否"
        print(f"{dist_type:<12} {stats['mean']:<10.3f} {result['final_adoption_rate']:<12.2%} {complete:<8}")

    # 3. 臨界質量敏感性分析
    print("\n3. 臨界質量敏感性分析")
    print("-" * 30)

    population_sizes = [50, 100, 200, 500]
    print(f"{'人口規模':<10} {'臨界質量':<10} {'臨界比例':<10}")
    print("-" * 32)

    for pop_size in population_sizes:
        thresholds = [random.uniform(0, 1) for _ in range(pop_size)]
        critical_analysis = model.critical_mass_analysis(thresholds, pop_size // 4)

        critical_mass = critical_analysis['critical_mass']
        ratio = critical_analysis['critical_mass_ratio']

        if critical_mass:
            print(f"{pop_size:<10} {critical_mass:<10} {ratio:<10.2%}")
        else:
            print(f"{pop_size:<10} {'無':<10} {'N/A':<10}")

    # 4. 級聯失效演示
    print("\n4. 級聯失效模型")
    print("-" * 30)

    # 創建簡單的環形網絡
    network_size = 20
    network = [[i-1, i+1] for i in range(network_size)]
    network[0] = [network_size-1, 1]  # 環形連接
    network[network_size-1] = [network_size-2, 0]

    failure_thresholds = [0.5] * network_size  # 所有節點閾值為0.5
    initial_failures = [0, 10]  # 兩個初始失效節點

    cascade_result = model.cascade_failure_model(network, failure_thresholds, initial_failures)

    print(f"網絡規模: {cascade_result['total_nodes']}")
    print(f"初始失效: {cascade_result['initial_failures']}")
    print(f"最終失效: {cascade_result['final_failures']}")
    print(f"存活率: {cascade_result['survival_rate']:.2%}")
    print(f"級聯步數: {cascade_result['cascade_steps']}")

    # 5. 創新擴散的參數敏感性
    print("\n5. 創新擴散參數敏感性")
    print("-" * 30)

    innovator_ratios = [0.01, 0.025, 0.05, 0.1]
    print(f"{'創新者比例':<12} {'最終採用率':<12} {'擴散成功':<10}")
    print("-" * 38)

    for ratio in innovator_ratios:
        diffusion = model.innovation_diffusion_model(
            innovators_ratio=ratio,
            population_size=500
        )

        final_rate = diffusion['diffusion_result']['final_adoption_rate']
        success = "是" if final_rate > 0.8 else "否"

        print(f"{ratio:<12.1%} {final_rate:<12.2%} {success:<10}")

    print("\n演示完成！")


if __name__ == "__main__":
    main()