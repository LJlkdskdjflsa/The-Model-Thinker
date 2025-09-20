"""
孔多塞陪審團定理 (Condorcet Jury Theorem) 實作與驗證

定理內容：
如果每個陪審員做出正確判斷的概率 p > 0.5，
那麼隨著陪審團人數增加，多數決做出正確判斷的概率趨近於1。
反之，如果 p < 0.5，則集體錯誤的概率趨近於1。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from typing import List, Tuple, Optional


class CondorcetJuryTheorem:
    """
    孔多塞陪審團定理的實作與驗證

    這個定理展示了群體智慧的數學基礎：
    當個體判斷力高於隨機（p > 0.5）時，群體決策的準確性隨規模增長
    """

    def __init__(self, individual_accuracy: float = 0.6):
        """
        初始化模型

        Args:
            individual_accuracy: 每個陪審員做出正確判斷的概率 (0 < p < 1)
        """
        if not 0 < individual_accuracy < 1:
            raise ValueError("個體準確率必須在0和1之間")

        self.p = individual_accuracy

    def majority_accuracy(self, n: int) -> float:
        """
        計算n人陪審團多數決的正確概率

        使用二項分佈計算至少(n+1)/2人正確的概率

        Args:
            n: 陪審團人數（必須為奇數以避免平局）

        Returns:
            多數決正確的概率
        """
        if n % 2 == 0:
            raise ValueError("陪審團人數必須為奇數以避免平局")

        # 需要至少 (n+1)/2 人正確才能做出正確決定
        min_correct = (n + 1) // 2

        # 計算概率：P(X >= min_correct)，其中X ~ Binomial(n, p)
        prob = 0
        for k in range(min_correct, n + 1):
            prob += comb(n, k, exact=True) * (self.p ** k) * ((1 - self.p) ** (n - k))

        return prob

    def majority_accuracy_vectorized(self, jury_sizes: np.ndarray) -> np.ndarray:
        """
        向量化計算多個陪審團規模的正確概率

        Args:
            jury_sizes: 陪審團規模數組

        Returns:
            對應的正確概率數組
        """
        probabilities = []
        for n in jury_sizes:
            if n % 2 == 0:  # 如果是偶數，使用n+1
                n = n + 1
            probabilities.append(self.majority_accuracy(n))

        return np.array(probabilities)

    def simulate_jury_decisions(self, n: int, n_trials: int = 10000) -> float:
        """
        通過蒙特卡洛模擬驗證理論計算

        Args:
            n: 陪審團人數
            n_trials: 模擬次數

        Returns:
            模擬得到的多數決正確概率
        """
        if n % 2 == 0:
            n = n + 1  # 確保奇數

        correct_decisions = 0

        for _ in range(n_trials):
            # 模擬每個陪審員的決定
            individual_votes = np.random.random(n) < self.p
            # 計算多數決
            if np.sum(individual_votes) > n / 2:
                correct_decisions += 1

        return correct_decisions / n_trials

    def plot_theorem(self, max_jury_size: int = 101,
                     accuracy_levels: Optional[List[float]] = None) -> None:
        """
        視覺化定理：展示不同準確率下，陪審團規模對集體決策準確性的影響

        Args:
            max_jury_size: 最大陪審團規模
            accuracy_levels: 要比較的準確率水平列表
        """
        if accuracy_levels is None:
            accuracy_levels = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        jury_sizes = np.arange(1, max_jury_size + 1, 2)  # 只使用奇數

        plt.figure(figsize=(12, 8))

        for p in accuracy_levels:
            model = CondorcetJuryTheorem(p)
            accuracies = model.majority_accuracy_vectorized(jury_sizes)

            if p > 0.5:
                linestyle = '-'
                marker = 'o'
            elif p == 0.5:
                linestyle = '--'
                marker = 's'
            else:
                linestyle = ':'
                marker = '^'

            plt.plot(jury_sizes, accuracies,
                    label=f'p = {p}',
                    linestyle=linestyle,
                    marker=marker,
                    markevery=10,
                    markersize=4,
                    linewidth=2)

        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='隨機猜測')
        plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.3)

        plt.xlabel('陪審團人數', fontsize=12)
        plt.ylabel('集體決策正確概率', fontsize=12)
        plt.title('孔多塞陪審團定理：群體智慧的數學證明', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim(1, max_jury_size)
        plt.ylim(0, 1.05)

        # 添加註解
        plt.annotate('p > 0.5: 準確率隨規模提升',
                    xy=(50, 0.9), xytext=(60, 0.85),
                    arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                    fontsize=10, color='green')

        plt.annotate('p < 0.5: 準確率隨規模下降',
                    xy=(50, 0.1), xytext=(60, 0.15),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                    fontsize=10, color='red')

        plt.tight_layout()
        plt.show()

    def plot_simulation_vs_theory(self, jury_sizes: List[int] = None,
                                 n_trials: int = 1000) -> None:
        """
        比較理論計算和模擬結果

        Args:
            jury_sizes: 要測試的陪審團規模列表
            n_trials: 每個規模的模擬次數
        """
        if jury_sizes is None:
            jury_sizes = [3, 7, 11, 21, 31, 51, 71, 101]

        theoretical = []
        simulated = []

        print(f"比較理論與模擬（個體準確率 p = {self.p}）")
        print("-" * 50)
        print(f"{'陪審團規模':<12} {'理論值':<12} {'模擬值':<12} {'差異':<12}")
        print("-" * 50)

        for n in jury_sizes:
            if n % 2 == 0:
                n = n + 1

            theory = self.majority_accuracy(n)
            sim = self.simulate_jury_decisions(n, n_trials)

            theoretical.append(theory)
            simulated.append(sim)

            diff = abs(theory - sim)
            print(f"{n:<12} {theory:<12.4f} {sim:<12.4f} {diff:<12.4f}")

        # 繪製比較圖
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 左圖：理論vs模擬
        axes[0].plot(jury_sizes, theoretical, 'b-o', label='理論計算', linewidth=2, markersize=8)
        axes[0].plot(jury_sizes, simulated, 'r--s', label=f'模擬 (n={n_trials})', linewidth=2, markersize=6)
        axes[0].set_xlabel('陪審團規模', fontsize=11)
        axes[0].set_ylabel('正確概率', fontsize=11)
        axes[0].set_title(f'理論 vs 模擬 (p={self.p})', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # 右圖：誤差分析
        errors = np.abs(np.array(theoretical) - np.array(simulated))
        axes[1].bar(range(len(jury_sizes)), errors, color='orange', alpha=0.7)
        axes[1].set_xlabel('陪審團規模索引', fontsize=11)
        axes[1].set_ylabel('絕對誤差', fontsize=11)
        axes[1].set_title('理論與模擬的誤差', fontsize=12, fontweight='bold')
        axes[1].set_xticks(range(len(jury_sizes)))
        axes[1].set_xticklabels(jury_sizes, rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.show()

    def critical_mass_analysis(self, target_accuracy: float = 0.95) -> int:
        """
        計算達到目標準確率所需的最小陪審團規模

        Args:
            target_accuracy: 目標準確率

        Returns:
            所需的最小陪審團規模
        """
        if self.p <= 0.5:
            print(f"警告：個體準確率 p={self.p} <= 0.5，無法達到高準確率")
            return None

        for n in range(1, 1001, 2):  # 只考慮奇數，最多到1001
            if self.majority_accuracy(n) >= target_accuracy:
                return n

        return None  # 如果1001人還不夠

    def plot_critical_mass(self, accuracy_range: Tuple[float, float] = (0.51, 0.9),
                          target_accuracies: List[float] = None) -> None:
        """
        視覺化不同個體準確率下，達到目標準確率所需的陪審團規模

        Args:
            accuracy_range: 個體準確率範圍
            target_accuracies: 目標準確率列表
        """
        if target_accuracies is None:
            target_accuracies = [0.9, 0.95, 0.99]

        p_values = np.linspace(accuracy_range[0], accuracy_range[1], 20)

        plt.figure(figsize=(12, 8))

        for target in target_accuracies:
            required_sizes = []

            for p in p_values:
                model = CondorcetJuryTheorem(p)
                size = model.critical_mass_analysis(target)
                required_sizes.append(size if size else 1000)

            plt.plot(p_values, required_sizes, 'o-',
                    label=f'目標準確率 = {target}',
                    linewidth=2, markersize=6)

        plt.xlabel('個體準確率 (p)', fontsize=12)
        plt.ylabel('所需陪審團規模', fontsize=12)
        plt.title('達到目標準確率所需的最小陪審團規模', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        # 添加重要閾值
        plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='p = 0.5 (臨界點)')

        plt.tight_layout()
        plt.show()


def demonstrate_condorcet_theorem():
    """
    展示孔多塞陪審團定理的完整分析
    """
    print("=" * 60)
    print("孔多塞陪審團定理 - 完整驗證與分析")
    print("=" * 60)

    # 1. 基本計算
    print("\n1. 基本定理驗證")
    print("-" * 40)

    model = CondorcetJuryTheorem(individual_accuracy=0.7)

    test_sizes = [1, 3, 5, 11, 21, 51, 101]
    print(f"個體準確率: p = {model.p}")
    print(f"{'陪審團規模':<12} {'集體準確率':<12}")
    print("-" * 24)

    for n in test_sizes:
        if n % 2 == 0:
            continue
        accuracy = model.majority_accuracy(n)
        print(f"{n:<12} {accuracy:<12.6f}")

    # 2. 臨界規模分析
    print("\n2. 臨界規模分析")
    print("-" * 40)

    for p in [0.55, 0.6, 0.7, 0.8]:
        model = CondorcetJuryTheorem(p)
        size_90 = model.critical_mass_analysis(0.90)
        size_95 = model.critical_mass_analysis(0.95)
        size_99 = model.critical_mass_analysis(0.99)

        print(f"p = {p}:")
        print(f"  90% 準確率需要: {size_90} 人")
        print(f"  95% 準確率需要: {size_95} 人")
        print(f"  99% 準確率需要: {size_99} 人")

    # 3. 反例：當 p < 0.5
    print("\n3. 反例展示 (p < 0.5)")
    print("-" * 40)

    model_bad = CondorcetJuryTheorem(individual_accuracy=0.4)
    print(f"當個體準確率 p = {model_bad.p} < 0.5 時：")

    for n in [3, 11, 51]:
        accuracy = model_bad.majority_accuracy(n)
        print(f"  {n} 人陪審團準確率: {accuracy:.6f}")

    print("\n觀察：準確率隨人數增加而下降！")

    return model


if __name__ == "__main__":
    # 運行完整演示
    model = demonstrate_condorcet_theorem()

    # 生成視覺化
    print("\n生成視覺化圖表...")

    # 圖1：定理主要視覺化
    model.plot_theorem(max_jury_size=101)

    # 圖2：理論vs模擬
    model_sim = CondorcetJuryTheorem(individual_accuracy=0.65)
    model_sim.plot_simulation_vs_theory()

    # 圖3：臨界規模分析
    model.plot_critical_mass()