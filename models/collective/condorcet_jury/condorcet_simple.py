"""
孔多塞陪審團定理 - 簡化版本（無視覺化依賴）

核心計算實作，不依賴matplotlib等視覺化庫
"""

import math
from typing import List, Optional


class CondorcetJuryTheoremSimple:
    """
    孔多塞陪審團定理的核心實作
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

    def binomial_coefficient(self, n: int, k: int) -> int:
        """計算二項式係數 C(n, k)"""
        return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

    def majority_accuracy(self, n: int) -> float:
        """
        計算n人陪審團多數決的正確概率

        Args:
            n: 陪審團人數（必須為奇數）

        Returns:
            多數決正確的概率
        """
        if n % 2 == 0:
            raise ValueError("陪審團人數必須為奇數以避免平局")

        min_correct = (n + 1) // 2
        prob = 0

        for k in range(min_correct, n + 1):
            prob += (
                self.binomial_coefficient(n, k)
                * (self.p**k)
                * ((1 - self.p) ** (n - k))
            )

        return prob

    def simulate_jury_decisions(self, n: int, n_trials: int = 10000) -> float:
        """
        通過蒙特卡洛模擬驗證理論計算

        Args:
            n: 陪審團人數
            n_trials: 模擬次數

        Returns:
            模擬得到的多數決正確概率
        """
        import random

        if n % 2 == 0:
            n = n + 1

        correct_decisions = 0

        for _ in range(n_trials):
            # 模擬每個陪審員的決定
            correct_votes = sum(1 for _ in range(n) if random.random() < self.p)
            # 判斷是否多數正確
            if correct_votes > n / 2:
                correct_decisions += 1

        return correct_decisions / n_trials

    def critical_mass_analysis(self, target_accuracy: float = 0.95) -> Optional[int]:
        """
        計算達到目標準確率所需的最小陪審團規模

        Args:
            target_accuracy: 目標準確率

        Returns:
            所需的最小陪審團規模
        """
        if self.p <= 0.5:
            return None

        for n in range(1, 1001, 2):  # 只考慮奇數
            if self.majority_accuracy(n) >= target_accuracy:
                return n

        return None


def demonstrate():
    """
    展示定理的核心概念
    """
    print("=" * 60)
    print("孔多塞陪審團定理驗證")
    print("=" * 60)

    # 案例1：p > 0.5
    print("\n【案例1】當個體準確率 p = 0.7 > 0.5")
    print("-" * 40)

    model1 = CondorcetJuryTheoremSimple(individual_accuracy=0.7)

    print(f"{'陪審團規模':<12} {'集體準確率':<15} {'提升效果':<15}")
    print("-" * 42)

    for n in [1, 3, 5, 11, 21, 51, 101]:
        accuracy = model1.majority_accuracy(n)
        improvement = ((accuracy - model1.p) / model1.p * 100) if n > 1 else 0
        print(f"{n:<12} {accuracy:<15.6f} {improvement:>13.2f}%")

    print("\n結論：隨著陪審團規模增加，集體準確率持續提高！")

    # 案例2：p < 0.5
    print("\n【案例2】當個體準確率 p = 0.4 < 0.5")
    print("-" * 40)

    model2 = CondorcetJuryTheoremSimple(individual_accuracy=0.4)

    print(f"{'陪審團規模':<12} {'集體準確率':<15}")
    print("-" * 27)

    for n in [1, 3, 5, 11, 21]:
        accuracy = model2.majority_accuracy(n)
        print(f"{n:<12} {accuracy:<15.6f}")

    print("\n結論：當p < 0.5時，規模越大反而準確率越低！")

    # 案例3：現實應用
    print("\n【案例3】現實應用分析")
    print("-" * 40)

    # 法庭陪審團
    print("\n法庭陪審團（假設個體準確率=0.75）:")
    jury_model = CondorcetJuryTheoremSimple(individual_accuracy=0.75)

    jury_sizes = {
        "小型陪審團": 13,  # 12人改為奇數
        "大型陪審團": 23,
    }

    for name, size in jury_sizes.items():
        accuracy = jury_model.majority_accuracy(size)
        print(f"  {name}({size}人): 準確率 = {accuracy:.4f}")

    # 醫療會診
    print("\n醫療會診（假設醫生準確率=0.8）:")
    medical_model = CondorcetJuryTheoremSimple(individual_accuracy=0.8)

    for n_doctors in [1, 3, 5]:
        accuracy = medical_model.majority_accuracy(n_doctors)
        print(f"  {n_doctors}位醫生會診: 準確率 = {accuracy:.4f}")

    # 案例4：臨界規模分析
    print("\n【案例4】達到目標準確率所需的陪審團規模")
    print("-" * 40)

    target = 0.95
    print(f"目標：達到 {target:.0%} 的準確率\n")

    for p in [0.55, 0.60, 0.70, 0.80]:
        model = CondorcetJuryTheoremSimple(individual_accuracy=p)
        size = model.critical_mass_analysis(target)
        if size:
            print(f"個體準確率 p={p}: 需要 {size} 人")
        else:
            print(f"個體準確率 p={p}: 無法達到目標")

    # 案例5：模擬驗證
    print("\n【案例5】蒙特卡洛模擬驗證")
    print("-" * 40)

    test_model = CondorcetJuryTheoremSimple(individual_accuracy=0.65)
    n = 11

    theoretical = test_model.majority_accuracy(n)
    simulated = test_model.simulate_jury_decisions(n, n_trials=10000)

    print(f"參數：11人陪審團，個體準確率=0.65")
    print(f"理論計算: {theoretical:.6f}")
    print(f"模擬結果: {simulated:.6f}")
    print(f"誤差: {abs(theoretical - simulated):.6f}")

    # 關鍵洞察
    print("\n" + "=" * 60)
    print("關鍵洞察")
    print("=" * 60)

    print("""
1. 定理的核心條件：個體準確率必須 > 0.5
2. 群體智慧的數學基礎：多數決可以放大個體優勢
3. 實際應用：陪審團制度、醫療會診、群眾投票
4. 侷限性：假設個體判斷獨立、忽略群體思維等因素
    """)


if __name__ == "__main__":
    demonstrate()
