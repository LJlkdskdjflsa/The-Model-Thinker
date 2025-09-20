"""
簡單測試孔多塞陪審團定理（無需視覺化庫）
"""

from condorcet_simple import CondorcetJuryTheoremSimple as CondorcetJuryTheorem


def simple_test():
    """
    簡單測試定理的核心計算
    """
    print("=" * 60)
    print("孔多塞陪審團定理 - 簡單測試")
    print("=" * 60)

    # 測試1：p > 0.5 的情況
    print("\n測試1：當個體準確率 p > 0.5 時")
    print("-" * 40)

    model = CondorcetJuryTheorem(individual_accuracy=0.7)
    print(f"個體準確率: p = {model.p}")
    print(f"\n{'陪審團規模':<12} {'集體準確率':<15} {'相對提升':<15}")
    print("-" * 42)

    for n in [1, 3, 5, 11, 21, 51, 101]:
        accuracy = model.majority_accuracy(n)
        improvement = (accuracy - model.p) / model.p * 100
        print(f"{n:<12} {accuracy:<15.6f} {improvement:>13.2f}%")

    # 測試2：p < 0.5 的情況
    print("\n測試2：當個體準確率 p < 0.5 時")
    print("-" * 40)

    model2 = CondorcetJuryTheorem(individual_accuracy=0.4)
    print(f"個體準確率: p = {model2.p}")
    print(f"\n{'陪審團規模':<12} {'集體準確率':<15}")
    print("-" * 27)

    for n in [1, 3, 5, 11, 21]:
        accuracy = model2.majority_accuracy(n)
        print(f"{n:<12} {accuracy:<15.6f}")

    print("\n觀察：當 p < 0.5 時，集體準確率隨規模增大而降低！")

    # 測試3：臨界點 p = 0.5
    print("\n測試3：臨界點 p = 0.5")
    print("-" * 40)

    model3 = CondorcetJuryTheorem(individual_accuracy=0.5)
    print(f"個體準確率: p = {model3.p}")

    for n in [3, 11, 51]:
        accuracy = model3.majority_accuracy(n)
        print(f"{n} 人陪審團: {accuracy:.6f}")

    print("觀察：當 p = 0.5 時，集體準確率始終為 0.5")

    # 測試4：模擬驗證
    print("\n測試4：蒙特卡洛模擬驗證")
    print("-" * 40)

    model4 = CondorcetJuryTheorem(individual_accuracy=0.65)
    n = 11
    theoretical = model4.majority_accuracy(n)
    simulated = model4.simulate_jury_decisions(n, n_trials=10000)

    print(f"11人陪審團，個體準確率 p = 0.65")
    print(f"理論計算: {theoretical:.6f}")
    print(f"模擬結果: {simulated:.6f}")
    print(f"差異: {abs(theoretical - simulated):.6f}")

    # 測試5：找出達到目標準確率所需規模
    print("\n測試5：達到目標準確率所需的最小陪審團規模")
    print("-" * 40)

    for p in [0.55, 0.60, 0.70, 0.80]:
        model5 = CondorcetJuryTheorem(individual_accuracy=p)
        size_90 = model5.critical_mass_analysis(0.90)
        size_95 = model5.critical_mass_analysis(0.95)

        print(f"\n個體準確率 p = {p}:")
        print(f"  達到 90% 準確率需要: {size_90} 人")
        print(f"  達到 95% 準確率需要: {size_95} 人")


if __name__ == "__main__":
    simple_test()