"""
孔多塞陪審團定理 - 互動式演示

這個腳本提供了定理的互動式探索和實際應用案例
"""

from condorcet_jury import CondorcetJuryTheorem
import numpy as np
import matplotlib.pyplot as plt


def real_world_examples():
    """
    展示現實世界的應用案例
    """
    print("=" * 60)
    print("孔多塞陪審團定理 - 現實應用案例")
    print("=" * 60)

    # 案例1：陪審團制度
    print("\n案例1：法律陪審團")
    print("-" * 40)
    print("假設每位陪審員正確判斷案件的概率為 0.7")

    jury_model = CondorcetJuryTheorem(individual_accuracy=0.7)

    jury_sizes = {
        "小陪審團": 12,  # 使用13人（奇數）
        "大陪審團": 23
    }

    for name, size in jury_sizes.items():
        if size % 2 == 0:
            size = size + 1  # 轉為奇數
        accuracy = jury_model.majority_accuracy(size)
        print(f"{name} ({size}人): 正確判決概率 = {accuracy:.4f}")

    # 案例2：醫療診斷
    print("\n案例2：醫療會診")
    print("-" * 40)
    print("假設每位醫生正確診斷的概率為 0.75")

    medical_model = CondorcetJuryTheorem(individual_accuracy=0.75)

    consultation_sizes = [1, 3, 5, 7]
    print(f"{'醫生數量':<10} {'診斷準確率':<12} {'提升幅度':<12}")
    print("-" * 34)

    base_accuracy = medical_model.majority_accuracy(1)
    for n in consultation_sizes:
        if n % 2 == 0:
            n = n + 1
        accuracy = medical_model.majority_accuracy(n)
        improvement = (accuracy - base_accuracy) / base_accuracy * 100
        print(f"{n:<10} {accuracy:<12.4f} {improvement:>10.1f}%")

    # 案例3：群眾智慧 vs 專家
    print("\n案例3：群眾智慧 vs 專家")
    print("-" * 40)

    # 專家：高準確率但人數少
    expert_model = CondorcetJuryTheorem(individual_accuracy=0.85)
    expert_accuracy = expert_model.majority_accuracy(3)  # 3位專家

    # 群眾：較低準確率但人數多
    crowd_model = CondorcetJuryTheorem(individual_accuracy=0.6)
    crowd_accuracy = crowd_model.majority_accuracy(101)  # 101位普通人

    print(f"3位專家 (p=0.85): 準確率 = {expert_accuracy:.4f}")
    print(f"101位普通人 (p=0.6): 準確率 = {crowd_accuracy:.4f}")

    if crowd_accuracy > expert_accuracy:
        print("\n結論：大量普通人的集體智慧超過了少數專家！")
    else:
        print("\n結論：在這個案例中，專家團隊仍然更準確。")

    # 找出平衡點
    for n in range(5, 200, 2):
        if crowd_model.majority_accuracy(n) > expert_accuracy:
            print(f"需要至少 {n} 位普通人才能超越3位專家的準確率")
            break


def voting_system_analysis():
    """
    分析不同投票系統的可靠性
    """
    print("\n" + "=" * 60)
    print("投票系統可靠性分析")
    print("=" * 60)

    # 不同類型的投票者
    voter_types = {
        "高信息選民": 0.75,
        "一般選民": 0.60,
        "低信息選民": 0.55,
        "隨機投票": 0.50
    }

    population_sizes = [100, 1000, 10000]

    print(f"{'選民類型':<15} {'準確率':<10}", end="")
    for size in population_sizes:
        print(f"{str(size)+'人':<12}", end="")
    print()
    print("-" * 55)

    results = {}
    for voter_type, accuracy in voter_types.items():
        model = CondorcetJuryTheorem(individual_accuracy=accuracy)
        results[voter_type] = []

        print(f"{voter_type:<15} {accuracy:<10.2f}", end="")

        for size in population_sizes:
            # 使用最接近的奇數
            n = size + 1 if size % 2 == 0 else size
            collective_accuracy = model.majority_accuracy(min(n, 501))  # 限制計算規模
            results[voter_type].append(collective_accuracy)
            print(f"{collective_accuracy:<12.4f}", end="")

        print()


def interactive_exploration():
    """
    互動式參數探索
    """
    print("\n" + "=" * 60)
    print("互動式參數探索")
    print("=" * 60)

    while True:
        print("\n選擇探索選項：")
        print("1. 計算特定參數的陪審團準確率")
        print("2. 比較不同準確率水平")
        print("3. 找出達到目標準確率所需的規模")
        print("4. 退出")

        choice = input("\n請選擇 (1-4): ")

        if choice == '1':
            try:
                p = float(input("輸入個體準確率 (0-1): "))
                n = int(input("輸入陪審團規模 (奇數): "))

                if n % 2 == 0:
                    n = n + 1
                    print(f"調整為奇數: {n}")

                model = CondorcetJuryTheorem(individual_accuracy=p)
                accuracy = model.majority_accuracy(n)

                print(f"\n結果：")
                print(f"個體準確率: {p}")
                print(f"陪審團規模: {n}")
                print(f"集體決策準確率: {accuracy:.6f}")

                if p > 0.5 and n > 1:
                    individual = p
                    improvement = (accuracy - individual) / individual * 100
                    print(f"相對個體的提升: {improvement:.2f}%")

            except ValueError as e:
                print(f"輸入錯誤: {e}")

        elif choice == '2':
            try:
                p1 = float(input("輸入第一個準確率 (0-1): "))
                p2 = float(input("輸入第二個準確率 (0-1): "))
                max_n = int(input("最大陪審團規模: "))

                fig, ax = plt.subplots(figsize=(10, 6))

                jury_sizes = range(1, max_n + 1, 2)

                for p in [p1, p2]:
                    model = CondorcetJuryTheorem(individual_accuracy=p)
                    accuracies = [model.majority_accuracy(n) for n in jury_sizes]
                    ax.plot(jury_sizes, accuracies, label=f'p = {p}', linewidth=2)

                ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
                ax.set_xlabel('陪審團規模')
                ax.set_ylabel('準確率')
                ax.set_title('準確率比較')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.show()

            except ValueError as e:
                print(f"輸入錯誤: {e}")

        elif choice == '3':
            try:
                p = float(input("輸入個體準確率 (0.5-1): "))
                target = float(input("目標集體準確率 (0-1): "))

                model = CondorcetJuryTheorem(individual_accuracy=p)
                required_size = model.critical_mass_analysis(target)

                if required_size:
                    print(f"\n結果：")
                    print(f"達到 {target:.2%} 準確率需要至少 {required_size} 人的陪審團")
                else:
                    print(f"\n無法達到目標準確率 {target:.2%}")
                    if p <= 0.5:
                        print("原因：個體準確率必須大於 0.5")

            except ValueError as e:
                print(f"輸入錯誤: {e}")

        elif choice == '4':
            print("退出程式")
            break

        else:
            print("無效選擇，請重試")


def main():
    """
    主程式：運行所有演示
    """
    print("\n歡迎來到孔多塞陪審團定理演示程式！\n")

    # 1. 現實案例
    real_world_examples()

    # 2. 投票系統分析
    voting_system_analysis()

    # 3. 生成主要視覺化
    print("\n生成視覺化...")
    model = CondorcetJuryTheorem(individual_accuracy=0.65)
    model.plot_theorem()

    # 4. 互動探索（可選）
    response = input("\n是否進入互動探索模式？ (y/n): ")
    if response.lower() == 'y':
        interactive_exploration()


if __name__ == "__main__":
    main()