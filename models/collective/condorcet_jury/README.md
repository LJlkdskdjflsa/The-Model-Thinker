# 孔多塞陪審團定理 (Condorcet Jury Theorem)

## 📚 理論介紹

孔多塞陪審團定理是由法國數學家和政治理論家馬奎斯·德·孔多塞（Marquis de Condorcet）在1785年提出的重要定理，它為群體決策的優越性提供了數學基礎。

### 核心內容

**定理陳述**：
- 如果每個陪審員做出正確判斷的概率 p > 0.5
- 那麼隨著陪審團人數 n 增加，多數決做出正確判斷的概率趨近於 1
- 反之，如果 p < 0.5，則集體錯誤的概率趨近於 1

### 數學公式

對於 n 人陪審團（n為奇數），多數決正確的概率為：

```
P(正確) = Σ(k=⌈n/2⌉ to n) C(n,k) × p^k × (1-p)^(n-k)
```

其中：
- C(n,k) 是組合數
- p 是個體正確率
- ⌈n/2⌉ 是最小多數

## 🚀 快速開始

### 運行簡單演示

```bash
# 運行核心計算演示（無需視覺化庫）
python condorcet_jury/condorcet_simple.py
```

### 代碼示例

```python
from condorcet_simple import CondorcetJuryTheoremSimple

# 創建模型實例
model = CondorcetJuryTheoremSimple(individual_accuracy=0.7)

# 計算11人陪審團的集體準確率
accuracy = model.majority_accuracy(11)
print(f"11人陪審團準確率: {accuracy:.4f}")  # 輸出: 0.9218

# 找出達到95%準確率所需的陪審團規模
required_size = model.critical_mass_analysis(0.95)
print(f"達到95%準確率需要: {required_size}人")  # 輸出: 17人
```

## 📊 實驗結果

### 實驗1：準確率隨規模變化

| 個體準確率 | 1人 | 3人 | 11人 | 51人 | 101人 |
|-----------|-----|-----|------|------|-------|
| p=0.4 | 0.40 | 0.35 | 0.25 | 0.08 | 0.02 |
| p=0.5 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |
| p=0.6 | 0.60 | 0.65 | 0.75 | 0.92 | 0.98 |
| p=0.7 | 0.70 | 0.78 | 0.92 | 0.99 | 1.00 |

### 實驗2：達到目標準確率所需規模

| 個體準確率 | 90%目標 | 95%目標 | 99%目標 |
|-----------|---------|---------|---------|
| p=0.55 | 109人 | 269人 | 1001+人 |
| p=0.60 | 29人 | 67人 | 267人 |
| p=0.70 | 9人 | 17人 | 39人 |
| p=0.80 | 5人 | 7人 | 11人 |

## 🔍 關鍵洞察

### 1. 臨界點的重要性
- **p > 0.5**：群體智慧放大效應
- **p = 0.5**：集體決策等同於隨機
- **p < 0.5**：群體愚蠢放大效應

### 2. 收斂速度
個體準確率越高，達到高準確率所需的陪審團規模越小：
- p=0.55 需要數百人
- p=0.80 只需要不到10人

### 3. 現實應用

**法律系統**：
- 陪審團制度的理論基礎
- 13人陪審團，p=0.75時，準確率達97.6%

**醫療診斷**：
- 會診制度的數學支撐
- 5位醫生會診，p=0.8時，準確率達94.2%

**民主投票**：
- 解釋了為何大規模民主投票通常有效
- 前提是選民判斷力優於隨機

## 🎯 實際應用案例

### 案例1：陪審團制度
```python
# 美國陪審團系統
jury = CondorcetJuryTheoremSimple(0.75)  # 假設陪審員75%正確
print(f"12人陪審團: {jury.majority_accuracy(13):.2%}")  # 97.57%
print(f"23人大陪審團: {jury.majority_accuracy(23):.2%}")  # 99.54%
```

### 案例2：醫療會診
```python
# 醫療診斷系統
medical = CondorcetJuryTheoremSimple(0.80)  # 醫生80%診斷正確
for n in [1, 3, 5, 7]:
    print(f"{n}位醫生: {medical.majority_accuracy(n):.2%}")
# 1位: 80.00%, 3位: 89.60%, 5位: 94.21%, 7位: 96.66%
```

### 案例3：群眾 vs 專家
```python
# 比較群眾智慧與專家意見
experts = CondorcetJuryTheoremSimple(0.85)  # 3位專家
crowd = CondorcetJuryTheoremSimple(0.60)    # 101位普通人

expert_acc = experts.majority_accuracy(3)    # 93.93%
crowd_acc = crowd.majority_accuracy(101)      # 97.93%
# 結論：大量普通人可能超越少數專家！
```

## ⚠️ 模型局限性

1. **獨立性假設**：現實中個體判斷往往相互影響
2. **同質性假設**：假設所有個體有相同的準確率
3. **二元決策**：僅適用於是非判斷，不適用於多選項
4. **信息完整性**：假設所有人基於相同信息做判斷
5. **理性假設**：忽略了情緒、偏見等非理性因素

## 📂 文件結構

```
condorcet_jury/
├── README.md              # 本文檔
├── condorcet_simple.py    # 核心實現（無依賴）
├── condorcet_jury.py      # 完整實現（含視覺化）
├── demo.py                # 互動式演示
└── test.py                # 測試腳本
```

## 🔗 延伸閱讀

- Condorcet, M. (1785). *Essai sur l'application de l'analyse à la probabilité des décisions rendues à la pluralité des voix*
- List, C., & Goodin, R. E. (2001). "Epistemic democracy: Generalizing the Condorcet jury theorem"
- Surowiecki, J. (2004). *The Wisdom of Crowds*
- Page, S. E. (2018). *The Model Thinker*, Chapter on Collective Decision Making

## 💡 思考題

1. 為什麼現代民主制度通常採用簡單多數決？
2. 如果陪審員之間會相互影響，定理還成立嗎？
3. 在什麼情況下，3位專家的判斷會優於100位普通人？
4. 如何將此定理應用於機器學習的集成方法？
5. 社交媒體的信息傳播如何影響群體決策的準確性？