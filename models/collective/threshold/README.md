# 閾值模型 (Threshold Models)

## 📚 理論介紹

閾值模型是由Mark Granovetter在1978年提出的重要理論，用於解釋集體行為中的臨界現象。該模型假設個體的決策取決於已經做出相同決策的其他人數量或比例，當這個數量超過個體的"閾值"時，個體就會採取行動。

### 核心內容

**理論陳述**：
- 每個個體都有一個閾值，表示促使其採取行動所需的其他人採取行動的最小比例
- 集體行為的結果取決於閾值分布，而不僅僅是平均閾值
- 微小的初始條件差異可能導致截然不同的最終結果

### 數學模型

**基本閾值模型**：
```
個體 i 在時間 t 採取行動，當且僅當：
A(t-1)/N ≥ τᵢ
```

**擴散動力學**：
```
A(t) = |{i : A(t-1)/N ≥ τᵢ}|
```

**平衡條件**：
```
A* = |{i : A*/N ≥ τᵢ}|
```

其中：
- A(t)：時間t的採用者數量
- N：總人口
- τᵢ：個體i的閾值
- A*：平衡狀態的採用者數量

**臨界質量**：使得完全採用成為可能的最小初始採用者數量

## 🚀 快速開始

### 運行簡單演示

```bash
# 運行核心計算演示
python threshold/threshold_simple.py
```

### 代碼示例

```python
from threshold_simple import ThresholdModelSimple

# 創建模型實例
model = ThresholdModelSimple(seed=42)

# Granovetter閾值模型
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
result = model.granovetter_threshold_model(thresholds, initial_adopters=2)

print(f"最終採用率: {result['final_adoption_rate']:.2%}")
print(f"完全採用: {result['complete_adoption']}")

# 臨界質量分析
critical_mass = model.critical_mass_analysis(thresholds)
print(f"臨界質量: {critical_mass['critical_mass']}")

# 創新擴散模型
diffusion = model.innovation_diffusion_model(population_size=1000)
print(f"創新擴散成功率: {diffusion['diffusion_result']['final_adoption_rate']:.2%}")

# 級聯失效
network = [[1, 2], [0, 2], [0, 1]]  # 三節點完全圖
failures = model.cascade_failure_model(network, [0.5, 0.5, 0.5], [0])
print(f"級聯失效後存活率: {failures['survival_rate']:.2%}")
```

## 📊 實驗結果

### 實驗1：基本閾值模型表現

| 指標 | 結果 |
|------|------|
| 人口規模 | 100 |
| 初始採用者 | 5 |
| 最終採用者 | 100 |
| 最終採用率 | 100.00% |
| 平衡步數 | 4 |

### 實驗2：不同閾值分布的影響

| 分布類型 | 平均閾值 | 最終採用率 | 完全採用 |
|----------|----------|------------|----------|
| 均勻分布 | 0.497 | 0.50% | 否 |
| 正態分布 | 0.478 | 0.50% | 否 |
| 雙峰分布 | 0.521 | 100.00% | 是 |

### 實驗3：臨界質量與人口規模

| 人口規模 | 臨界質量 | 臨界比例 |
|----------|----------|----------|
| 50 | 3 | 6.00% |
| 100 | 4 | 4.00% |
| 200 | 2 | 1.00% |
| 500 | 2 | 0.40% |

### 實驗4：Rogers創新擴散模型

| 群體 | 比例 | 閾值範圍 |
|------|------|----------|
| 創新者 | 2.5% | 0.00-0.05 |
| 早期採用者 | 13.5% | 0.05-0.15 |
| 早期大眾 | 34% | 0.15-0.50 |
| 晚期大眾 | 34% | 0.50-0.80 |
| 落後者 | 16% | 0.80-1.00 |

## 🔍 關鍵洞察

### 1. 翻轉點現象
**小變化，大影響**：
- 微小的初始條件差異可能導致完全不同的結果
- 臨界質量附近的系統極其敏感
- 解釋了社會現象中的突然爆發

### 2. 閾值分布的重要性
**分布形狀決定結果**：
- 平均閾值相同，分布不同，結果迥異
- 雙峰分布容易產生完全採用或完全拒絕
- 均勻分布傾向於產生部分採用的平衡

### 3. 級聯效應
**連鎖反應**：
- 網絡結構影響級聯失效的範圍
- 少數關鍵節點的失效可能導致系統崩潰
- 韌性設計需要考慮閾值分布

## 🎯 實際應用案例

### 案例1：社交媒體病毒傳播
```python
# 模擬社交媒體上的病毒內容傳播
viral_model = ThresholdModelSimple()

# 用戶的分享閾值（需要看到多少朋友分享才會自己分享）
sharing_thresholds = [random.uniform(0.1, 0.8) for _ in range(10000)]

# 從少數影響者開始
viral_result = viral_model.granovetter_threshold_model(sharing_thresholds, 50)

print(f"病毒傳播最終覆蓋率: {viral_result['final_adoption_rate']:.2%}")
print(f"傳播步數: {viral_result['steps_to_equilibrium']}")
```

### 案例2：新技術採用
```python
# 企業採用新技術的決策模型
tech_adoption = ThresholdModelSimple()

# 基於Rogers創新擴散理論的技術採用
tech_diffusion = tech_adoption.innovation_diffusion_model(
    innovators_ratio=0.025,     # 2.5%的技術先驅
    early_adopters_ratio=0.135, # 13.5%的早期採用者
    population_size=5000        # 5000家企業
)

adoption_rate = tech_diffusion['diffusion_result']['final_adoption_rate']
print(f"新技術最終採用率: {adoption_rate:.2%}")
```

### 案例3：金融系統風險傳染
```python
# 銀行間風險傳染模型
financial_contagion = ThresholdModelSimple()

# 構建銀行網絡（簡化）
bank_network = [
    [1, 2, 3],      # 銀行0連接到銀行1,2,3
    [0, 2, 4],      # 銀行1連接到銀行0,2,4
    [0, 1, 3, 4],   # 銀行2連接到銀行0,1,3,4
    [0, 2, 4],      # 銀行3連接到銀行0,2,4
    [1, 2, 3]       # 銀行4連接到銀行1,2,3
]

# 銀行的風險容忍度（破產閾值）
risk_thresholds = [0.3, 0.4, 0.2, 0.5, 0.3]

# 假設銀行0首先破產
contagion = financial_contagion.cascade_failure_model(
    bank_network, risk_thresholds, [0]
)

print(f"金融系統存活率: {contagion['survival_rate']:.2%}")
print(f"風險傳染步數: {contagion['cascade_steps']}")
```

## ⚠️ 模型局限性

1. **二元決策假設**：現實中決策往往是連續的或多選的
2. **同質性假設**：忽略了個體間的異質性和社會網絡結構
3. **靜態閾值**：假設閾值不隨時間或經驗變化
4. **完全信息**：假設個體能完美觀察他人的行為
5. **線性閾值**：忽略了非線性的社會影響機制

## 📈 模型擴展

### 1. 網絡閾值模型
```python
# 在複雜網絡上的閾值模型
def network_threshold_model(network, thresholds, initial_adopters):
    # 考慮網絡拓撲結構的影響
    # 個體只觀察直接鄰居的行為
    pass
```

### 2. 動態閾值模型
```python
# 閾值隨時間變化的模型
def dynamic_threshold_model(initial_thresholds, adaptation_rate):
    # 閾值根據過去經驗調整
    # 學習效應和適應性行為
    pass
```

### 3. 多狀態閾值模型
```python
# 多個狀態間轉換的閾值模型
def multi_state_threshold_model(transition_thresholds, states):
    # 不只是採用/不採用的二元選擇
    # 多個競爭性創新或行為
    pass
```

## 🔬 高級應用

### 1. 政策干預分析
```python
# 分析政策干預對集體行為的影響
def policy_intervention_analysis(base_thresholds, intervention_effects):
    """評估不同政策對行為擴散的影響"""
    # 補貼、教育、強制等政策工具
    # 如何最有效地達到政策目標
    pass
```

### 2. 市場採用預測
```python
# 預測新產品的市場採用軌跡
def market_adoption_forecast(product_attributes, market_characteristics):
    """基於產品特性和市場特徵預測採用模式"""
    # 價格、質量、網絡效應等因素
    # 競爭產品的影響
    pass
```

### 3. 社會運動分析
```python
# 分析社會運動的爆發和傳播
def social_movement_analysis(grievance_levels, mobilization_resources):
    """模擬社會運動的動員過程"""
    # 不滿程度、組織資源、媒體關注
    # 政府反應和鎮壓效果
    pass
```

## 📂 文件結構

```
threshold/
├── README.md              # 本文檔
├── threshold_simple.py    # 核心實現
├── threshold.py           # 完整實現（含視覺化）
├── demo.py                # 互動式演示
└── test.py                # 測試腳本
```

## 🔗 延伸閱讀

- Granovetter, M. (1978). "Threshold Models of Collective Behavior"
- Rogers, E. M. (2003). *Diffusion of Innovations*
- Schelling, T. C. (1971). "Dynamic Models of Segregation"
- Young, H. P. (1998). *Individual Strategy and Social Structure*
- Watts, D. J. (2002). "A Simple Model of Global Cascades on Random Networks"
- Centola, D. (2018). *How Behavior Spreads*

## 💡 思考題

1. 為什麼某些創新能夠快速傳播，而另一些卻停滯不前？
2. 如何設計政策來促進有益行為的採用（如疫苗接種、環保行為）？
3. 社交媒體如何改變了閾值模型的適用性？
4. 在什麼情況下，少數人可以影響多數人的行為？
5. 如何利用閾值模型來預防系統性金融風險？
6. 人工智能推薦算法如何影響人們的閾值和決策？