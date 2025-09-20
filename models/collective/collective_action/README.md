# 集體行動問題模型 (Collective Action Problem Models)

## 📚 理論介紹

集體行動問題是由曼卡·奧爾森（Mancur Olson）在1965年《集體行動的邏輯》中首次系統化闡述的重要理論。該理論解釋了為什麼理性的個體往往難以自發地提供公共物品，即使這樣做符合集體利益。

### 核心內容

**理論陳述**：
- 在沒有外部強制或激勵的情況下，理性個體往往不會自願為公共物品做出最優貢獻
- 群體規模越大，個體貢獻公共物品的動機越弱（奧爾森悖論）
- 搭便車問題導致公共物品供應不足

### 數學模型

個體 i 的效用函數：
```
Uᵢ = bᵢ × G - cᵢ × gᵢ
```

公共物品生產函數：
```
G = f(g₁ + g₂ + ... + gₙ)
```

其中：
- Uᵢ：個體 i 的效用
- bᵢ：個體 i 從公共物品中獲得的邊際收益
- G：公共物品總量
- cᵢ：個體 i 的邊際成本
- gᵢ：個體 i 的貢獻
- f(·)：生產函數

**納什均衡條件**：
```
∂Uᵢ/∂gᵢ = bᵢ × (∂G/∂gᵢ) - cᵢ = 0
```

## 🚀 快速開始

### 運行簡單演示

```bash
# 運行核心計算演示
python collective_action/collective_action_simple.py
```

### 代碼示例

```python
from collective_action_simple import CollectiveActionSimple

# 創建模型實例
model = CollectiveActionSimple()

# 分析5人群體的納什均衡
equilibrium = model.nash_equilibrium_analysis(
    n_players=5,
    benefit_coefficients=[1.0] * 5,
    cost_coefficients=[1.0] * 5
)

print(f"均衡貢獻: {equilibrium['equilibrium_contributions']}")
print(f"公共物品水準: {equilibrium['public_good_level']:.3f}")
print(f"總效用: {equilibrium['total_utility']:.3f}")

# 分析搭便車問題
free_rider = model.free_rider_analysis(n_players=5)
print(f"搭便車誘因: {free_rider['free_rider_incentive']:.3f}")
```

## 📊 實驗結果

### 實驗1：群體規模效應（奧爾森悖論）

| 群體規模 | 平均貢獻 | 平均效用 | 提供率 | 搭便車比例 |
|----------|----------|----------|--------|------------|
| 2人 | 0.500 | 0.500 | 100% | 0% |
| 5人 | 0.200 | 0.400 | 100% | 0% |
| 10人 | 0.100 | 0.350 | 100% | 0% |
| 20人 | 0.050 | 0.325 | 100% | 0% |
| 50人 | 0.020 | 0.310 | 100% | 0% |

### 實驗2：不同生產函數的影響

| 生產函數類型 | 總貢獻 | 公共物品水準 | 總效用 |
|-------------|--------|-------------|--------|
| 線性 | 1.000 | 1.000 | 2.000 |
| 閾值 | 2.500 | 2.500 | 2.500 |
| 加速遞增 | 1.000 | 1.000 | 2.000 |
| 邊際遞減 | 1.000 | 1.000 | 2.000 |

### 實驗3：激勵機制效果比較

| 激勵機制 | 平均貢獻 | 平均效用 | 總福利 |
|----------|----------|----------|--------|
| 基線（無激勵） | 0.100 | 0.350 | 3.500 |
| 補貼機制 | 0.600 | 0.800 | 8.000 |
| 配對機制 | 0.250 | 0.625 | 6.250 |

## 🔍 關鍵洞察

### 1. 奧爾森悖論
- **小群體優勢**：小群體更容易克服集體行動問題
- **大群體困境**：隨著群體規模增大，個體貢獻動機減弱
- **理論機制**：個體影響力隨群體規模遞減

### 2. 搭便車問題
**數學特徵**：
- 個體邊際收益 < 個體邊際成本
- 但集體邊際收益 > 集體邊際成本
- 導致納什均衡次優於帕累托最優

### 3. 生產函數的影響
- **線性**：標準情況，貢獻與產出成正比
- **閾值**：需要達到臨界質量才能產生公共物品
- **加速遞增**：早期貢獻者可以激勵後續參與
- **邊際遞減**：後期貢獻的邊際效用下降

## 🎯 實際應用案例

### 案例1：環境保護
```python
# 氣候變化減排行動
climate_model = CollectiveActionSimple()

# 100個國家的減排博弈
countries_analysis = climate_model.nash_equilibrium_analysis(
    n_players=100,
    benefit_coefficients=[1.0] * 100,  # 所有國家受益相同
    cost_coefficients=[0.8] * 100      # 減排成本
)

print(f"各國平均減排貢獻: {sum(countries_analysis['equilibrium_contributions'])/100:.3f}")
print(f"全球氣候改善水準: {countries_analysis['public_good_level']:.3f}")
```

### 案例2：開源軟件開發
```python
# 開源項目貢獻者分析
opensource_model = CollectiveActionSimple()

# 不同規模開發者群體
small_team = opensource_model.nash_equilibrium_analysis(5, [2.0] * 5)  # 小團隊，高收益
large_community = opensource_model.nash_equilibrium_analysis(50, [1.5] * 50)  # 大社區

print(f"小團隊平均貢獻: {sum(small_team['equilibrium_contributions'])/5:.3f}")
print(f"大社區平均貢獻: {sum(large_community['equilibrium_contributions'])/50:.3f}")
```

### 案例3：公共設施維護
```python
# 社區公園維護
community_model = CollectiveActionSimple()

# 比較不同激勵機制
baseline = community_model.nash_equilibrium_analysis(20, [1.0] * 20)
subsidy = community_model.nash_equilibrium_analysis(20, [1.0] * 20, [0.5] * 20)

print(f"無激勵總貢獻: {baseline['total_contribution']:.3f}")
print(f"政府補貼總貢獻: {subsidy['total_contribution']:.3f}")
```

## ⚠️ 模型局限性

1. **理性人假設**：假設所有個體都是完全理性的經濟人
2. **完全信息**：假設個體對收益和成本有完全信息
3. **靜態分析**：主要關注一次性博弈，較少考慮重複博弈
4. **同質性假設**：通常假設個體具有相似的偏好和能力
5. **外部性忽略**：可能忽略社會規範、聲譽等非經濟因素

## 📈 解決策略

### 1. 制度設計
- **強制機制**：政府法規和稅收
- **激勵相容**：設計使個體利益與集體利益一致的機制
- **分級提供**：不同規模群體提供不同層次的公共物品

### 2. 社會機制
- **社會規範**：建立合作文化和同伴壓力
- **聲譽機制**：長期重複博弈中的聲譽考量
- **選擇性激勵**：為貢獻者提供私人物品激勵

### 3. 技術解決方案
- **降低成本**：技術進步降低貢獻成本
- **提高效率**：改善生產函數，提高貢獻效果
- **信息透明**：增加信息透明度，促進協調

## 📂 文件結構

```
collective_action/
├── README.md                      # 本文檔
├── collective_action_simple.py    # 核心實現
├── collective_action.py           # 完整實現（含視覺化）
├── demo.py                        # 互動式演示
└── test.py                        # 測試腳本
```

## 🔗 延伸閱讀

- Olson, M. (1965). *The Logic of Collective Action: Public Goods and the Theory of Groups*
- Ostrom, E. (1990). *Governing the Commons: The Evolution of Institutions*
- Sandler, T. (2004). *Global Collective Action*
- Hardin, G. (1968). "The Tragedy of the Commons"
- Marwell, G., & Oliver, P. (1993). *The Critical Mass in Collective Action*

## 💡 思考題

1. 為什麼小國往往在國際合作中比大國更積極？
2. 如何解釋開源軟件項目的成功？
3. 氣候變化談判中的"共同但有區別的責任"原則如何體現集體行動理論？
4. 為什麼有些公共物品（如國防）需要政府提供，而有些（如慈善）可以自願提供？
5. 數字時代的網絡效應如何改變傳統的集體行動問題？
6. 如何設計機制來促進疫情期間的集體防疫行為？