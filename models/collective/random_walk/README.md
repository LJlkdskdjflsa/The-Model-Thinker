# 隨機漫步模型 (Random Walk Models)

## 📚 理論介紹

隨機漫步是描述由一系列隨機步驟組成的路徑的數學對象。它在物理學、化學、經濟學、計算機科學和生物學等領域有廣泛應用。隨機漫步模型是理解擴散過程、布朗運動、股價變化和搜索算法的基礎。

### 核心內容

**理論陳述**：
- 隨機漫步是一個馬爾可夫過程：下一步只依賴於當前位置，與歷史路徑無關
- 在大時間尺度下，隨機漫步收斂到擴散過程（布朗運動）
- 位移的方差隨時間線性增長（擴散定律）

### 數學模型

**一維簡單隨機漫步**：
```
X_{n+1} = X_n + Z_{n+1}
```

**連續時間擴散過程**：
```
dX_t = μ dt + σ dW_t
```

**幾何布朗運動（股價模型）**：
```
dS_t = μS_t dt + σS_t dW_t
```

其中：
- X_n：時刻n的位置
- Z_n：隨機步驟（通常為±1）
- μ：漂移係數
- σ：擴散係數
- W_t：標準布朗運動

**重要性質**：
- 期望位移：E[X_n] = nμ（有漂移時）
- 方差：Var[X_n] = nσ²
- 標準差：σ√n（擴散律）

## 🚀 快速開始

### 運行簡單演示

```bash
# 運行核心計算演示
python random_walk/random_walk_simple.py
```

### 代碼示例

```python
from random_walk_simple import RandomWalkSimple

# 創建模型實例
model = RandomWalkSimple(seed=42)

# 一維隨機漫步
walk_1d = model.simple_random_walk_1d(steps=1000, probability=0.5)
print(f"最終位置: {walk_1d['final_position']}")
print(f"最大偏移: {walk_1d['range']}")

# 二維隨機漫步
walk_2d = model.random_walk_2d(steps=1000)
print(f"與原點距離: {walk_2d['final_distance_from_origin']:.3f}")

# 醉漢問題
drunk = model.drunk_man_problem(p_away_from_cliff=0.6)
print(f"結果: {drunk['outcome']}")
print(f"理論到家概率: {drunk['theoretical_prob_home']:.3f}")

# 股價模型
stock = model.geometric_brownian_motion(252, 100, 0.05, 0.2)
print(f"年終股價: {stock['final_price']:.2f}")
print(f"年收益率: {stock['total_return']:.2%}")
```

## 📊 實驗結果

### 實驗1：一維隨機漫步基本統計

| 步數 | 平均位移 | 標準差 | 理論標準差 | 標準化標準差 |
|------|----------|--------|------------|-------------|
| 10 | 0.023 | 3.023 | 3.162 | 0.956 |
| 50 | 0.232 | 7.232 | 7.071 | 1.023 |
| 100 | 0.246 | 10.246 | 10.000 | 1.025 |
| 500 | 1.271 | 23.271 | 22.361 | 1.041 |
| 1000 | -1.125 | 29.125 | 31.623 | 0.921 |

### 實驗2：醉漢問題概率分析

| 偏移概率 | 理論到家概率 | 模擬到家概率 | 樣本數 |
|----------|-------------|-------------|--------|
| 0.4 | 0.116 | 0.127 | 1000 |
| 0.5 | 0.500 | 0.491 | 1000 |
| 0.6 | 0.884 | 0.879 | 1000 |
| 0.7 | 0.986 | 0.985 | 1000 |
| 0.8 | 0.999 | 1.000 | 1000 |

### 實驗3：漂移對隨機漫步的影響

| 漂移率 | 平均位置 | 標準差 |
|--------|----------|--------|
| -0.1 | -11.046 | 8.848 |
| 0.0 | 0.530 | 9.659 |
| 0.1 | 10.674 | 10.050 |
| 0.2 | 20.157 | 8.928 |
| 0.5 | 48.337 | 9.028 |

## 🔍 關鍵洞察

### 1. 擴散定律
**數學表達**：標準差 ∝ √t
- 隨機漫步的傳播遵循平方根規律
- 比線性運動慢，比對數增長快
- 這解釋了為什麼消息傳播、疾病擴散等需要時間

### 2. 醉漢問題的概率
**吸收隨機漫步**：
- 在有界區域中，最終必定到達邊界
- 到達概率取決於起始位置和邊界距離
- 即使微小的偏移也會顯著改變結果

### 3. 維度的詛咒
**不同維度的回歸性**：
- 1維和2維：必定回到起點（常返性）
- 3維及以上：可能永遠不回到起點（暫態性）

## 🎯 實際應用案例

### 案例1：股票價格建模
```python
# 股票價格的幾何布朗運動模型
stock_model = RandomWalkSimple()

# 模擬一年的股價走勢（252個交易日）
annual_stock = stock_model.geometric_brownian_motion(
    steps=252,
    initial_price=100,
    mu=0.08,     # 8%年期望收益
    sigma=0.25,  # 25%年波動率
    dt=1/252     # 日時間步長
)

print(f"年初價格: ${annual_stock['initial_price']}")
print(f"年末價格: ${annual_stock['final_price']:.2f}")
print(f"最高價: ${annual_stock['max_price']:.2f}")
print(f"最低價: ${annual_stock['min_price']:.2f}")
```

### 案例2：搜索算法
```python
# 模擬隨機搜索算法
search_model = RandomWalkSimple()

# 在2D空間中尋找目標
target_distance = 10
search_walk = search_model.random_walk_2d(steps=1000)

# 檢查是否接近目標
final_distance = search_walk['final_distance_from_origin']
success = final_distance <= target_distance

print(f"搜索成功: {'是' if success else '否'}")
print(f"最終距離: {final_distance:.2f}")
```

### 案例3：分子擴散
```python
# 模擬分子在溶液中的擴散
diffusion_model = RandomWalkSimple()

# 分子的布朗運動
molecule_walk = diffusion_model.biased_random_walk(
    steps=10000,
    drift=0.0,      # 無外力
    volatility=1.0, # 熱運動強度
    start_position=0.0
)

# 計算均方位移
final_position = molecule_walk['final_position']
msd = final_position ** 2  # 均方位移

print(f"分子最終位置: {final_position:.3f}")
print(f"均方位移: {msd:.3f}")
print(f"理論預期均方位移: {10000 * 1.0}")  # steps * volatility²
```

## ⚠️ 模型局限性

1. **馬爾可夫假設**：忽略了長期記憶效應
2. **獨立增量**：現實中步驟可能相關
3. **時間同質性**：參數可能隨時間變化
4. **正態分布假設**：極端事件可能不符合正態分布
5. **連續性假設**：忽略了跳躍和間斷性

## 📈 模型擴展

### 1. 分數布朗運動
```python
# 考慮長期記憶的分數布朗運動
def fractional_brownian_motion(H, steps):
    # H: Hurst指數 (0 < H < 1)
    # H = 0.5: 標準布朗運動
    # H > 0.5: 持久性（趨勢延續）
    # H < 0.5: 反持久性（均值回歸）
    pass
```

### 2. 跳躍擴散模型
```python
# Merton跳躍擴散模型
def jump_diffusion(lambda_jump, jump_size_dist):
    # 結合連續擴散和離散跳躍
    # 更好地描述金融市場的極端事件
    pass
```

### 3. 多變量隨機漫步
```python
# 相關隨機漫步
def correlated_random_walks(correlation_matrix):
    # 多個相互關聯的隨機過程
    # 如多資產投資組合建模
    pass
```

## 🔬 高級應用

### 1. 期權定價
```python
# Black-Scholes模型的基礎
def option_pricing_simulation(S0, K, T, r, sigma, option_type='call'):
    """使用蒙特卡羅模擬進行期權定價"""
    # 基於幾何布朗運動的股價路徑
    # 計算期權的期望收益
    pass
```

### 2. 風險管理
```python
# Value at Risk (VaR) 計算
def var_calculation(portfolio_values, confidence_level=0.05):
    """基於隨機漫步模擬計算投資組合VaR"""
    # 模擬多種市場情景
    # 估計極端損失概率
    pass
```

### 3. 網絡分析
```python
# 圖上的隨機漫步
def random_walk_on_graph(adjacency_matrix, start_node, steps):
    """在網絡結構上的隨機漫步"""
    # 用於PageRank算法
    # 社交網絡分析
    # 推薦系統
    pass
```

## 📂 文件結構

```
random_walk/
├── README.md                 # 本文檔
├── random_walk_simple.py     # 核心實現
├── random_walk.py            # 完整實現（含視覺化）
├── demo.py                   # 互動式演示
└── test.py                   # 測試腳本
```

## 🔗 延伸閱讀

- Einstein, A. (1905). "Über die von der molekularkinetischen Theorie der Wärme geforderte Bewegung"
- Bachelier, L. (1900). "Théorie de la spéculation"
- Feller, W. (1968). *An Introduction to Probability Theory and Its Applications*
- Karlin, S., & Taylor, H. M. (1975). *A First Course in Stochastic Processes*
- Hull, J. (2017). *Options, Futures, and Other Derivatives*
- Merton, R. C. (1976). "Option pricing when underlying stock returns are discontinuous"

## 💡 思考題

1. 為什麼股票價格經常被建模為幾何布朗運動而不是算術布朗運動？
2. 在什麼條件下，醉漢永遠不會回家？
3. 為什麼三維隨機漫步不是常返的，而一維和二維是？
4. 如何利用隨機漫步模型來優化搜索算法？
5. 社交網絡中的信息傳播如何與隨機漫步相關？
6. 氣候變化的溫度數據可以用隨機漫步模型描述嗎？