# 📊 Fama-French Models: Portfolio Optimization

This project explores **portfolio optimization using the Fama-French 3-Factor (FF3) model**, comparing performance against the S&P 500 benchmark. It includes data collection, return calculation, performance evaluation, and regression analysis for multiple tickers.

---

## ✅ Current Features (FF3)

### 🔹 1. User Input
- Specify tickers, start date, and end date interactively
- Automatically formats and fetches data via `yfinance`

### 🔹 2. Data Collection
- Pulls adjusted close prices and calculates monthly returns
- Merges stock returns with Fama-French 3-Factor data

### 🔹 3. Portfolio Optimization
- Uses CVXPY to maximize the Sharpe Ratio based on FF3 regression
- Constraints: full investment (sum(weights)=1), no short selling

### 🔹 4. Performance Metrics
- Calculates and compares:
  - Total Return
  - Annualized Return & Volatility
  - Sharpe Ratio
  - Max Drawdown

### 🔹 5. Visualization
- Drawdown plots (FF3 Optimized Portfolio vs. S&P 500)
- Optional bar chart of FF3 regression coefficients

### 🔹 6. CSV Export
- Save outputs to `/data/output/` folder:
  - `merged_data.csv`
  - `portfolio_returns.csv`
  - `benchmark_returns.csv`
  - `performance_summary.csv`
  - `regression_coefficients.csv` (if applicable)

---

## 🚀 Planned Features (FF5 Expansion)

### Coming Soon:
- 🔸 **Fama-French 5-Factor Model** implementation
- 🔸 Compare FF3 vs FF5 optimized portfolios
- 🔸 Factor exposure visualization: heatmaps and radar charts
- 🔸 Add transaction cost and turnover analysis
- 🔸 Monte Carlo simulation for risk stress testing

---

## 🧠 Tech Stack

- **Python**
- `pandas`, `NumPy`, `yfinance`, `statsmodels`, `cvxpy`
- `matplotlib`, `seaborn`, `tabulate`

---

## 📁 Folder Structure


## 👤 Author
Wielly Halim 
🔗 [https://www.linkedin.com/in/wiellyhalim]  
📬 Feel free to fork, contribute, or reach out!
