# 📊 Fama-French Models: FF3 vs FF5 Portfolio Analysis & Optimization

This project builds, evaluates, and compares optimized portfolios using the **Fama-French 3-Factor (FF3)** and **Fama-French 5-Factor (FF5)** models. It combines quantitative finance theory with Python for robust portfolio construction, backtesting, and performance visualization.

---

## 🚀 Features

- 📈 **Portfolio Optimization** using factor exposures (FF3 & FF5)
- 🔍 **Factor Regression** with statsmodels (OLS)
- 🔁 **Rolling Sharpe Ratio Plots** for dynamic risk-adjusted returns
- 📉 **Drawdown Analysis** and Max Drawdown visualization
- 🧠 Key Metrics: CAGR, Volatility, Sharpe, Sortino, Calmar Ratios
- 📊 Comparison with S&P 500 benchmark (if provided)
- 📂 Automated CSV export and high-res PNG chart saving

---

## 📁 Project Structure

fama-french-models/
│
├── main.py                      # Main script to run full analysis
├── modules/
│   ├── analysis.py              # Core performance & risk metrics
│   └── plotting.py              # Plot functions (cumulative, rolling Sharpe, drawdown)
│
├── Data/
│   ├── output/                  # Saved plots and CSVs
│   └── raw/                     # Raw or downloaded financial data (optional)
│
├── FF3_FF5_model_notebook/     # Jupyter notebook version (exploration & visuals)
├── README.md                   # You’re reading it!
├── Requirements.txt            # Python dependencies

---

## 🧠 How It Works (Process Flow)

1. Load Price & Factor Data
2. Calculate Daily & Cumulative Returns
3. Run OLS Regression (FF3 & FF5)
4. Compute Performance Metrics
5. Visualize Sharpe, Drawdown
6. Export Charts and CSVs
