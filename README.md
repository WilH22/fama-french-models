# 📊 Fama-French Models: Portfolio Optimization

This project explores **portfolio optimization using the Fama-French 3-Factor (FF3) model**, comparing performance against the S&P 500 benchmark. It includes data collection, return calculation, performance evaluation, and regression analysis for multiple tickers.

---
📈 About Fama-French 3-Factor Model
The Fama-French 3-Factor Model is an asset pricing model that extends the Capital Asset Pricing Model (CAPM) by adding two additional factors to explain stock returns:

1️⃣ Market Risk (Market Premium):
Similar to CAPM, it considers the excess return of the market over the risk-free rate.

2️⃣ Size Factor (SMB: Small Minus Big):
Accounts for the tendency of small-cap stocks to outperform large-cap stocks over the long term.

3️⃣ Value Factor (HML: High Minus Low):
Captures the tendency of high book-to-market (value) stocks to outperform low book-to-market (growth) stocks.

The model’s formula:
![image](https://github.com/user-attachments/assets/4819ed82-e06e-475f-8734-020ee03d66ae)
![image](https://github.com/user-attachments/assets/076bd523-90aa-4b22-a082-6a1a9cc305d0)

This model helps in understanding asset returns by attributing performance to market exposure, company size, and value factors, allowing investors to better evaluate and adjust their portfolio strategies.


## ✅ Current Features (FF3)

### 🔹 1. User Input
- Specify tickers, start date, and end date interactively
- Automatically formats and fetches data via `yfinance`

![image](https://github.com/user-attachments/assets/de003612-9e8f-40c0-9698-39b9d788f82d)

### 🔹 2. Data Collection
- Pulls adjusted close prices and calculates monthly returns
- Merges stock returns with Fama-French 3-Factor data
- Generate Regression summary and Coefficients for Alpha, Mkt-RF, SMB, HML per ticker
- Plotting actual vs. fitted regression per ticker

![image](https://github.com/user-attachments/assets/a0129c0c-22ea-4148-b131-1f9f3403dd61)

![image](https://github.com/user-attachments/assets/cc24fe00-d30c-4b6b-93d0-ff39653e6fff)

### 🔹 3. Portfolio Optimization
- Uses CVXPY to maximize the Sharpe Ratio based on FF3 regression
- Users can input their risk aversion (default = 10)
- Constraints: full investment (sum(weights)=1), no short selling
- Generated pie chart for the optimized portfolio weight
- 
![image](https://github.com/user-attachments/assets/1ad6c9dc-c987-4e61-80aa-fc3a1f7235e3)

    
### 🔹 4. Performance Metrics
- Calculates and compares:
  - Total Return
  - Annualized Return & Volatility
  - Sharpe Ratio
  - Max Drawdown

### 🔹 5. Visualization
- Drawdown plots (FF3 Optimized Portfolio vs. S&P 500)
- Drawdown plots (FF3 Optimized Portfolio vs. S&P 500)

![image](https://github.com/user-attachments/assets/944b18ec-b854-42d8-bf5e-ed4549c917a3)

### 🔹 6. CSV Export
- Save outputs to `/data/FF3_output/` folder:
  - `merged_data.csv`
  - `portfolio_returns.csv`
  - `benchmark_returns.csv`
  - `performance_summary.csv`
  - `regression_coefficients.csv` (if applicable)

![image](https://github.com/user-attachments/assets/2d9d9ee5-4fee-4468-8d8d-4b64340d4564)

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
