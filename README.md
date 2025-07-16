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

<img width="505" height="286" alt="image" src="https://github.com/user-attachments/assets/20452a7b-81fb-4b4a-ba4d-cd1120c23686" />

---

## 🧠 How It Works (Process Flow)

1. **📥 Load Price & Factor Data**  
   <img width="608" height="46" alt="image" src="https://github.com/user-attachments/assets/d56d16c9-ba8a-428b-b766-ec2764ecb8c6" />
 
   The user inputs stock tickers and a time range. The script automatically fetches both price data and Fama-French factor datasets.

2. **📈 Calculate Daily & Cumulative Returns**  
<img width="471" height="163" alt="image" src="https://github.com/user-attachments/assets/dddc5c06-b7b6-411b-b3a8-cb6aa83257bc" />
<img width="689" height="702" alt="image" src="https://github.com/user-attachments/assets/fb65fc74-9ef6-4282-87d2-d46799f7255b" />

   - Computes daily and cumulative returns for each ticker  
   - Performs OLS regression with FF3 and FF5 factors  
   - Automatically calculates coefficients and statistical details  
   - Saves regression outputs to `Data/output/data/` and `Data/output/plot/`  
   - Displays summary interpretations in the terminal

<img width="464" height="126" alt="image" src="https://github.com/user-attachments/assets/c4202877-57e5-41d4-b6ef-31bf9d2e37e0" />



3. **⚖️ Set Risk Aversion & Optimize Portfolios**  
   - User inputs **risk aversion level** (default = `10`)

<img width="380" height="95" alt="image" src="https://github.com/user-attachments/assets/05238e22-aff8-4f66-a778-b1fca0355028" />

   - Choose to optimize with:
     - Fama-French **3-Factor (FF3)**
     - Fama-French **5-Factor (FF5)**
     - Select parameter method: **mean** or **last value**

<img width="445" height="402" alt="image" src="https://github.com/user-attachments/assets/943bfa0b-4866-4e3e-869f-30c064353a94" />


   - View pie charts for each optimized portfolio

<img width="837" height="701" alt="image" src="https://github.com/user-attachments/assets/43b28714-cd77-4589-b220-35282701e08c" />



4. **🔁 Backtest vs Benchmark or Compare Models**  
   - Choose to:
     - Compare FF3 & FF5 directly  
     - Or include **S&P 500** benchmark

<img width="499" height="35" alt="image" src="https://github.com/user-attachments/assets/be7d0319-cfa7-4812-9c48-58a3cf6e9a69" />

   - Plots cumulative performance:
     - FF3 Optimized Portfolio  
     - FF5 Optimized Portfolio  
     - S&P 500 (if provided)

<img width="829" height="336" alt="image" src="https://github.com/user-attachments/assets/334356fd-723a-4bf5-b0b4-a3863ff44e4d" />
  
   - Output saved to `Data/output/plot/`  
   - If any dataset is missing, the plot adjusts dynamically



5. **📉 Drawdown Analysis**  
   - Computes **Max Drawdown** for each model

<img width="713" height="289" alt="image" src="https://github.com/user-attachments/assets/3c5df5cc-1082-40f9-a6f9-65270d36cdf9" />
  
   - Visualizes downside risk profile

<img width="833" height="544" alt="image" src="https://github.com/user-attachments/assets/fafa0add-263b-46f4-8282-2f9870684855" />



6. **📊 Compute Performance Metrics**  
   - Prompt for **risk-free rate** (default = `0.03`)  
   - Calculates:
     - CAGR  
     - Volatility  
     - Sharpe Ratio  
     - Sortino Ratio  
     - Calmar Ratio  
   - Generates rolling 12-month Sharpe Ratio plots

<img width="203" height="328" alt="image" src="https://github.com/user-attachments/assets/7f8ca6be-e1d6-4ca4-94a0-45c0cf92b358" />
<img width="818" height="405" alt="image" src="https://github.com/user-attachments/assets/47792424-97a4-4ee6-b6ab-6d7261392d8c" />



7. **📊Plotting Efficient Frontier for FF3 and FF5**
   - This chart visualizes the **efficient frontier** for portfolios optimized using Fama-French 3-Factor and 5-Factor models.
   - The optimal portfolios are highlighted, allowing you to compare risk-return trade-offs.

<img width="705" height="985" alt="image" src="https://github.com/user-attachments/assets/317e44b6-f5ca-4c88-a14c-3f0b03ddaa57" />



8. **💾 Final Save Option**  
   - Prompt to save **raw data outputs** to `Data/output/data/`  
   - All visualizations and CSVs are saved automatically


## 📌 Example Output

### Exports

- `Data/output/data/
<img width="623" height="493" alt="image" src="https://github.com/user-attachments/assets/9fcf37ac-e060-490b-94e9-f14405f8f157" />


- `Data/output/plot
<img width="1196" height="457" alt="image" src="https://github.com/user-attachments/assets/f86776e2-a883-4f0c-bc99-f3e29f55c52e" />

---
### 🚧 Future Work

- 🧪 **Monte Carlo Simulation for Optimized Portfolios**
  - Implement simulation using **Geometric Brownian Motion (GBM)** and **Jump Diffusion models**
  - Enable forward-looking scenario analysis for FF3 and FF5 optimized portfolios
  - Visualize potential price paths, volatility cones, and probabilistic drawdowns

- 🔍 **Stress Testing & Tail Risk Estimation**
  - Incorporate **Value-at-Risk (VaR)** and **Conditional VaR (CVaR)** using historical and Monte Carlo methods
---

## 📝 User Inputs

- **📌 Tickers**: Stock symbols to analyze (e.g., `AAPL`, `MSFT`, `TSLA`)  
- **📅 Date Range**: Start and end date for historical prices and factor data  
- **🎯 Risk Aversion Level**: Used in portfolio optimization (default = `10`)  
- **📈 Optimization Type**:  
  - Optimize using **mean values** or **latest values** of parameters  
  - Choose between **FF3**, **FF5**, or both  
- **📊 Benchmark**: Optionally include **S&P 500** for performance comparison  
- **📉 Risk-Free Rate**: Used for Sharpe and Sortino (default = `0.03`)  
- **💾 Save Option**: Prompt to save raw data and output plots  

---

## 📦 Requirements

```bash
pip install yfinance pandas numpy matplotlib statsmodels seaborn scipy tabulate

