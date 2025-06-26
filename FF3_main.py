# Import essential libraries for data analysis, visualization, and modeling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
from scipy.stats import norm
import seaborn as sns
from tabulate import tabulate
import statsmodels.api as sm
import cvxpy as cp

# Load the modules
from FF3_modules.data_loader import fetch_data
from FF3_modules.analysis import ff3_regression_analysis
from FF3_modules.visualizer import ff3_fitted_plt
from FF3_modules.calculation import optimize_portfolio_ff3_risk_adjusted
from FF3_modules.visualizer import plot_portfolio_pie
from FF3_modules.visualizer import backtest_portfolio
from FF3_modules.visualizer import plot_drawdown
from FF3_modules.analysis import performance_summary
from FF3_modules.visualizer import plot_relative_performance

#Step 1: Input tickers, start and end date
tickers = input("Enter tickers (comma separated, e.g., AAPL, MSFT, TSLA): ").split(",")
start_date = input("Enter start date (YYYY-MM-DD): ")
end_date = input("Enter end date (YYYY-MM-DD): ")
tickers = [t.strip().upper() for t in tickers]

# Step 2: Fetch tickers 'Close' and 'Return'
adj_close, returns = fetch_data(tickers, start=start_date, end=end_date)

# Step 3: Load FF3 data 
ff3 = pd.read_csv(f'data/ff_factors/F-F_Research_Data_Factors.CSV', skiprows=3)
ff3.rename(columns={ff3.columns[0]: 'Date'}, inplace=True)
ff3 = ff3[ff3['Date'].astype(str).str.len() == 6]
ff3['Date'] = pd.to_datetime(ff3['Date'], format='%Y%m')

for col in ['Mkt-RF', 'SMB', 'HML', 'RF']:
    ff3[col] = ff3[col].astype(float) / 100

# ✅ Proper filtering using input dates
ff3 = ff3[(ff3['Date'] >= pd.to_datetime(start_date)) & (ff3['Date'] <= pd.to_datetime(end_date))]

# Step 4: Processing Data
# Reset index and convert wide return table to long format
returns_reset = returns.reset_index()  # Date becomes a column
returns_melted = returns_reset.melt(id_vars='Date', var_name='Ticker', value_name='Return')

# Ensure 'Date' columns are datetime in both DataFrames
returns_melted['Date'] = pd.to_datetime(returns_melted['Date'])
ff3['Date'] = pd.to_datetime(ff3['Date'])

# Merge on 'Date'
merged_data = pd.merge(returns_melted, ff3, on='Date', how='inner')

# Calculate excess return
merged_data['Excess Return'] = merged_data['Return'] - merged_data['RF']

# Step 5: Generate Regression summary and Coefficients for Alpha, Mkt-RF, SMB, HML per Ticker
# Initialize lists
model,coef_df = ff3_regression_analysis(merged_data)
print (coef_df)

# Step 6: Plotting actual vs.fitted regression per ticker
ff3_fitted_plt(merged_data)

# Step 7: Portfolio optimization with FF3 Model
if len(tickers) > 1:
    # Ask user to input risk aversion level
    print("\n🎯 Risk Aversion Level (higher = more conservative):")
    print(" - 1–5   ➜ Aggressive (maximize returns)")
    print(" - 10–20 ➜ Balanced (moderate risk-return trade-off)")
    print(" - 30+   ➜ Conservative (prefer low volatility)")
    try:
        risk_aversion = float(input("Enter your risk aversion level (e.g., 10): ").strip())
    except ValueError:
        print("❌ Invalid input. Using default: 10")
        risk_aversion = 10

    # 1. Compute average factor values per ticker
    mean_merged_data = merged_data.groupby('Ticker').mean(numeric_only=True)

    # 2. Sort coefficient DataFrame to align with tickers
    coef_df = coef_df[sorted(coef_df.columns)]

    # 3. Extract alpha and factor loadings
    alpha    = coef_df.T['Alpha'].values
    beta_mkt = coef_df.T['Mkt-RF'].values
    beta_smb = coef_df.T['SMB'].values
    beta_hml = coef_df.T['HML'].values

    # 4. Compute expected excess return using FF3 model
    expected_excess_return = (
        alpha
        + beta_mkt * mean_merged_data['Mkt-RF'].values
        + beta_smb * mean_merged_data['SMB'].values
        + beta_hml * mean_merged_data['HML'].values
    )

    # 5. Compute covariance matrix of returns (match ticker order)
    tickers = coef_df.columns.tolist()
    cov_matrix = returns[tickers].cov().values

    # 6. Optimize portfolio weights
    portfolio_weights = optimize_portfolio_ff3_risk_adjusted(
        expected_excess_return,
        cov_matrix,
        tickers,
        risk_aversion
    )

    # 7. Display final results
    portfolio_weights['Weight (%)'] = (portfolio_weights['Weight'] * 100).round(2)
    print("\n✅ Optimized Portfolio Weights:")
    print(portfolio_weights[['Ticker', 'Weight (%)']])
    plot_portfolio_pie(portfolio_weights)
else:
    print("⚠️ Not enough tickers to perform optimization.")

# Step 8: Backtesting w/ and w/o S&P500 and drawdown/relative performance to S&P500
# Download S&P 500 daily data
snp = yf.download('^GSPC', start='2020-01-01', end='2025-01-01', interval='1d', auto_adjust=True)

# Resample to monthly and compute returns
snp_monthly = snp['Close'].resample('ME').last()
snp_returns = snp_monthly.pct_change().dropna()

# Optional: Rename for clarity
snp_returns.name = snp_response = input("Input S&P500 as benchmark? (y/n): ").strip().lower()

if snp_response == 'y':
    portfolio_returns, benchmark_returns = backtest_portfolio(portfolio_weights, returns, snp_returns)
    plot_relative_performance(portfolio_returns, benchmark_returns)
else:
    backtest_portfolio(portfolio_weights, returns)

#Plot drawdown plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
# FF3 Portfolio
plot_drawdown(portfolio_returns, label="FF3 Optimized Portfolio", ax=ax1)
# S&P 500
plot_drawdown(benchmark_returns, label="S&P 500", ax=ax2)
# Force same x-axis limits
ax2.set_xlim(ax1.get_xlim())
plt.tight_layout()
plt.show()

#Portfolio Performance vs. Benchmark (S&P500) Summary 
performance_summary(portfolio_returns, label="FF3 Optimized Portfolio")
if benchmark_returns is not None:
    performance_summary(benchmark_returns, label="S&P 500")

# Ask user whether to save outputs
if input("Save all outputs to CSV? (y/n): ").strip().lower() == 'y':
    # Define all outputs
   # Get raw metrics from performance_summary
    ff3_metrics = performance_summary(portfolio_returns, label="FF3 Optimized Portfolio")
    if benchmark_returns is not None:
        sp_metrics = performance_summary(benchmark_returns, label="S&P 500")
     
    outputs = {
        "merged_data.csv": merged_data,
        "portfolio_returns.csv": portfolio_returns,
        "benchmark_returns.csv": benchmark_returns if benchmark_returns is not None else None,
        "performance_summary.csv": pd.DataFrame({
            "Metric": list(ff3_metrics.keys()),
            "FF3 Optimized": list(ff3_metrics.values()),
            "S&P 500": list(sp_metrics.values()) if benchmark_returns is not None else [None]*5
        }),
        "regression_coefficients.csv": coef_df if 'coef_df' in locals() else None
    }

    # Save each available output
    for name, df in outputs.items():
        if df is not None:
            file_path = f"data/output/{name}.csv"
            df.to_csv(file_path, index=True)
            print(f"✅ {name}.csv saved.")
            print(f"📄 {name.replace('_', ' ').title()} data saved to {file_path}\n")
        else:
            print(f"⚠️ {name}.csv not available.")
else:
    print("❌ Save cancelled.")