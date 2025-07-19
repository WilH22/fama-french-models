# main.py - Fama-French 3-Factor and 5-Factor Model Analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
import cvxpy as cp
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import norm
from IPython.display import display

from modules.analysis import fetch_data
from modules.analysis import ff_regression_analysis
from modules.analysis import optimize_portfolio
from modules.analysis import compute_portfolio_returns
from modules.analysis import compute_cumulative_returns
from modules.analysis import performance_summary
from modules.analysis import compare_ff3_ff5
from modules.analysis import analyze_ff3_ff5_comparison
from modules.analysis import calculate_drawdown_duration
from modules.analysis import calculate_performance_metrics
from modules.analysis import compute_modeling_variable_auto
from modules.analysis import compute_modeling_variable_manual
from modules.analysis import simulate_price_paths

from modules.plotting import display_regression_parameters
from modules.plotting import ff_fitted_plt
from modules.plotting import plot_portfolio_pie
from modules.plotting import compare_portfolios_with_benchmark
from modules.plotting import plot_relative_performance
from modules.plotting import plot_drawdown_duration_comparison
from modules.plotting import plot_drawdown_series
from modules.plotting import plot_rolling_sharpe_auto
from modules.plotting import plot_efficient_frontier_with_optimal
from modules.plotting import plot_price_paths

#Step 1: Input ticker, start and end date
tickers = input("Enter tickers (comma separated, e.g., AAPL, MSFT, TSLA): ").split(",")
start_date = input("Enter start date (YYYY-MM-DD): ")
end_date = input("Enter end date (YYYY-MM-DD): ")
tickers = [t.strip().upper() for t in tickers]

#Step 2: Fetch tickers 'Close' and 'Return'
adj_close, returns = fetch_data(tickers, start=start_date, end=end_date)

#FF3-Factor
# Step 1: Load FF3 data 
ff3 = pd.read_csv('Data/ff_factors/F-F_Research_Data_Factors.CSV', skiprows=3)
ff3.rename(columns={ff3.columns[0]: 'Date'}, inplace=True)
ff3 = ff3[ff3['Date'].astype(str).str.len() == 6]
ff3['Date'] = pd.to_datetime(ff3['Date'], format='%Y%m')

for col in ['Mkt-RF', 'SMB', 'HML', 'RF']:
    ff3[col] = ff3[col].astype(float) / 100

# ✅ Proper filtering using input dates
ff3 = ff3[(ff3['Date'] >= pd.to_datetime(start_date)) & (ff3['Date'] <= pd.to_datetime(end_date))]

# Step 2: Processing Data
# Reset index and convert wide return table to long format
returns_reset = returns.reset_index()  # Date becomes a column
returns_melted = returns_reset.melt(id_vars='Date', var_name='Ticker', value_name='Return')

# Ensure 'Date' columns are datetime in both DataFrames
returns_melted['Date'] = pd.to_datetime(returns_melted['Date'])
ff3['Date'] = pd.to_datetime(ff3['Date'])

# Merge on 'Date'
merged_data_ff3 = pd.merge(returns_melted, ff3, on='Date', how='inner')

# Calculate excess return
merged_data_ff3['Excess Return'] = merged_data_ff3['Return'] - merged_data_ff3['RF']

# Step 3: Generate Regression summary and Coefficients for Alpha, Mkt-RF, SMB, HML per Ticker
# Initialize lists
model_ff3,coef_df_ff3 = ff_regression_analysis(merged_data_ff3,'FF3')

# Step 4: Display FF3 Regression Parameters 
display_regression_parameters(coef_df_ff3,'FF3')

# Step 5: Plotting actual vs. fitted regression per ticker
ff_fitted_plt(merged_data_ff3, 'FF3')

#FF5-Factor
# Step 1: Load FF5 data 
ff5 = pd.read_csv('Data/ff_factors/F-F_Research_Data_5_Factors_2x3.csv', skiprows=3)
ff5.rename(columns={ff5.columns[0]: 'Date'}, inplace=True)
ff5 = ff5[ff5['Date'].astype(str).str.len() == 6]
ff5['Date'] = pd.to_datetime(ff5['Date'], format='%Y%m')

for col in ['Mkt-RF', 'SMB', 'HML','RMW','CMA','RF']:
    ff5[col] = ff5[col].astype(float) / 100

# ✅ Proper filtering using input dates
ff5 = ff5[(ff5['Date'] >= pd.to_datetime(start_date)) & (ff5['Date'] <= pd.to_datetime(end_date))]

# Step 2: Processing Data for FF5
# Reset index and convert wide return table to long format
returns_reset = returns.reset_index()  # Date becomes a column
returns_melted = returns_reset.melt(id_vars='Date', var_name='Ticker', value_name='Return')

# Ensure 'Date' columns are datetime in both DataFrames
returns_melted['Date'] = pd.to_datetime(returns_melted['Date'])
ff5['Date'] = pd.to_datetime(ff5['Date'])

# Merge on 'Date'
merged_data_ff5 = pd.merge(returns_melted, ff5, on='Date', how='inner')

# Calculate excess return
merged_data_ff5['Excess Return'] = merged_data_ff5['Return'] - merged_data_ff5['RF']

#Summary for FF5
model,coef_df_ff5 = ff_regression_analysis(merged_data_ff5,'FF5')

#Display FF5 Regression Parameters
display_regression_parameters(coef_df_ff5,'FF5')

# Step 5: Plotting actual vs. fitted regression per ticker
ff_fitted_plt(merged_data_ff5, 'FF5')

# 🚩 Comparing FF3 vs FF5: Adjusted R² and Alpha via structured analysis and visualization
print("\n🔍 Comparing FF3 vs FF5 Models...")
comparison_df = compare_ff3_ff5(coef_df_ff3, coef_df_ff5)

# Summary for FF3 vs. FF5 Model
analyze_ff3_ff5_comparison(comparison_df)

# 🚩 Portfolio Optimization using FF3 and FF5 Models
if len(tickers) > 1:
    # 🎯 Risk Aversion Input
    print("\n🎯 Risk Aversion Level (higher = more conservative):")
    print(" - 1–5   ➜ Aggressive (maximize returns)")
    print(" - 10–20 ➜ Balanced (moderate risk-return trade-off)")
    print(" - 30+   ➜ Conservative (prefer low volatility)")

    try:
        risk_aversion = float(input("Enter your risk aversion level (e.g., 10):").strip())
    except ValueError:
        print("❌ Invalid input. Using default: 10")
        risk_aversion = 10

    # ✅ FF3 Optimization
    run_ff3 = input("\nCalculate Optimize Portfolio Using FF3 Model? (y/n): ").strip().lower()
    if run_ff3 == 'y':
        long_term = input('Calculate for long-term mean factor premium for FF3? (y/n): ').strip().lower() == 'y'

        factor_premiums_ff3 = (
            merged_data_ff3[['Mkt-RF', 'SMB', 'HML']].mean()
            if long_term else
            merged_data_ff3[['Mkt-RF', 'SMB', 'HML']].iloc[-1]
        )

        tickers_sorted = sorted(coef_df_ff3.columns)
        coef_df_ff3_sorted = coef_df_ff3[tickers_sorted]

        alpha = coef_df_ff3_sorted.T['Alpha'].values
        beta_mkt = coef_df_ff3_sorted.T['Mkt-RF'].values
        beta_smb = coef_df_ff3_sorted.T['SMB'].values
        beta_hml = coef_df_ff3_sorted.T['HML'].values

        expected_excess_return_ff3 = (
            alpha +
            beta_mkt * factor_premiums_ff3['Mkt-RF'] +
            beta_smb * factor_premiums_ff3['SMB'] +
            beta_hml * factor_premiums_ff3['HML']
        )

        cov_matrix_ff3 = returns[tickers_sorted].cov().values

        portfolio_weights_ff3 = optimize_portfolio(
            expected_excess_return_ff3,
            cov_matrix_ff3,
            tickers_sorted,
            risk_aversion
        )

        portfolio_weights_ff3['Weight (%)'] = (portfolio_weights_ff3['Weight'] * 100).round(2)
        print("\n✅ Optimized Portfolio Weights (FF3):")
        print(portfolio_weights_ff3[['Ticker', 'Weight (%)']])

        plot_portfolio_pie(portfolio_weights_ff3, "FF3")

    else:
        print("⚠️ Pass FF3 Portfolio Optimization.")

    # ✅ FF5 Optimization
    run_ff5 = input("\nCalculate Optimize Portfolio Using FF5 Model? (y/n): ").strip().lower()
    if run_ff5 == 'y':
        long_term = input('Calculate for long-term mean factor premium for FF5? (y/n): ').strip().lower() == 'y'

        factor_premiums_ff5 = (
            merged_data_ff5[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].mean()
            if long_term else
            merged_data_ff5[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].iloc[-1]
        )

        tickers_sorted = sorted(coef_df_ff5.columns)
        coef_df_ff5_sorted = coef_df_ff5[tickers_sorted]

        alpha = coef_df_ff5_sorted.T['Alpha'].values
        beta_mkt = coef_df_ff5_sorted.T['Mkt-RF'].values
        beta_smb = coef_df_ff5_sorted.T['SMB'].values
        beta_hml = coef_df_ff5_sorted.T['HML'].values
        beta_rmw = coef_df_ff5_sorted.T['RMW'].values
        beta_cma = coef_df_ff5_sorted.T['CMA'].values

        expected_excess_return_ff5 = (
            alpha +
            beta_mkt * factor_premiums_ff5['Mkt-RF'] +
            beta_smb * factor_premiums_ff5['SMB'] +
            beta_hml * factor_premiums_ff5['HML'] +
            beta_rmw * factor_premiums_ff5['RMW'] +
            beta_cma * factor_premiums_ff5['CMA']
        )

        cov_matrix_ff5 = returns[tickers_sorted].cov().values

        portfolio_weights_ff5 = optimize_portfolio(
            expected_excess_return_ff5,
            cov_matrix_ff5,
            tickers_sorted,
            risk_aversion
        )

        portfolio_weights_ff5['Weight (%)'] = (portfolio_weights_ff5['Weight'] * 100).round(2)
        print('\n✅ Optimized Portfolio Weights (FF5):')
        print(portfolio_weights_ff5[['Ticker', 'Weight (%)']])

        plot_portfolio_pie(portfolio_weights_ff5,'FF5')

    else:
        print('⚠️ Pass FF5 Portfolio Optimization.')

else:
    print('⚠️ Not enough tickers to perform optimization.')


# FF3 Portfolio Returns
portfolio_returns_ff3 = compute_portfolio_returns(portfolio_weights_ff3, returns) \
    if 'portfolio_weights_ff3' in locals() else None

# FF5 Portfolio Returns
portfolio_returns_ff5 = compute_portfolio_returns(portfolio_weights_ff5, returns) \
    if 'portfolio_weights_ff5' in locals() else None

#Backtesting Portfolio with Optional S&P 500 Benchmark

# ✅ User prompt for benchmark inclusion
use_benchmark = input('Benchmark with S&P500 for backtesting? (y/n): ').strip().lower() == 'y'

# ✅ Compute FF3 Portfolio Returns if available
portfolio_returns_ff3 = compute_portfolio_returns(portfolio_weights_ff3, returns) \
    if 'portfolio_weights_ff3' in locals() else None

# ✅ Compute FF5 Portfolio Returns if available
portfolio_returns_ff5 = compute_portfolio_returns(portfolio_weights_ff5, returns) \
    if 'portfolio_weights_ff5' in locals() else None

# ✅ Compute cumulative returns for available portfolios
cum_ff3 = compute_cumulative_returns(portfolio_returns_ff3) if portfolio_returns_ff3 is not None else None
cum_ff5 = compute_cumulative_returns(portfolio_returns_ff5) if portfolio_returns_ff5 is not None else None

# ✅ If benchmark selected, download and compute benchmark returns
if use_benchmark:
    snp = yf.download('^GSPC', start=start_date, end=end_date, interval='1d', auto_adjust=True)
    snp_monthly = snp['Close'].resample('ME').last()
    snp_returns = snp_monthly.pct_change().shift(-1).dropna()
    snp_returns.name = 'S&P 500'
    cum_bench = compute_cumulative_returns(snp_returns)
else:
    cum_bench = None

# ✅ Plot results
plt.figure(figsize=(14, 6))

if cum_ff3 is not None:
    plt.plot(cum_ff3, label='FF3 Optimized Portfolio', color='green')

if cum_ff5 is not None:
    plt.plot(cum_ff5, label='FF5 Optimized Portfolio', color='blue')

if cum_bench is not None:
    plt.plot(cum_bench, label='Benchmark (S&P 500)', color='black', linestyle='--')

# ✅ Dynamic title based on availability
if cum_bench is not None and cum_ff3 is not None and cum_ff5 is not None:
    title = ' FF3 vs FF5 vs S&P 500 Cumulative Performance'
elif cum_bench is not None and cum_ff3 is not None:
    title = ' FF3 vs S&P 500 Cumulative Performance'
elif cum_bench is not None and cum_ff5 is not None:
    title = ' FF5 vs S&P 500 Cumulative Performance'
elif cum_ff3 is not None and cum_ff5 is not None:
    title = ' FF3 vs FF5 Cumulative Performance'
elif cum_ff3 is not None:
    title = ' FF3 Cumulative Performance'
elif cum_ff5 is not None:
    title = ' FF5 Cumulative Performance'
else:
    title = ' No Portfolio Data Available for Plotting'

plt.title(title)
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()

# ✅ Clean title for filename
filename = title.replace("📈", "").replace("⚠️", "").replace(" ", "_").replace("(", "").replace(")", "").replace("-", "").lower() + ".png"
save_path = f"Data/output/plot/{filename}"

plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show(block=False)
print(f"✅ Cumulative plot saved as '{save_path}'")
plt.close()

dd_results = []

if cum_ff3 is not None:
    dd_results.append(calculate_drawdown_duration(cum_ff3, label="FF3 Optimized Portfolio"))

# FF5
if cum_ff5 is not None:
    dd_results.append(calculate_drawdown_duration(cum_ff5, label="FF5 Optimized Portfolio"))

# Benchmark
if cum_bench is not None:
    dd_results.append(calculate_drawdown_duration(cum_bench, label="Benchmark (S&P 500)"))
dd_df = pd.DataFrame(dd_results, columns=["Model", "Max Drawdown Duration (months)", "Avg Drawdown Duration (months)"])

plot_drawdown_duration_comparison(dd_df)

# ✅ Safe drawdown plotting block with conditional checks

# FF3 Drawdown
if 'cum_ff3' in locals() and cum_ff3 is not None:
    plot_drawdown_series(cum_ff3, label='FF3 Optimized Portfolio')
else:
    print("⚠️ FF3 cumulative returns not available, skipping FF3 drawdown plot.")

# FF5 Drawdown
if 'cum_ff5' in locals() and cum_ff5 is not None:
    plot_drawdown_series(cum_ff5, label='FF5 Optimized Portfolio')
else:
    print("⚠️ FF5 cumulative returns not available, skipping FF5 drawdown plot.")

# Benchmark Drawdown
if 'cum_bench' in locals() and cum_bench is not None:
    plot_drawdown_series(cum_bench, label='Benchmark (S&P 500)')
else:
    print("⚠️ Benchmark cumulative returns not available, skipping benchmark drawdown plot.")

# 🚀 Extend FF3 and FF5 metrics table with user input for risk-free rate
# Get user input safely with default fallback
try:
    risk_free_rate = float(input("Enter your risk-free rate (default = 0.03): ") or 0.03)
except ValueError:
    print("Invalid input. Using default risk-free rate = 0.03")
    risk_free_rate = 0.03

# Compute metrics with the provided risk-free rate
ff3_metrics = calculate_performance_metrics(cum_ff3, risk_free_rate=risk_free_rate)
ff5_metrics = calculate_performance_metrics(cum_ff5, risk_free_rate=risk_free_rate)

# Display cleanly
ff3_metrics=ff3_metrics.rename('FF3 Optimized Portfolio').to_frame().T 
display(ff3_metrics)
ff5_metrics=ff5_metrics.rename('FF5 Optimized Portfolio').to_frame().T
display(ff5_metrics)

# Rolling Sharpe plots
cum_returns_dict = {}
if 'cum_ff3' in locals():
    cum_returns_dict['FF3 Optimized Portfolio'] = cum_ff3
if 'cum_ff5' in locals():
    cum_returns_dict['FF5 Optimized Portfolio'] = cum_ff5

#FF3 Efficient Frontier Plotting
opt_weights_ff3 = portfolio_weights_ff3['Weight'].values
expected_excess_return_ff3_series = pd.Series(expected_excess_return_ff3, index=tickers, name='Expected Excess Return FF3')
df_frontier_ff3 = plot_efficient_frontier_with_optimal(
    expected_excess_return_ff3_series, 
    cov_matrix_ff3, 
    opt_weights_ff3, 
    risk_free_rate,
    tickers=tickers, 
    model_name="FF3")

#FF5 Efficient Frontier Plotting
opt_weights_ff5 = portfolio_weights_ff5['Weight'].values
expected_excess_return_ff5_series = pd.Series(expected_excess_return_ff5, index=tickers, name='Expected Excess Return FF5')
df_frontier_ff5 = plot_efficient_frontier_with_optimal(
    expected_excess_return_ff5_series, 
    cov_matrix_ff5, 
    opt_weights_ff5, 
    risk_free_rate,
    tickers=tickers,
    model_name="FF5")

plot_rolling_sharpe_auto(cum_returns_dict, risk_free_rate)

# ✅ Check if there's data to simulate first
if cum_ff3 is not None or cum_ff5 is not None:
    run_gbm = input("Run Monte Carlo Simulation with GBM + Jump Diffusion Model? (y/n): ").strip().lower() == "y"

    if run_gbm:
        # Step 1: Simulation Parameters
        try:
            months = int(input("\n🗓️ Enter number of simulation months (e.g., 12): ").strip())
        except ValueError:
            print("❌ Invalid input. Using default of 12 months.")
            months = 12

        try:
            trials = int(input("\n🔁 Enter number of simulation trials (e.g., 10000): ").strip())
        except ValueError:
            print("❌ Invalid input. Using default of 10,000 trials.")
            trials = 10_000

        try:
            num_paths_to_plot = int(input("\n📈 Enter number of paths to plot (e.g., 100): ").strip())
        except ValueError:
            print("❌ Invalid input. Displaying all paths.")
            num_paths_to_plot = trials

        # Step 2: Handle Jump Parameters (Auto vs Manual)
        use_auto_jump = input("\n⚙️ Do you want to auto-detect jumps? (y/n): ").strip().lower()

        if use_auto_jump == "y":
            sigma_cutoff_input = input("Input your sigma cutoff (default = 1.5): ").strip()
            sigma_cutoff = float(sigma_cutoff_input) if sigma_cutoff_input else 1.5
        else:
            print("\n📌 Manual Jump Parameter Input Mode")
            print("╭──────────────────────────────────────────────────────────────────╮")
            print("│ Parameter   │ Typical Range       │ Interpretation               │")
            print("├─────────────┼─────────────────────┼──────────────────────────────┤")
            print("│ λ (lambda)  │ 1–4 / year          │ Jump frequency assumption    │")
            print("│             │                     │                              │")
            print("│ μⱼ (Jump)   │ -0.02 to -0.10      │ Downward jumps (e.g., crash) │")
            print("│             │ +0.01 to +0.05      │ Upward jumps (e.g., spike)   │")
            print("│             │                     │                              │")
            print("│ σⱼ (Vol)    │ 0.05 to 0.15        │ Volatility around jumps      │")
            print("╰──────────────────────────────────────────────────────────────────╯")

            avg_jump_input = input("Enter average jumps per year (e.g., 2.0). Leave blank or enter 0 for no jumps (Standard GBM model): ").strip()
            lambda_manual = float(avg_jump_input) if avg_jump_input else 0.0

            if lambda_manual > 0:
                mu_j_input = input("Enter average jump size: ").strip()
                sigma_j_input = input("Enter jump volatility: ").strip()

                mu_j = float(mu_j_input) if mu_j_input else -0.02
                sigma_j = float(sigma_j_input) if sigma_j_input else 0.10
                print("✅ Manual jump parameters accepted.")
            else:
                mu_j, sigma_j = 0.0, 0.0
                print("⚠️ No jumps will be simulated (λ = 0) → standard GBM model.")

        # Step 3: FF3 Simulation
        if cum_ff3 is not None:
            print("\n🚀 Running simulation for FF3 Portfolio")
            cum_ff3_df = cum_ff3.to_frame(name="Return")

            if use_auto_jump == "y":
                drift_ff3, sigma_ff3, total_months_ff3, total_jumps_ff3, jump_df_ff3, lambda_ff3, mu_j_ff3, sigma_j_ff3 = compute_modeling_variable_auto(cum_ff3, sigma_cutoff)
            else:
                drift_ff3, sigma_ff3, total_months_ff3, expected_jumps_ff3, jump_df_ff3, lambda_ff3, mu_j_ff3, sigma_j_ff3 = compute_modeling_variable_manual(cum_ff3, lambda_manual, mu_j, sigma_j)

            if lambda_ff3 < 0.1:
                print("⚠️ Jump rate too low. Simulating FF3 with standard GBM.")
                mu_j_ff3, sigma_j_ff3 = 0.0, 0.0

            simulated_paths_ff3 = simulate_price_paths(cum_ff3_df, drift_ff3, sigma_ff3, months, trials, lambda_ff3, mu_j_ff3, sigma_j_ff3)
            simulated_df_ff3 = pd.DataFrame(simulated_paths_ff3)
            plot_price_paths("FF3", simulated_paths_ff3, num_paths_to_plot)
        else:
            print("⚠️ FF3 data not available. Skipping.")

        # Step 4: FF5 Simulation
        if cum_ff5 is not None:
            print("\n🚀 Running simulation for FF5 Portfolio")
            cum_ff5_df = cum_ff5.to_frame(name="Return")

            if use_auto_jump == "y":
                drift_ff5, sigma_ff5, total_months_ff5, total_jumps_ff5, jump_df_ff5, lambda_ff5, mu_j_ff5, sigma_j_ff5 = compute_modeling_variable_auto(cum_ff5, sigma_cutoff)
            else:
                drift_ff5, sigma_ff5, total_months_ff5, expected_jumps_ff5, jump_df_ff5, lambda_ff5, mu_j_ff5, sigma_j_ff5 = compute_modeling_variable_manual(cum_ff5, lambda_manual, mu_j, sigma_j)

            if lambda_ff5 < 0.1:
                print("⚠️ Jump rate too low. Simulating FF5 with standard GBM.")
                mu_j_ff5, sigma_j_ff5 = 0.0, 0.0

            simulated_paths_ff5 = simulate_price_paths(cum_ff5_df, drift_ff5, sigma_ff5, months, trials, lambda_ff5, mu_j_ff5, sigma_j_ff5)
            simulated_df_ff5 = pd.DataFrame(simulated_paths_ff5)
            plot_price_paths("FF5", simulated_paths_ff5, num_paths_to_plot)
        else:
            print("⚠️ FF5 data not available. Skipping.")
    else:
        print("❌ Skipping Monte Carlo Simulation.")
else:
    print("❌ No FF3 or FF5 portfolio data available. Nothing to simulate.")


#Saving summmary df output
summary_df_ff3, summary_df_ff5, summary_df_manual = None, None, None

if use_auto_jump == "y":
    if cum_ff3 is not None:
        summary_df_ff3 = pd.DataFrame({
            "Metric": ["Threshold (σ-cutoff)", "Timeframe (years)", "Timeframe (months)", "Total Jumps", "λ (Jump Intensity)", "μⱼ (Jump Size)", "σⱼ (Jump Volatility)"],
            "Value": [
                round(sigma_cutoff * cum_ff3_df["Return"].std(), 4),
                round(total_months_ff3 / 12, 2),
                total_months_ff3,
                total_jumps_ff3,
                round(lambda_ff3, 4),
                round(mu_j_ff3, 4),
                round(sigma_j_ff3, 4)
            ]
        })

    if cum_ff5 is not None:
        summary_df_ff5 = pd.DataFrame({
            "Metric": ["Threshold (σ-cutoff)", "Timeframe (years)", "Timeframe (months)", "Total Jumps", "λ (Jump Intensity)", "μⱼ (Jump Size)", "σⱼ (Jump Volatility)"],
            "Value": [
                round(sigma_cutoff * cum_ff5_df["Return"].std(), 4),
                round(total_months_ff5 / 12, 2),
                total_months_ff5,
                total_jumps_ff5,
                round(lambda_ff5, 4),
                round(mu_j_ff5, 4),
                round(sigma_j_ff5, 4)
            ]
        })
else:
    # Manual mode applies to both if present
    if cum_ff3 is not None:
        summary_df_manual = pd.DataFrame({
            "Metric": ["Drift", "λ (Jump Intensity)", "μⱼ (Jump Size)", "σⱼ (Jump Volatility)", "Expected Jumps", "Timeframe (years)", "Timeframe (months)"],
            "Value": [
                round(drift_ff3, 4),
                lambda_ff3,
                mu_j_ff3,
                sigma_j_ff3,
                round(expected_jumps_ff3, 2),
                round(total_months_ff3 / 12, 2),
                total_months_ff3
            ]
        })
    elif cum_ff5 is not None:
        summary_df_manual = pd.DataFrame({
            "Metric": ["Drift", "λ (Jump Intensity)", "μⱼ (Jump Size)", "σⱼ (Jump Volatility)", "Expected Jumps", "Timeframe (years)", "Timeframe (months)"],
            "Value": [
                round(drift_ff5, 4),
                lambda_ff5,
                mu_j_ff5,
                sigma_j_ff5,
                round(expected_jumps_ff5, 2),
                round(total_months_ff5 / 12, 2),
                total_months_ff5
            ]
        })

# 📤 Optional Display/Save
if summary_df_ff3 is not None:
    print("\n📘 FF3 Jump Summary")
    print(summary_df_ff3.to_markdown(index=False))

if summary_df_ff5 is not None:
    print("\n📙 FF5 Jump Summary")
    print(summary_df_ff5.to_markdown(index=False))

if summary_df_manual is not None:
    print("\n📗 Manual Jump Summary")
    print(summary_df_manual.to_markdown(index=False))

# Ask user whether to save outputs
if input("Save all outputs to CSV in Data/output/data? (y/n): ").strip().lower() == 'y':

    # List of potential outputs to save with friendly names
    outputs = {
        "merged_data_ff3.csv": "merged_data_ff3",
        "merged_data_ff5.csv": "merged_data_ff5",
        "coef_df_ff3.csv": "coef_df_ff3",
        "coef_df_ff5.csv": "coef_df_ff5",
        "portfolio_weights_ff3.csv": "portfolio_weights_ff3",
        "portfolio_weights_ff5.csv": "portfolio_weights_ff5",
        "portfolio_returns_ff3.csv": "portfolio_returns_ff3",
        "portfolio_returns_ff5.csv": "portfolio_returns_ff5",
        "cum_ff3.csv": "cum_ff3",
        "cum_ff5.csv": "cum_ff5",
        "benchmark_returns.csv": "snp_returns",
        "ff3_metrics.csv": "ff3_metrics",
        "ff5_metrics.csv": "ff5_metrics",
        "dd_df.csv": "dd_df",
        "comparison_df.csv": "comparison_df",
        "df_frontier_ff5.csv":"df_frontier_ff5",
        "df_frontier_ff3.csv":"df_frontier_ff3"
    }

    for filename, var_name in outputs.items():
        try:
            var_value = globals().get(var_name, None)
            if var_value is not None:
                if isinstance(var_value, (pd.DataFrame, pd.Series)):
                    file_path = "Data/output/data/" + filename
                    var_value.to_csv(file_path, index=True)
                    print(f"✅ {filename} saved to 'Data/output/data'.")
                else:
                    print(f"⚠️ {filename} is not a DataFrame/Series, skipping.")
            else:
                print(f"⚠️ {filename} not available, skipping.")
        except Exception as e:
            print(f"❌ Error saving {filename}: {e}")

else:
    print("❌ Save cancelled.")

