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

# Plot Actual vs Fitted for each ticker
def ff3_fitted_plt(merged_data):
    sns.set_style("whitegrid")
    tickers = merged_data['Ticker'].unique()
    for ticker in tickers:
        group = merged_data[merged_data['Ticker'] == ticker].copy()
        if group.shape[0] < 5:
            print(f"⚠️ Skipping {ticker} — not enough data points.")
            continue

        group = group.sort_values(by='Mkt-RF')

        X = group[['Mkt-RF', 'SMB', 'HML']]
        X = sm.add_constant(X)
        y = group['Excess Return']
        model = sm.OLS(y, X).fit()

        plt.figure(figsize=(8, 5))
        plt.scatter(group['Mkt-RF'], y, label='Actual', alpha=0.7)
        plt.plot(group['Mkt-RF'], model.fittedvalues, color='red', label='Fitted', linewidth=2)
        plt.xlabel('Mkt-RF')
        plt.ylabel('Excess Return')
        plt.title(f'FF3 Regression: {ticker} (R² = {model.rsquared:.2f})')
        plt.legend()
        plt.tight_layout()
        plt.show()

#Plot weightage in Pie Chart
def plot_portfolio_pie(weights_df, title="Optimized Portfolio Allocation (FF3)"):
    # Filter zero weights
    filtered_df = weights_df[weights_df['Weight'] > 0].copy()
    labels = filtered_df['Ticker'] + ' (' + (filtered_df['Weight'] * 100).round(1).astype(str) + '%)'
    sizes = filtered_df['Weight'] * 100
    colors = plt.cm.Blues([0.4, 0.7])  # fixed shades of blue

    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts = ax.pie(
    sizes,
    labels=labels,
    autopct=None,  # Only 2 values returned
    startangle=90,
    textprops={'fontsize': 13, 'weight': 'bold'},
    wedgeprops=dict(width=0.4, edgecolor='white'),
    colors=colors
    )

    # Add center circle
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax.add_artist(centre_circle)

    # Add title
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Add legend (optional)
    ax.legend(wedges, filtered_df['Ticker'], title="Tickers", loc='lower center', bbox_to_anchor=(0.5, -0.15),
              ncol=len(labels), fontsize=11, title_fontsize=12, frameon=False)

    ax.axis('equal')
    plt.tight_layout()
    plt.show()


# Backtest the portfolio using historical returns
def backtest_portfolio(weights_df, returns_df, benchmark_returns=None):
    tickers = weights_df['Ticker']
    weights = weights_df['Weight'].values

    # Compute portfolio returns
    returns_selected = returns_df[tickers].dropna()
    portfolio_returns = returns_selected @ weights

    # Normalize both indexes to monthly period (year-month)
    portfolio_returns.index = portfolio_returns.index.to_period('M').to_timestamp()
    if benchmark_returns is not None:
        benchmark_returns.index = benchmark_returns.index.to_period('M').to_timestamp()

    cumulative_portfolio = (1 + portfolio_returns).cumprod()

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_portfolio, label="FF3 Optimized Portfolio", color='green')

    # Add benchmark if applicable
    if benchmark_returns is not None:
        if isinstance(benchmark_returns, pd.DataFrame):
            benchmark_returns = benchmark_returns.squeeze()

        benchmark_returns = benchmark_returns.dropna()
        common_index = cumulative_portfolio.index.intersection(benchmark_returns.index)

        if not common_index.empty:
            benchmark_aligned = benchmark_returns.loc[common_index]
            benchmark_cumulative = (1 + benchmark_aligned).cumprod()
            plt.plot(benchmark_cumulative, label='S&P 500 Benchmark', color='black', linestyle='--')
        else:
            print("⚠️ Warning: No overlapping dates between portfolio and benchmark after reindexing.")

    plt.title('Backtested Portfolio Performance')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return portfolio_returns, benchmark_returns if benchmark_returns is not None else None

# Plot drawdown
def plot_drawdown(returns, label="Drawdown", ax=None):
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))

    ax.plot(drawdown, label=label, color='red')
    ax.set_title(f"{label} Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True)
    return ax

#Relative Performance (Portfolio – Benchmark)
def plot_relative_performance(portfolio_ret, benchmark_ret):
    # Ensure aligned
    aligned = portfolio_ret.dropna().copy()
    benchmark_aligned = benchmark_ret.reindex(aligned.index).dropna()
    aligned = aligned.loc[benchmark_aligned.index]  # Align both

    # Compute relative cumulative return
    portfolio_cum = (1 + aligned).cumprod()
    benchmark_cum = (1 + benchmark_aligned).cumprod()
    relative = portfolio_cum - benchmark_cum

    plt.figure(figsize=(10, 4))
    plt.plot(relative, label='FF3 Portfolio - S&P 500', color='purple')
    plt.axhline(0, linestyle='--', color='grey')
    plt.title("Relative Performance (FF3 - S&P 500)")
    plt.xlabel("Date")
    plt.ylabel("Excess Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()