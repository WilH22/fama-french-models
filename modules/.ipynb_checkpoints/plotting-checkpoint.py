# plotting.py - FF3/FF5 Portfolio Analysis Module

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
import cvxpy as cp
import seaborn as sns
import statsmodels.api as sm
from IPython.display import display

def display_regression_parameters(coef_df, model_type):
    print(f"\n📊 {model_type} Regression Parameters:")
    display(
        coef_df.round(4).style
        .format("{:.4f}")
        .background_gradient(cmap='Blues')
        .set_caption(f"{model_type} Regression Coefficients and Adjusted R²")
    )

def ff_fitted_plt(merged_data, model_type):
    sns.set_style("whitegrid")

    if model_type == 'FF5':
        factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    else:
        factors = ['Mkt-RF', 'SMB', 'HML']

    tickers = merged_data['Ticker'].unique()
    for ticker in tickers:
        group = merged_data[merged_data['Ticker'] == ticker].copy()
        if group.shape[0] < 5:
            print(f"⚠️ Skipping {ticker} — not enough data points.")
            continue

        group = group.sort_values(by='Mkt-RF')  # For consistent plotting

        X = group[factors]
        X = sm.add_constant(X)
        y = group['Excess Return']
        model = sm.OLS(y, X).fit()

        plt.figure(figsize=(8, 5))
        plt.scatter(group['Mkt-RF'], y, label='Actual', alpha=0.7)
        plt.plot(group['Mkt-RF'], model.fittedvalues, color='red', label='Fitted', linewidth=2)
        plt.xlabel('Mkt-RF')
        plt.ylabel('Excess Return')
        plt.title(f'{model_type} Regression: {ticker} (R² = {model.rsquared:.2f})')
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)

def plot_portfolio_pie(weights_df, model_type):
    # Filter out zero-weight tickers
    filtered_df = weights_df[weights_df['Weight'] > 0].copy()
    labels = filtered_df['Ticker'] + ' (' + (filtered_df['Weight'] * 100).round(1).astype(str) + '%)'
    sizes = filtered_df['Weight'] * 100

    # Dynamic color gradient based on number of assets
    cmap = plt.cm.Blues
    colors = cmap(np.linspace(0.4, 0.8, len(filtered_df)))

    # Initialize plot
    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 13, 'weight': 'bold'},
        wedgeprops=dict(width=0.4, edgecolor='white'),
        colors=colors
    )

    # Add center circle for donut style
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax.add_artist(centre_circle)

    # Add title with model type
    ax.set_title(f"Optimized Portfolio Allocation ({model_type})",
                 fontsize=16, fontweight='bold', pad=20)

    # Add legend below the plot
    ax.legend(
        wedges,
        filtered_df['Ticker'],
        title="Tickers",
        loc='lower center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=min(len(labels), 4),
        fontsize=11,
        title_fontsize=12,
        frameon=False
    )

    # Ensure circular aspect ratio
    ax.axis('equal')
    plt.tight_layout()
    plt.show(block=False)

def compare_portfolios_with_benchmark(
    portfolio_returns_ff3=None,
    portfolio_returns_ff5=None,
    benchmark_returns=None,
    label_ff3='FF3',
    label_ff5='FF5',
    benchmark_label='S&P 500'
):
    has_ff3 = portfolio_returns_ff3 is not None
    has_ff5 = portfolio_returns_ff5 is not None
    has_benchmark = benchmark_returns is not None

    if not has_ff3 and not has_ff5:
        print("⚠️ No portfolio returns provided. Skipping plot.")
        return

    plt.figure(figsize=(14, 6))

    if has_benchmark:
        if has_ff3:
            common_idx = portfolio_returns_ff3.index.intersection(benchmark_returns.index)
            ff3_cum = (1 + portfolio_returns_ff3.loc[common_idx]).cumprod()
            bench_cum = (1 + benchmark_returns.loc[common_idx]).cumprod()
            plt.plot(ff3_cum, label=f'{label_ff3} Optimized Portfolio', color='green')
        
        if has_ff5:
            common_idx = portfolio_returns_ff5.index.intersection(benchmark_returns.index)
            ff5_cum = (1 + portfolio_returns_ff5.loc[common_idx]).cumprod()
            bench_cum = (1 + benchmark_returns.loc[common_idx]).cumprod()
            plt.plot(ff5_cum, label=f'{label_ff5} Optimized Portfolio', color='blue')

        # Plot benchmark only once
        plt.plot(bench_cum, label=benchmark_label, color='black', linestyle='--')

        plt.title(f"📈 Portfolio(s) vs {benchmark_label} Comparison")

    else:
        if has_ff3 and has_ff5:
            # Compare FF3 vs FF5 directly
            common_idx = portfolio_returns_ff3.index.intersection(portfolio_returns_ff5.index)
            ff3_cum = (1 + portfolio_returns_ff3.loc[common_idx]).cumprod()
            ff5_cum = (1 + portfolio_returns_ff5.loc[common_idx]).cumprod()
            plt.plot(ff3_cum, label=f'{label_ff3} Optimized Portfolio', color='green')
            plt.plot(ff5_cum, label=f'{label_ff5} Optimized Portfolio', color='blue')
            plt.title("📈 FF3 vs FF5 Portfolio Performance")
        elif has_ff3:
            ff3_cum = (1 + portfolio_returns_ff3).cumprod()
            plt.plot(ff3_cum, label=f'{label_ff3} Optimized Portfolio', color='green')
            plt.title(f"📈 {label_ff3} Portfolio Performance")
        elif has_ff5:
            ff5_cum = (1 + portfolio_returns_ff5).cumprod()
            plt.plot(ff5_cum, label=f'{label_ff5} Optimized Portfolio', color='blue')
            plt.title(f"📈 {label_ff5} Portfolio Performance")

    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)

def plot_relative_performance(portfolio_ret, benchmark_ret, model_type, benchmark_name="S&P 500"):
    # Align dates and drop missing values
    aligned = portfolio_ret.dropna().copy()
    benchmark_aligned = benchmark_ret.reindex(aligned.index).dropna()
    aligned = aligned.loc[benchmark_aligned.index]  # Align both to same dates

    # Compute cumulative returns
    portfolio_cum = (1 + aligned).cumprod()
    benchmark_cum = (1 + benchmark_aligned).cumprod()

    # Compute relative performance
    relative = portfolio_cum - benchmark_cum

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(relative, label=f"{model_type} Portfolio - {benchmark_name}", color='purple')
    plt.axhline(0, linestyle='--', color='grey')
    plt.title(f"Relative Performance ({model_type} - {benchmark_name})")
    plt.xlabel("Date")
    plt.ylabel("Excess Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)

def plot_drawdown_duration_comparison(dd_df):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    models = dd_df["Model"]
    max_dd = dd_df["Max Drawdown Duration (months)"]
    avg_dd = dd_df["Avg Drawdown Duration (months)"]

    # Dynamically assign colors to match your project or default palette
    colors = ["green", "blue", "black"][:len(models)]

    # Plot Max Duration
    bars0 = ax[0].bar(models, max_dd, color=colors, alpha=0.8)
    ax[0].set_title("Max Drawdown Duration")
    ax[0].set_ylabel("Months")
    ax[0].grid(True, linestyle='--', alpha=0.5)
    ax[0].set_ylim(0, max(max_dd) * 1.2)

    # Annotate values
    for bar in bars0:
        height = bar.get_height()
        ax[0].annotate(f'{height:.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)

    # Plot Average Duration
    bars1 = ax[1].bar(models, avg_dd, color=colors, alpha=0.8)
    ax[1].set_title("📉 Average Drawdown Duration")
    ax[1].set_ylabel("Months")
    ax[1].grid(True, linestyle='--', alpha=0.5)
    ax[1].set_ylim(0, max(avg_dd) * 1.2)

    # Annotate values
    for bar in bars1:
        height = bar.get_height()
        ax[1].annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)

    # Rotate x labels for clarity
    for axis in ax:
        axis.tick_params(axis='x', rotation=20)

    plt.suptitle("Drawdown Duration Comparison Across Portfolios", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show(block=False)

def plot_drawdown_series(cum_returns, label, color="crimson"):
    # Compute drawdown
    high_water_mark = cum_returns.cummax()
    drawdown = (cum_returns / high_water_mark - 1)*100

    # Ensure drawdown is 1D Series
    if isinstance(drawdown, pd.DataFrame):
        drawdown = drawdown.iloc[:, 0]

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(drawdown.index, drawdown, label=f"{label} Drawdown", color=color, lw=1.5)
    ax.fill_between(drawdown.index, drawdown.values, color=color, alpha=0.3)

    ax.set_title(f"Drawdown Series for {label}", fontsize=14, weight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Drawdown (%)", fontsize=12)

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    ax.set_ylim(drawdown.min() * 1.1, 0.01)  # slight margin for clarity

    plt.tight_layout()
    plt.show(block=False)


def plot_rolling_sharpe_auto(risk_free_rate):
    window = 12
    periods_per_year = 12
    returns_dict = {}

    # Auto-detect
    if 'cum_ff3' in globals() and isinstance(globals()['cum_ff3'], pd.Series):
        returns_dict['FF3 Optimized Portfolio'] = globals()['cum_ff3']
    if 'cum_ff5' in globals() and isinstance(globals()['cum_ff5'], pd.Series):
        returns_dict['FF5 Optimized Portfolio'] = globals()['cum_ff5']

    if not returns_dict:
        print("⚠️ No cumulative return series (cum_ff3 or cum_ff5) found in environment. Skipping plot.")
        return

    plt.figure(figsize=(12, 6))

    # Fixed colors
    color_map = {
        'FF3 Optimized Portfolio': 'green',
        'FF5 Optimized Portfolio': 'blue'
    }
    
    for label, returns in returns_dict.items():
        if returns is None or returns.empty:
            continue

        periodic_returns = returns.pct_change().dropna()
        excess_returns = periodic_returns - (risk_free_rate / periods_per_year)
        rolling_mean = excess_returns.rolling(window=window).mean()
        rolling_std = excess_returns.rolling(window=window).std(ddof=0)
        rolling_sharpe = (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(periods_per_year)

        plt.plot(
            rolling_sharpe.index,
            rolling_sharpe,
            label=f"{label} Rolling Sharpe ({window}-month)",
            color=color_map.get(label, None)  # default to auto if unknown
        )

    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.title(f"Rolling Sharpe Ratio ({window}-month)")
    plt.xlabel("Date")
    plt.ylabel("Sharpe Ratio")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show(block=False)