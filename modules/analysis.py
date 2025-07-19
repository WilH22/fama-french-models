# analysis.py - FF3/FF5 Portfolio Analysis Module
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

def fetch_data(tickers, start=None, end=None):
    # Validate start date
    if not start:
        raise ValueError("❌ Start date is required in 'YYYY-MM-DD' format.")
    try:
        datetime.strptime(start, "%Y-%m-%d")
    except ValueError:
        raise ValueError("❌ Start date must be in 'YYYY-MM-DD' format (e.g., 2020-01-01).")

    # Validate end date if provided
    if end:
        try:
            datetime.strptime(end, "%Y-%m-%d")
        except ValueError:
            raise ValueError("❌ End date must be in 'YYYY-MM-DD' format (e.g., 2020-12-31).")

    # Download monthly data using yfinance
    data = yf.download(
        tickers,
        start=start,
        end=end,
        group_by='ticker',
        interval='1mo',
        auto_adjust=True
    )

    # Handle data structure: single ticker vs multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        # MultiIndex columns: extract 'Close' prices for all tickers
        try:
            adj_close = data.xs('Close', level=1, axis=1)
        except KeyError:
            raise ValueError("❌ 'Close' prices not found — possibly all tickers failed.")
    else:
        # Single ticker: ensure 'Close' is present
        if 'Close' in data.columns:
            adj_close = data['Close'].to_frame()
        else:
            raise ValueError("❌ No 'Close' column found in data.")

    # Drop columns with all NaN values (fully failed tickers)
    adj_close = adj_close.dropna(axis=1, how='all')

    # Identify successfully fetched tickers vs failed tickers
    successful = list(adj_close.columns)
    failed = [t for t in tickers if t not in successful]

    if not successful:
        raise ValueError("❌ All tickers failed to download.")

    if failed:
        print(f"⚠️ The following tickers failed to download (check spelling or delisting): {failed}")

    # Calculate monthly percentage returns
    returns = adj_close.pct_change().dropna()

    return adj_close, returns

def ff_regression_analysis(merged_data, model_type):
    if model_type == 'FF5':
        factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    else:
        factors = ['Mkt-RF', 'SMB', 'HML']

    alpha_list = []
    stderr_alpha = []
    adj_r_squared_list = []
    factor_lists = {factor: [] for factor in factors}
    stderr_lists = {factor: [] for factor in factors}

    tickers = merged_data['Ticker'].unique()

    for ticker in tickers:
        group = merged_data[merged_data['Ticker'] == ticker]
        X = group[factors]
        X = sm.add_constant(X)
        y = group['Excess Return']

        model = sm.OLS(y, X).fit()
        print(f"\n📊 {model_type} Regression Summary for {ticker}")
        print(model.summary(), end="\n\n")

        alpha_list.append(model.params['const'])
        stderr_alpha.append(model.bse['const'])
        adj_r_squared_list.append(model.rsquared_adj)

        for factor in factors:
            factor_lists[factor].append(model.params[factor])
            stderr_lists[factor].append(model.bse[factor])

    data = {
        'Ticker': tickers,
        'Alpha': alpha_list,
        'Alpha_err': stderr_alpha,
        'Adj_R2': adj_r_squared_list
    }
    for factor in factors:
        data[factor] = factor_lists[factor]
        data[factor + '_err'] = stderr_lists[factor]

    coef_df = pd.DataFrame(data).set_index('Ticker')

    coef_columns = ['Alpha'] + factors
    err_columns = ['Alpha_err'] + [f + '_err' for f in factors]

    coef_df[coef_columns].plot(
        kind='bar',
        yerr=coef_df[err_columns].values.T,
        capsize=4,
        figsize=(12, 6)
    )
    plt.title(f'Fama-French {model_type}-Factor Coefficients with Error Bars')
    plt.ylabel('Coefficient Value')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Data/output/plot/Fama-French_{model_type}.png", dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.close()

    return model, coef_df.T

def optimize_portfolio(expected_excess_return, cov_matrix, tickers, risk_aversion, allow_short=False):
    n = len(tickers)
    w = cp.Variable(n)

    # Objective: maximize expected return minus risk penalty
    objective = cp.Maximize(expected_excess_return @ w - risk_aversion * cp.quad_form(w, cov_matrix))

    # Constraints
    constraints = [cp.sum(w) == 1]
    if not allow_short:
        constraints.append(w >= 0)

    prob = cp.Problem(objective, constraints)
    prob.solve()

    if prob.status != 'optimal':
        raise ValueError(f"Optimization did not converge: {prob.status}")

    # Clean small weights and package
    weights = np.where(np.abs(w.value) < 1e-5, 0, w.value.round(6))
    return pd.DataFrame({'Ticker': tickers, 'Weight': weights})

def compute_portfolio_returns(weights_df, returns_df):
    tickers = weights_df['Ticker']
    weights = weights_df['Weight'].values

    # Ensure the returns_df contains required tickers
    missing_tickers = [t for t in tickers if t not in returns_df.columns]
    if missing_tickers:
        raise ValueError(f"❌ Missing tickers in returns_df: {missing_tickers}")

    returns_subset = returns_df[tickers].dropna()
    portfolio_returns = returns_subset @ weights

    return portfolio_returns

def compute_cumulative_returns(returns_series):
    if returns_series is None:
        return None
    return (1 + returns_series).cumprod()

def performance_summary(returns, label="Portfolio"):
    # Calculate cumulative returns
    cumulative = (1 + returns).cumprod()

    # Total return over the period
    total_return = cumulative.iloc[-1] - 1

    # Annualized return using CAGR formula
    annualized_return = (1 + total_return) ** (12 / len(returns)) - 1

    # Annualized volatility
    annualized_volatility = returns.std() * (12 ** 0.5)

    # Sharpe Ratio (risk-free rate assumed to be zero)
    sharpe_ratio = annualized_return / annualized_volatility

    # Maximum drawdown
    max_drawdown = ((cumulative / cumulative.cummax()) - 1).min()

    # Prepare display summary with clear formatting
    summary_display = {
        "Total Return": f"{total_return:.2%}",
        "Annualized Return": f"{annualized_return:.2%}",
        "Annualized Volatility": f"{annualized_volatility:.2%}",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}",
        "Max Drawdown": f"{max_drawdown:.2%}"
    }

    print(f"\n📊 Performance Summary: {label}")
    for metric, value in summary_display.items():
        print(f"{metric:<25}: {value}")

    # Return raw numeric values for further processing or table creation
    return {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown
    }

def compare_ff3_ff5(ff3_df, ff5_df, display_table=True):
    comparison_df = pd.DataFrame({
        'FF3_Adj_R2': ff3_df.T['Adj_R2'],
        'FF5_Adj_R2': ff5_df.T['Adj_R2'],
        'R2_Improvement': ff5_df.T['Adj_R2'] - ff3_df.T['Adj_R2'],
        'R2_%_Improvement': ((ff5_df.T['Adj_R2'] - ff3_df.T['Adj_R2']) / ff3_df.T['Adj_R2']) * 100,
        'FF3_Alpha': ff3_df.T['Alpha'],
        'FF5_Alpha': ff5_df.T['Alpha'],
        'Alpha_Reduction': ff5_df.T['Alpha'] - ff3_df.T['Alpha']
    })

    print('\n📈 Generating FF3 vs FF5 Comparison Summary...')

    # Bar chart for Adjusted R²
    comparison_df[['FF3_Adj_R2', 'FF5_Adj_R2']].plot(
        kind='bar', figsize=(12, 5),
        title='Adjusted R² Comparison: FF3 vs FF5',
        ylabel='Adjusted R²',
        colormap='viridis'
    )
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"Data/output/plot/FF3_vs_FF5_Comparison_Summary.png", dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.close()

    # Bar chart for Alpha
    comparison_df[['FF3_Alpha', 'FF5_Alpha']].plot(
        kind='bar', figsize=(12, 5),
        title='Alpha Comparison: FF3 vs FF5',
        ylabel='Alpha',
        colormap='plasma'
    )
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"Data/output/plot/Alpha_Comparison_FF3_vs_FF5.png", dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.close()

    if display_table:
        styled_table = (
            comparison_df.round(4)
            .style.format("{:.4f}")
            .background_gradient(cmap='Blues', subset=['FF3_Adj_R2', 'FF5_Adj_R2'])
            .background_gradient(cmap='RdYlGn', subset=['R2_Improvement', 'R2_%_Improvement'])
            .background_gradient(cmap='Greens', subset=['Alpha_Reduction'])
            .set_caption("📊 FF3 vs FF5 Comparison: Adjusted R², Alpha, and Improvements")
        )
        display(styled_table)

    return comparison_df

def analyze_ff3_ff5_comparison(comparison_df):
    # Calculate % improvement in Adjusted R²
    comparison_df['R2_%_Change'] = (comparison_df['R2_Improvement'] / comparison_df['FF3_Adj_R2']) * 100

    # Plot R² % improvement
    ax1 = comparison_df['R2_%_Change'].plot(kind='bar', figsize=(8, 5), color='skyblue', edgecolor='black')
    plt.title('Percentage Improvement in Adjusted R²: FF5 vs FF3')
    plt.ylabel('% Improvement')
    plt.xlabel('Ticker')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"Data/output/plot/R2_Percentage_Improvement_FF5_vs_FF3.png", dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.close()

    # Plot Alpha reduction
    ax2 = comparison_df['Alpha_Reduction'].plot(kind='bar', figsize=(8, 5), color='salmon', edgecolor='black')
    plt.title('Alpha Reduction: FF5 vs FF3')
    plt.ylabel('Alpha Difference (FF5 - FF3)')
    plt.xlabel('Ticker')
    plt.axhline(0, linestyle='--', color='gray', linewidth=1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"Data/output/plot/Alpha_Reduction_FF5_vs_FF3.png", dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.close()

    # Text summary per ticker
    print("\n📊 Summary Interpretation:")
    for ticker in comparison_df.index:
        r2_improv = comparison_df.loc[ticker, 'R2_%_Change']
        alpha_red = comparison_df.loc[ticker, 'Alpha_Reduction']
        print(f"{ticker}: Adj R² improved by {r2_improv:.2f}% and Alpha reduced by {alpha_red:.5f}.")

def calculate_drawdown_duration(cum_returns, label="Portfolio", period_label="months", verbose=True, return_df=False):
    if isinstance(cum_returns, pd.DataFrame):
        cum_returns = cum_returns.iloc[:, 0]

    high_water_mark = cum_returns.cummax()
    drawdown = cum_returns / high_water_mark - 1
    in_drawdown = drawdown < 0

    durations = []
    duration = 0
    for d in in_drawdown:
        if d:
            duration += 1
        elif duration != 0:
            durations.append(duration)
            duration = 0
    if duration != 0:
        durations.append(duration)

    max_duration = int(max(durations)) if durations else 0
    avg_duration = round(float(np.mean(durations)), 2) if durations else 0.0

    if verbose:
        print(f"📊 {label} Drawdown Duration Analysis")
        print(f"   • Max Drawdown Duration   : {max_duration} {period_label}")
        print(f"   • Average Drawdown Duration: {avg_duration:.2f} {period_label}\n")

    if return_df:
        result_df = pd.DataFrame({
            "Model": [label],
            "Max Drawdown Duration": [max_duration],
            "Avg Drawdown Duration": [avg_duration]
        })
        return result_df

    return label, max_duration, avg_duration


def calculate_performance_metrics(cum_returns, risk_free_rate):

    periods_per_year = 12

    returns = cum_returns.pct_change().dropna()

    # CAGR
    cagr = ((cum_returns.iloc[-1]) ** (1 / (len(cum_returns) / periods_per_year)) - 1) * 100

    # Volatility
    volatility = returns.std(ddof=1) * (periods_per_year ** 0.5) * 100

    # Sharpe
    excess_returns = returns - risk_free_rate / periods_per_year
    sharpe = excess_returns.mean() / returns.std(ddof=1) * (periods_per_year ** 0.5)

    # Sortino
    downside = returns.copy()
    downside[downside > 0] = 0
    downside_std = downside.std(ddof=1)
    sortino = (
        excess_returns.mean() / downside_std * (periods_per_year ** 0.5)
        if downside_std != 0 else np.nan
    )

    # Max Drawdown
    high_water = cum_returns.cummax()
    drawdown = cum_returns / high_water - 1
    max_drawdown = drawdown.min() * 100

    # Calmar
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan

    return pd.Series({
        "CAGR": round(cagr, 2),
        "Volatility": round(volatility, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Sortino Ratio": round(sortino, 2),
        "Max Drawdown": round(max_drawdown, 2),
        "Calmar Ratio": round(calmar, 2)
    })
    
def compute_modeling_variable_auto(cum_ff: pd.Series, sigma_cutoff):

    cum_ff = cum_ff.squeeze()
    log_returns = cum_ff.pct_change().apply(lambda x: np.log(1 + x)).dropna()

    mu = log_returns.mean()
    sigma = log_returns.std()
    var = log_returns.var()
    drift = mu - 0.5 * var

    threshold = sigma_cutoff * sigma
    jump_mask = (np.abs(log_returns - mu) > threshold)
    jump_returns = log_returns[jump_mask]

    total_months = len(log_returns)
    years = total_months / 12
    total_jumps = int(jump_mask.sum())
    lambda_ = total_jumps / years if years > 0 else 0

    mu_j = jump_returns.mean() if total_jumps > 0 else 0
    sigma_j = jump_returns.std() if total_jumps > 0 else 0

    jump_df = pd.DataFrame({
        "Date": log_returns.index,
        "LogAdjReturn": log_returns.values,
        "Jump": jump_mask.astype(int)
    })

    print(f"📊 Jump Detection Summary:")
    print(f" - Threshold (σ-cutoff): {threshold:.4f}")
    print(f" - Timeframe: {years:.2f} years ({total_months} months)")
    print(f" - Total Jumps Detected: {total_jumps}")

    return drift, sigma, total_months, total_jumps, jump_df, lambda_, mu_j, sigma_j

def compute_modeling_variable_manual(cum_ff: pd.Series, 
                                     lambda_: float, 
                                     mu_j: float, 
                                     sigma_j: float):
    
    cum_ff = cum_ff.squeeze()
    log_returns = cum_ff.pct_change().apply(lambda x: np.log(1 + x)).dropna()

    mu = log_returns.mean()
    sigma = log_returns.std()
    var = log_returns.var()
    drift = mu - 0.5 * var

    threshold = sigma_cutoff * sigma
    total_months = len(log_returns)
    years = total_months / 12
    expected_jumps = lambda_ * years  # for context, not detection

    jump_df = pd.DataFrame({
        "Date": log_returns.index,
        "LogAdjReturn": log_returns.values,
        "Jump": 0  # No jump detection used here
    })

    print(f"\n🛠️ Manual Jump Parameter Summary:")
    print(f" - Drift: {drift:.4f}")
    print(f" - λ (Jump Intensity): {lambda_}")
    print(f" - μⱼ (Jump Size): {mu_j}")
    print(f" - σⱼ (Jump Volatility): {sigma_j}")
    print(f" - Expected Jumps: {expected_jumps:.2f} over {years:.2f} years")

    return drift, sigma, total_months, expected_jumps, jump_df, lambda_, mu_j, sigma_j

def simulate_price_paths(data, drift, stdev, months, trials, lambda_, mu_j, sigma_j):

    dt = 1 / 12  # Monthly time step
    Z = norm.ppf(np.random.rand(months, trials))  # Normal random shocks
    N = np.random.poisson(lambda_ * dt, size=(months, trials))  # Number of jumps
    J = np.random.normal(mu_j, sigma_j, size=(months, trials))  # Jump magnitudes

    jumps = N * J
    monthly_returns = np.exp(drift * dt + stdev * np.sqrt(dt) * Z + jumps)

    # Initialize price paths
    price_paths = np.zeros((months, trials))
    price_paths[0] = data['Return'].iloc[-1]/data['Return'].iloc[-1]  # Start from last known price

    for t in range(1, months):
        price_paths[t] = price_paths[t - 1] * monthly_returns[t]

    return price_paths

