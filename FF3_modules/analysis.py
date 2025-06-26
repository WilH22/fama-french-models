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

#Regression Summary and coefficient calculation
def ff3_regression_analysis(merged_data):
    # Initialize storage lists
    alpha_list, mkt_list, smb_list, hml_list = [], [], [], []
    stderr_alpha, stderr_mkt, stderr_smb, stderr_hml = [], [], [], []

    tickers = merged_data['Ticker'].unique()

    for ticker in tickers:
        group = merged_data[merged_data['Ticker'] == ticker]
        X = group[['Mkt-RF', 'SMB', 'HML']]
        X = sm.add_constant(X)
        y = group['Excess Return']

        model = sm.OLS(y, X).fit()
        print(f"\n📊 Regression Summary for {ticker}")
        print(model.summary(), end="\n\n")
        # Coefficients
        alpha_list.append(model.params['const'])
        mkt_list.append(model.params['Mkt-RF'])
        smb_list.append(model.params['SMB'])
        hml_list.append(model.params['HML'])

        # Standard errors
        stderr_alpha.append(model.bse['const'])
        stderr_mkt.append(model.bse['Mkt-RF'])
        stderr_smb.append(model.bse['SMB'])
        stderr_hml.append(model.bse['HML'])

    # Compile into DataFrame
    coef_df = pd.DataFrame({
        'Ticker': tickers,
        'Alpha': alpha_list,
        'Mkt-RF': mkt_list,
        'SMB': smb_list,
        'HML': hml_list,
        'Alpha_err': stderr_alpha,
        'Mkt-RF_err': stderr_mkt,
        'SMB_err': stderr_smb,
        'HML_err': stderr_hml
    }).set_index('Ticker')

    # Plot bar chart with error bars
    fig, ax = plt.subplots(figsize=(10, 6))
    coef_df[['Alpha', 'Mkt-RF', 'SMB', 'HML']].plot(
        kind='bar',
        yerr=coef_df[['Alpha_err', 'Mkt-RF_err', 'SMB_err', 'HML_err']].values.T,
        capsize=4,
        ax=ax
    )
    plt.title('Fama-French 3-Factor Coefficients with Error Bars')
    plt.ylabel('Coefficient Value')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return model,coef_df.T # Return the regression summary table 

#Performance Metrics Summary Table
def performance_summary(returns, label="Portfolio"):
    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    annualized_return = (1 + total_return)**(12 / len(returns)) - 1
    annualized_vol = returns.std() * (12 ** 0.5)
    sharpe_ratio = annualized_return / annualized_vol
    max_drawdown = ((cumulative / cumulative.cummax()) - 1).min()

    summary = {
        "Total Return": f"{total_return:.2%}",
        "Annualized Return": f"{annualized_return:.2%}",
        "Annualized Volatility": f"{annualized_vol:.2%}",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}",
        "Max Drawdown": f"{max_drawdown:.2%}"
    }

    print(f"\nPerformance Summary: {label}")
    for k, v in summary.items():
        print(f"{k:<25}: {v}")

    return {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_vol,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown
    }