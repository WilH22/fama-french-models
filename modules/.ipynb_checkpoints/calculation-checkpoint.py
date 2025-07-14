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

#Calculate weightage using FF3 Model
def optimize_portfolio_ff3_risk_adjusted(expected_excess_return, cov_matrix, tickers, risk_aversion=10):
    n = len(tickers)
    w = cp.Variable(n)

    # Objective: maximize return - risk penalty
    objective = cp.Maximize(expected_excess_return @ w - risk_aversion * cp.quad_form(w, cov_matrix))

    # Constraints: weights sum to 1, no shorting
    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    # Clean small weights and package as DataFrame
    weights = np.where(np.abs(w.value) < 1e-5, 0, w.value.round(4))
    return pd.DataFrame({'Ticker': tickers, 'Weight': weights})