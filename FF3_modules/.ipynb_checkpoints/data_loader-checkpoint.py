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

#Fetching Close and Return data
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

    data = yf.download(tickers, start=start, end=end, group_by='ticker', interval='1mo', auto_adjust=True)

    # Handle both single and multi-level column format
    if isinstance(data.columns, pd.MultiIndex):
        try:
            adj_close = data.xs('Close', level=1, axis=1)
        except KeyError:
            raise ValueError("❌ 'Adj Close' prices not found — possibly all tickers failed.")
    else:
        if 'Close' in data.columns:
            adj_close = data['Close'].to_frame()
        else:
            raise ValueError("❌ No 'Adj Close' in single-column format.")

    adj_close = adj_close.dropna(axis=1, how='all')

    # Detect successful and failed tickers
    successful = list(adj_close.columns)
    failed = [t for t in tickers if t not in successful]

    if not successful:
        raise ValueError("❌ All tickers failed to download.")

    if failed:
        print(f"⚠️ The following tickers failed to download (check for typos or delisting): {failed}")

    returns = adj_close.pct_change().dropna()
    return adj_close, returns

