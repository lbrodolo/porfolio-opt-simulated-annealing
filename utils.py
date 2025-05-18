import numpy as np
import yfinance as yf
import pandas as pd

def get_market_data(tickers, start='2020-01-01', end='2024-01-01'):
    data = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=True)

    # Multi-ticker (MultiIndex)
    if isinstance(data.columns, pd.MultiIndex):
        close_prices = pd.concat(
            [data[ticker]['Close'] for ticker in tickers if ticker in data.columns.levels[0]],
            axis=1
        )
        close_prices.columns = [ticker for ticker in tickers if ticker in data.columns.levels[0]]
    elif 'Close' in data.columns:
        close_prices = data[['Close']]
        close_prices.columns = tickers[:1]
    else:
        raise ValueError("Format error.")

    returns = close_prices.pct_change().dropna()
    mu = returns.mean().values
    cov = returns.cov().values
    J = -cov
    return J, mu, close_prices.columns.tolist()

def energy(s, J, h):
    return -0.5 * s @ J @ s - h @ s
