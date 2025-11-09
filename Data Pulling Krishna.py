import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from typing import List

def fetch_hist_one(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    t = yf.Ticker(ticker)
    hist = t.history(start=start, end=end, interval=interval, auto_adjust=False)
    if hist.empty:
        return pd.DataFrame()
    hist = hist.reset_index().rename(columns=str.title)  # Date, Open, High, Low, Close, Adj Close, Volume, Dividends, Stock Splits
    hist["ticker"] = ticker
    # make date tz naive for consistency
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
    return hist[["Date","ticker","Open","High","Low","Close","Adj Close","Volume"]]

def fetch_hist_many(tickers: List[str], start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    frames = []
    for tk in tickers:
        df = fetch_hist_one(tk, start, end, interval)
        if df.empty:
            print(f"skipped empty: {tk}")
            continue
        frames.append(df)
    if not frames:
        raise RuntimeError("no data pulled")
    out = pd.concat(frames, ignore_index=True).sort_values(["ticker","Date"])
    return out.rename(columns={"Adj Close":"AdjClose"})

tickers = ["GME","USAR"]
prices = fetch_hist_many(tickers, start="2018-01-01", end=None)

print(prices.head())          # first 5 rows
print(prices.tail())          # last 5 rows
print(prices.shape)           # (rows, cols)
print(prices.columns.tolist())
