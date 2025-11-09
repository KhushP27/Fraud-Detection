import numpy as np
import pandas as pd
import yfinance as yf
from typing import List

# --- config ---
TICKERS = [
    "LCID","RIVN","NKLA","NIO","XPEV","LI","CHPT","RUN","BLNK","PLUG",
    "QS","SOFI","UPST","AFRM","COIN","HOOD","RIOT","MARA","HUT","CIFR",
    "MSTR","BITF","NVCR","NVAX","BLUE","SANA","DNA","BEAM","CRSP","EDIT",
    "BNTX","MRNA","PLTR","SNAP","BMBL","PATH","AI","BIGC","BILL","APP",
    "ZI","BYND","TLRY","CGC","RKT","ROKU","DKNG","CVNA","PTON","DASH",
    "GPRO","SPCE","FSR","FUBO","WISH","UPWK","ETSY","SE","BILI","IQ",
    "PDD","SHOP","PINS","NET","DDOG","CRWD","PENN","NCLH","CCL","AAL",
    "UBER","LYFT","IONQ","SOUN","ARM","CELH","TOST","CHWY","RBLX","TDOC",
    "OPEN","Z","RDFN","GOGO","LC","SQ","PYPL","PACB","ILMN","CRON",
    "JOBY","ACHR","EVGO","COHR","ENVX","ALLO","MGNI","ROIV","RKLB","VKTX"
]
START = "2018-01-01"
INTERVAL = "1d"

PUMP_RET_WINDOW = 5
PUMP_RET_MIN = 0.25
VOL_Z_WINDOW = 60
VOL_Z_MIN = 2.0
LOOKAHEAD = 20
CRASH_THRESH = -0.35

FEATURE_COLS = ["ret_5d","vol_20","vol_z","ema_gap_20"]

# --- functions ---
def fetch_hist(tickers: List[str], start="2018-01-01", end=None, interval="1d") -> pd.DataFrame:
    frames = []
    for tk in tickers:
        t = yf.Ticker(tk)
        h = t.history(start=start, end=end, interval=interval, auto_adjust=False)
        if h.empty:
            print(f"[warn] no data for {tk}")
            continue
        h = h.reset_index().rename(columns=str.title)
        h["ticker"] = tk
        h["Date"] = pd.to_datetime(h["Date"]).dt.tz_localize(None)
        frames.append(h[["Date","ticker","Open","High","Low","Close","Adj Close","Volume"]])
    if not frames:
        raise RuntimeError("No data pulled")
    out = pd.concat(frames, ignore_index=True).sort_values(["ticker","Date"])
    return out.rename(columns={"Adj Close":"AdjClose"})

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker","Date"]).copy()
    g = df.groupby("ticker", group_keys=False)
    def f(x):
        x = x.copy()
        x["ret_1d"]  = x["AdjClose"].pct_change()
        x["ret_5d"]  = x["AdjClose"].pct_change(PUMP_RET_WINDOW)
        x["vol_20"]  = x["ret_1d"].rolling(20).std()
        vol_ma = x["Volume"].rolling(VOL_Z_WINDOW).mean()
        vol_sd = x["Volume"].rolling(VOL_Z_WINDOW).std()
        x["vol_z"]   = (x["Volume"] - vol_ma) / vol_sd
        ema20       = x["AdjClose"].ewm(span=20, adjust=False).mean()
        x["ema_gap_20"] = (x["AdjClose"] - ema20) / ema20
        return x.replace([np.inf, -np.inf], np.nan)
    return g.apply(f)

def mark_pumps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("ticker", group_keys=False)
    cond_price = g["AdjClose"].apply(lambda s: s.pct_change(PUMP_RET_WINDOW) >= PUMP_RET_MIN)
    cond_vol   = df["vol_z"] >= VOL_Z_MIN
    df["pump_event"] = (cond_price & cond_vol).astype(int)
    return df

def make_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("ticker", group_keys=False)
    # future minimum price over lookahead
    future_min = g["AdjClose"].apply(
        lambda s: pd.concat([s.shift(-k) for k in range(1, LOOKAHEAD+1)], axis=1).min(axis=1)
    )
    drawdown = (future_min - df["AdjClose"]) / df["AdjClose"]
    df["y"] = ((drawdown <= CRASH_THRESH) & (df["pump_event"] == 1)).astype(int)
    return df

def build_training_set(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # shift features to avoid leakage
    out[FEATURE_COLS] = out.groupby("ticker")[FEATURE_COLS].shift(1)
    # keep only the pump days
    training = out[(out["pump_event"] == 1) & out[FEATURE_COLS].notna().all(axis=1)].copy()
    return training[["Date","ticker","AdjClose","y"] + FEATURE_COLS]

# --- build pipeline ---
prices = fetch_hist(TICKERS, start=START)
prices = add_features(prices)
prices = mark_pumps(prices)
prices = make_labels(prices)
training_set = build_training_set(prices)

# --- show output ---
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", None)
print("Training Set sample:")
print(training_set.to_string(index=False))
print(f"\nTotal rows: {len(training_set):,}  Positives (y=1): {training_set['y'].sum():,}")
