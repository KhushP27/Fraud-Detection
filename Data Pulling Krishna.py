# ======================================================
#  PUMP & DUMP MACHINE LEARNING SCREENER  (robust columns)
# ======================================================

import yfinance as yf, numpy as np, pandas as pd
from datetime import datetime, timedelta, UTC
from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

print("🤖 Pump & Dump Stock Screener (ML Edition)\n")

LOOKBACK_DAYS = 365

PUMP_TICKERS = ["GNS","NVOS","AITX","BBAI","CXAI","CVNA","SOUN","VFS","NKLA","BBBY"]
NORMAL_TICKERS = ["AAPL","MSFT","AMZN","TSLA","NVDA","GOOG","META","PEP","KO","XOM"]

# ------------------------------------------------------
def normalize_df(df):
    """flatten MultiIndex and lowercase column names"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [a if isinstance(a,str) else a[0] for a in df.columns.get_level_values(0)]
    df.columns = [str(c).lower() for c in df.columns]
    # prefer adj close if present
    if "adj close" in df.columns and "close" not in df.columns:
        df["close"] = df["adj close"]
    return df

def build_features(ticker):
    try:
        end = datetime.now(UTC); start = end - timedelta(days=LOOKBACK_DAYS)
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            print(f"⚠️  Skipping {ticker} (no data)")
            return None
        df = normalize_df(df)
        if not {"close","open","volume"}.issubset(df.columns):
            print(f"⚠️  Skipping {ticker} (missing columns: {df.columns.tolist()})")
            return None

        df["ret"] = df["close"].pct_change()
        df["dollar_vol"] = df["close"]*df["volume"]
        try:
            df["rsi14"] = RSIIndicator(df["close"],window=14,fillna=True).rsi()
        except: df["rsi14"]=50
        df["gap"] = df["open"]/df["close"].shift(1)-1

        f = {}
        f["close_last"] = df["close"].iloc[-1]
        f["vol_med60"] = df["volume"].tail(60).median()
        f["ret5"] = (df["close"].iloc[-1]/df["close"].iloc[-5]-1) if len(df)>5 else 0
        f["vol_spike"] = (df["volume"].tail(3).mean()/(f["vol_med60"]+1))
        f["rsi_mean"] = df["rsi14"].tail(5).mean()
        f["gap_freq"] = (df["gap"].tail(20).abs()>0.1).mean()
        f["dollar_vol_med"] = df["dollar_vol"].tail(60).median()
        f["ret_vol"] = df["ret"].tail(60).std()
        return f
    except Exception as e:
        print(f"⚠️  Skipping {ticker} ({e})")
        return None

# ------------------------------------------------------
rows=[]
print("📈 Building training dataset...")
for tk in PUMP_TICKERS:
    f=build_features(tk)
    if f: f["label"]=1; f["ticker"]=tk; rows.append(f)
for tk in NORMAL_TICKERS:
    f=build_features(tk)
    if f: f["label"]=0; f["ticker"]=tk; rows.append(f)

df=pd.DataFrame(rows).dropna()
if df.empty:
    print("❌ No usable data fetched — check internet or ticker list.")
    raise SystemExit

print(f"✅ Dataset built with {len(df)} samples "
      f"({df['label'].sum()} pumps, {len(df)-df['label'].sum()} normals)\n")

# ------------------------------------------------------
X=df.drop(columns=["label","ticker"])
y=df["label"]
Xtr,Xte,Ytr,Yte=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
model=RandomForestClassifier(n_estimators=300,max_depth=6,class_weight="balanced",random_state=42)
model.fit(Xtr,Ytr)
print("🧠 Model trained!\n")
print(classification_report(Yte,model.predict(Xte)))

# ------------------------------------------------------
ticker=input("\nEnter a ticker to test (e.g. CEI, TSLA): ").strip().upper()
f=build_features(ticker)
if not f:
    print("❌ Could not compute features for this ticker.")
    raise SystemExit
Xnew=pd.DataFrame([f])
prob=model.predict_proba(Xnew)[0,1]
print(f"\n📊 ====== ML Pump & Dump Risk Report ======")
print(f"Ticker: {ticker}")
print(f"Last Close: ${f['close_last']:.2f}")
print(f"⚠️  ML Risk Score: {prob:.2f}  (1 = likely pump & dump)\n")
print(pd.Series(f))
print("==========================================")
