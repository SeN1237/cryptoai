import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def build_price_features(tickers, start=None, end=None):
    from datetime import datetime, timedelta
    if start is None:
        start = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")  # do dzisiaj
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    close = data['Close']
    volume = data['Volume']
    feats = {}
    for t in close.columns:
        px = close[t].dropna()
        vol = volume[t].reindex(px.index).ffill()
        df = pd.DataFrame(index=px.index)
        df['ret_1'] = px.pct_change(1)
        df['ret_5'] = px.pct_change(5)
        df['ret_21'] = px.pct_change(21)
        df['vol_21'] = df['ret_1'].rolling(21).std()
        df['vol_63'] = df['ret_1'].rolling(63).std()
        df['ma_10'] = px.rolling(10).mean()/px - 1
        df['ma_50'] = px.rolling(50).mean()/px - 1
        delta = px.diff()
        up = delta.clip(lower=0).rolling(14).mean()
        down = (-delta.clip(upper=0)).rolling(14).mean()
        rs = up / down.replace(0,np.nan)
        df['rsi_14'] = 100 - (100/(1+rs))
        df['vol_z'] = (vol.pct_change(1)).rolling(21).mean()
        df['ticker'] = t
        feats[t] = df
    big = pd.concat(feats.values(), axis=0).dropna()
    big['ticker'] = big['ticker'].astype('category')
    return big

