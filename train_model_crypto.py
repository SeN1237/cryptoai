import pandas as pd
from xgboost import XGBRegressor
from features_prices import build_price_features
from features_news import build_news_features
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

SIMULATION_NUMBER = int(os.environ.get("SIMULATION_NUMBER", 1))

# --- KONFIGURACJA KRYPTOWALUT ---
CRYPTO_TICKERS = [
    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD",
    "DOGE-USD", "DOT-USD", "MATIC-USD", "LTC-USD", "SHIB-USD", "AVAX-USD",
    "UNI-USD", "LINK-USD", "ALGO-USD", "ATOM-USD", "VET-USD", "FTT-USD",
    "FIL-USD", "TRX-USD", "NEAR-USD", "XLM-USD", "HBAR-USD", "ICP-USD",
    "EGLD-USD", "FLOW-USD", "EOS-USD", "AAVE-USD", "MKR-USD", "KSM-USD",
    "SAND-USD", "CHZ-USD", "XTZ-USD", "STX-USD", "CRV-USD", "MANA-USD",
    "GRT-USD", "BAT-USD", "CELO-USD", "1INCH-USD", "ENJ-USD", "ZEC-USD",
    "DASH-USD", "KAVA-USD", "RUNE-USD", "NEO-USD", "QTUM-USD", "ICX-USD",
    "HNT-USD", "ANKR-USD", "XMR-USD", "LRC-USD", "OCEAN-USD"
]

HORIZON = 21
TOP_K = 20
TCOST = 0.0015
INITIAL_CAPITAL = 10000  # $ na start

# --- Budowanie cech ---
print("Budowanie cech cenowych kryptowalut...")
price_feats = build_price_features(CRYPTO_TICKERS)

print("Budowanie cech z newsów...")
news_feats = build_news_features(CRYPTO_TICKERS, days=7)

# Upewniamy się, że obie ramki mają kolumnę 'date'
if 'date' not in price_feats.columns:
    price_feats = price_feats.rename_axis('date').reset_index()
if 'date' not in news_feats.columns:
    news_feats = news_feats.rename_axis('date').reset_index()

# Konwersja dat na datetime
price_feats['date'] = pd.to_datetime(price_feats['date'])
news_feats['date'] = pd.to_datetime(news_feats['date'])

# Łączenie cen i newsów
features = (
    price_feats
    .merge(news_feats, on=['ticker', 'date'], how='left')
    .set_index('date')
    .sort_index()
)

# Tworzymy target
features['target'] = features.groupby('ticker')['ret_21'].shift(-HORIZON)

features['sentiment'] = features['sentiment'].fillna(0)

# --- Trening modelu ---
features = features.dropna(subset=['target'])
Xcols = [c for c in features.columns if c not in ['target','ticker']]
X = features[Xcols]
y = features['target']

model = XGBRegressor(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

print("Trening modelu AI dla kryptowalut...")
model.fit(X, y)

# --- Prognozy ---
preds = pd.Series(model.predict(X), index=X.index)
snapshot = features[['ticker']].copy()
snapshot['pred'] = preds.values
last_snap = snapshot.loc[snapshot.index.max()]
top = last_snap.sort_values('pred', ascending=False).head(TOP_K)

# --- Dodaj prognozę % ---
top['pred_%'] = top['pred']*100

# --- Symulacja equity ---
equity = [INITIAL_CAPITAL]
last_hold = pd.Series(dtype=float)

for date in sorted(features.index.unique()):
    frame = features.loc[date]
    ret_map = frame.set_index('ticker')['ret_1']

    daily_top = frame.copy()
    daily_top['pred'] = model.predict(frame[Xcols])
    daily_top = daily_top.sort_values('pred', ascending=False).head(TOP_K)
    daily_top['weight'] = 1.0 / len(daily_top)
    
    daily_top = daily_top[~daily_top.index.duplicated(keep='first')]
    
    tickers_union = ret_map.index.union(daily_top.index).unique()
    ret_map = ret_map.reindex(tickers_union).fillna(0)
    weights = daily_top['weight'].reindex(tickers_union).fillna(0)
    
    turnover = (last_hold.reindex(tickers_union).fillna(0) - weights).abs().sum()
    r = (ret_map * weights).sum() - TCOST * turnover
    
    equity.append(equity[-1] * (1 + r))
    last_hold = weights.copy()

equity_series = pd.Series(equity[1:], index=sorted(features.index.unique()))

# --- Wyświetlenie wyników ---
print("Top kryptowaluty do kupienia (ostatni dzień):")
print(top[['ticker','pred_%']])

# --- Zapis wyników ---
os.makedirs("top_results_crypto", exist_ok=True)
top_file = f"top_results_crypto/last_top_crypto_{SIMULATION_NUMBER}.csv"
top.to_csv(top_file, index=False)
print(f"Wyniki zapisane do: {top_file}")

total_return = (equity_series[-1]/INITIAL_CAPITAL - 1)*100
print(f"Przykładowy zysk: ${equity_series[-1]:.2f} ({total_return:.2f}%) od {equity_series.index[0].date()} do {equity_series.index[-1].date()}")

# --- Wykres equity ---
equity_series.plot(title="Symulacja portfela kryptowalut")
plt.xlabel("Data")
plt.ylabel("Kapitał ($)")
plt.show()


