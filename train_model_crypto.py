import pandas as pd
from lightgbm import LGBMRegressor # ⬅️ ZMIANA: Używamy LightGBM zamiast XGBoost
from features_prices import build_price_features # Zakładamy, że ta funkcja działa
from features_news import build_news_features # Zakładamy, że ta funkcja działa
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

SIMULATION_NUMBER = int(os.environ.get("SIMULATION_NUMBER", 1))

# --- KONFIGURACJA KRYPTOWALUT (OPTYMALIZACJA STABILNOŚCI) ---
# Ograniczona lista, aby zmniejszyć ryzyko błędu pamięci (512 MB RAM)
CRYPTO_TICKERS = [
    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD",
    "DOGE-USD", "DOT-USD", "MATIC-USD", "LTC-USD", "SHIB-USD", "AVAX-USD",
    "UNI-USD", "LINK-USD", "ALGO-USD"
]

HORIZON = 21 # Horyzont prognozy (np. 21 dni handlowych)
TOP_K = 5 # Wybieramy Top 5 z prognozą
TCOST = 0.0015
INITIAL_CAPITAL = 10000  # $ na start

# --- Budowanie cech ---
print("Budowanie cech cenowych kryptowalut...")
# UWAGA: Te funkcje muszą być stabilne i działać w środowisku GitHub Actions
price_feats = build_price_features(CRYPTO_TICKERS) 

print("Budowanie cech z newsów...")
news_feats = build_news_features(CRYPTO_TICKERS, days=7)

# Upewniamy się, że obie ramki mają kolumnę 'date'
if 'date' not in price_feats.columns:
    price_feats = price_feats.rename_axis('date').reset_index()
if 'date' not in news_feats.columns:
    news_feats = news_feats.rename_axis('date').reset_index()

# Konwersja dat na datetime
price_feats['date'] = pd.to_datetime(price_feats['date']).dt.normalize()
news_feats['date'] = pd.to_datetime(news_feats['date']).dt.normalize()

# Łączenie i przygotowanie danych
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

# ⬅️ ZMIANA MODELU NA LIGHTGBM
model = LGBMRegressor(
    n_estimators=200, # Zmniejszona liczba dla szybszego treningu
    max_depth=4, 
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1 # Używa wszystkich dostępnych rdzeni
)

print(f"Trening modelu AI (LGBM) dla symulacji {SIMULATION_NUMBER}...")
model.fit(X, y)

# --- Prognozy i Zapis ---
preds = pd.Series(model.predict(X), index=X.index)
snapshot = features[['ticker']].copy()
snapshot['pred'] = preds.values
last_snap = snapshot.loc[snapshot.index.max()]
top = last_snap.sort_values('pred', ascending=False).head(TOP_K)

# Prognoza w %
top['pred_%'] = top['pred']*100

# Tworzymy folder, jeśli nie istnieje
os.makedirs("top_results_crypto", exist_ok=True)

# Zapis do pliku
top_file = f"top_results_crypto/last_top_crypto_{SIMULATION_NUMBER}.csv"
top.to_csv(top_file, index=False)
print(f"Wyniki zapisane do: {top_file}")

# [Usunięto kod Symulacji Equity i Wykresu, ponieważ jest niepotrzebny w Cron Job]
