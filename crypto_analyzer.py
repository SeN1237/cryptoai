import pandas as pd
import requests
import time
import numpy as np
from typing import List, Dict, Any, Union
from datetime import timedelta
import nltk

# Inicjalizacja VADER do analizy sentymentu
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
SID = SentimentIntensityAnalyzer()

# ML
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# ------------------------
# 1. POBIERANIE DANYCH
# ------------------------

def fetch_top_symbols(limit: int = 50) -> List[str]:
    """Pobiera listę symboli krypto z dużym wolumenem z Binance."""
    url = "https://api.binance.com/api/v3/ticker/24hr"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        all_tickers = response.json()
    except requests.exceptions.RequestException:
        return []

    MIN_QUOTE_VOLUME = 500000
    usdt_pairs = [
        ticker for ticker in all_tickers
        if ticker['symbol'].endswith('USDT') and float(ticker.get('quoteVolume', 0)) > MIN_QUOTE_VOLUME
    ]
    usdt_pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
    top_symbols = [pair['symbol'] for pair in usdt_pairs[:limit] if 'UP' not in pair['symbol'] and 'DOWN' not in pair['symbol']]
    return top_symbols

def fetch_crypto_data(symbol: str = 'BTCUSDT', interval: str = '1h', limit: int = 100) -> pd.DataFrame:
    """Pobiera dane OHLCV z Binance."""
    url = "https://api.binance.com/api/v3/klines"
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        klines = response.json()
        df = pd.DataFrame(klines, columns=[
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close time', 'Quote asset volume', 'Number of trades',
            'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
        ])
        df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        df.set_index('Open time', inplace=True)
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except requests.exceptions.RequestException:
        return pd.DataFrame()

# ------------------------
# 2. ANALIZA TECHNICZNA I SCORING
# ------------------------

def technical_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Oblicza SMA20 i RSI14."""
    if df.empty:
        return df
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df.dropna(inplace=True)
    return df

def score_asset(df_analyzed: pd.DataFrame, sentiment_change: float = 0.0) -> Dict[str, Any]:
    """Score na podstawie RSI, SMA20, wolumenu i sentymentu."""
    if df_analyzed.empty or len(df_analyzed) < 20:
        return {'score': -100, 'sugestion': "Brak danych", 'data': None}
    
    latest = df_analyzed.iloc[-1]
    score = 0
    sugestion = "TRZYMAJ"
    
    # RSI
    if latest['RSI'] < 30:
        score += 60
        sugestion = "MOCNE KUPNO"
    elif latest['RSI'] < 40:
        score += 30
        sugestion = "KUPNO"
    elif latest['RSI'] > 70:
        score -= 60
        sugestion = "MOCNA SPRZEDAŻ"
    elif latest['RSI'] > 60:
        score -= 30
        sugestion = "SPRZEDAŻ"
    
    # Trend SMA20
    if latest['Close'] > latest['SMA_20']:
        score += 40
        if sugestion in ["TRZYMAJ", "KUPNO"]: sugestion = "MOCNE KUPNO"
    else:
        score -= 20
    
    # Wolumen
    avg_vol = df_analyzed['Volume'].tail(50).mean()
    if latest['Volume'] > avg_vol * 1.5:
        score += 15
    
    # Sentyment
    score += int(sentiment_change)  # dodaje wpływ sentymentu
    
    return {'score': int(score), 'sugestion': sugestion, 'data': latest}

# ------------------------
# 3. ANALIZA RSI
# ------------------------

def get_rsi_analysis(df_analyzed: pd.DataFrame) -> Dict[str, str]:
    if df_analyzed.empty:
        return {'status': "Brak danych", 'action': "CZEKAJ"}
    rsi = df_analyzed['RSI'].iloc[-1]
    if rsi < 30:
        return {'status': "AKTYWO NIEDOWARTOŚCIOWANE", 'action': f"KUPNO (RSI: {rsi:.2f})"}
    elif rsi > 70:
        return {'status': "AKTYWO PRZEWARTOŚCIOWANE", 'action': f"SPRZEDAŻ (RSI: {rsi:.2f})"}
    elif rsi > 50:
        return {'status': "PRESJA KUPUJĄCYCH", 'action': f"TRZYMAJ (RSI: {rsi:.2f})"}
    else:
        return {'status': "PRESJA SPRZEDAJĄCYCH", 'action': f"CZEKAJ (RSI: {rsi:.2f})"}

# ------------------------
# 4. PROGNOZY ML
# ------------------------

def get_ml_forecast(df_analyzed: pd.DataFrame) -> Dict[str, Any]:
    """Prognoza 1 interwał (gwiazdka)."""
    if df_analyzed.empty or len(df_analyzed) < 20:
        return {'forecast_text': "N/A", 'next_price': None, 'change_percent': 0.0}
    
    df_ml = df_analyzed[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df_ml['Prev_Close'] = df_ml['Close'].shift(1)
    df_ml.dropna(inplace=True)
    
    X = df_ml[['Open','High','Low','Close','Volume','Prev_Close']]
    y = df_ml['Close']
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    last_row = X.iloc[-1:]
    next_price = model.predict(last_row)[0]
    current_price = df_analyzed['Close'].iloc[-1]
    diff_percent = (next_price - current_price) / current_price * 100
    
    return {'forecast_text': f"{diff_percent:+.2f}%", 'next_price': next_price, 'change_percent': diff_percent}

def get_ml_monthly_forecast(df_analyzed: pd.DataFrame, interval: str) -> Dict[str, Any]:
    """Prognoza 30 dniowa (duża kropka)."""
    if df_analyzed.empty or len(df_analyzed) < 20:
        return {'forecast_text': "Brak danych", 'monthly_price': None, 'change_percent_30day': 0.0, 'forecast_timestamp': None}
    
    interval_to_hours = {'1h':1, '4h':4, '1d':24}
    hours_in_month = 30*24
    steps = int(hours_in_month / interval_to_hours.get(interval,24))
    
    df_ml = df_analyzed[['Close']].reset_index()
    df_ml['TimeStep'] = np.arange(len(df_ml))
    
    X = df_ml[['TimeStep']]
    y = df_ml['Close']
    
    model = LinearRegression()
    model.fit(X, y)
    
    future_step = X['TimeStep'].iloc[-1] + steps
    future_price = model.predict([[future_step]])[0]
    current_price = df_analyzed['Close'].iloc[-1]
    diff_percent = (future_price - current_price) / current_price * 100
    
    future_timestamp = df_analyzed.index[-1] + pd.Timedelta(hours=hours_in_month)
    
    return {'forecast_text': f"{diff_percent:+.2f}%", 'monthly_price': future_price, 'change_percent_30day': diff_percent, 'forecast_timestamp': future_timestamp}

# ------------------------
# 5. Sentyment społecznościowy (symulowany)
# ------------------------

def get_social_sentiment_forecast(symbol: str) -> Dict[str, Any]:
    """Symulacja sentymentu społecznościowego na podstawie symbolu."""
    import random
    random.seed(hash(symbol)%100)
    compound = random.uniform(-0.3,0.3)
    
    if compound > 0.15:
        direction = "BULLISH"
    elif compound < -0.15:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"
    
    change_percent = compound*5
    summary = f"Wniosek: {direction} | Siła: {compound:+.2f}"
    
    return {'summary': summary, 'change_percent_30day': change_percent}

# ------------------------
# 6. FUNKCJA DO API / BACKENDU
# ------------------------

def scan_and_return_data_for_api(limit_symbols:int, top_n:int, interval:str) -> Dict[str, Any]:
    """Pełna analiza rynku w trybie backendowym."""
    
    all_symbols_dynamic = fetch_top_symbols(limit_symbols)
    MUST_SCAN_SYMBOLS = ['BTCUSDT','ETHUSDT','BNBUSDT','SOLUSDT','XRPUSDT','ADAUSDT','DOGEUSDT','AVAXUSDT','DOTUSDT','LINKUSDT','ZECUSDT']
    
    all_symbols_set = set(all_symbols_dynamic)|set(MUST_SCAN_SYMBOLS)
    all_symbols_to_scan = list(all_symbols_set)
    
    ranked_assets = []
    print(f"[API] Rozpoczynam skanowanie {len(all_symbols_to_scan)} symboli...")
    
    for s in all_symbols_to_scan:
        try:
            df = fetch_crypto_data(symbol=s, interval=interval, limit=100)
            sentiment = get_social_sentiment_forecast(s)
            scored = score_asset(technical_analysis(df.copy()), sentiment['change_percent_30day'])
            scored['symbol'] = s
            if scored['score'] > -100:
                ranked_assets.append(scored)
        except Exception:
            pass
    
    ranked_assets.sort(key=lambda x:x['score'], reverse=True)
    top_assets = ranked_assets[:min(top_n, len(ranked_assets))]
    
    results = {}
    for asset in top_assets:
        s = asset['symbol']
        df = fetch_crypto_data(symbol=s, interval=interval, limit=100)
        df_an = technical_analysis(df.copy())
        rsi_res = get_rsi_analysis(df_an)
        ml1 = get_ml_forecast(df_an)
        ml30 = get_ml_monthly_forecast(df_an, interval)
        sentiment = get_social_sentiment_forecast(s)
        price = df['Close'].iloc[-1] if not df.empty else 0.0
        
        results[s] = {
            'score': asset['score'],
            'sugestion': asset['sugestion'],
            'price': price,
            'analysis': {
                'rsi_action': rsi_res['action'],
                'rsi_status': rsi_res['status'],
                'ml_30day_percent': ml30['change_percent_30day'],
                'ml_30day_price': ml30['monthly_price'],
                'sentiment_percent': sentiment['change_percent_30day'],
                'sentiment_summary': sentiment['summary'],
                'forecast_1step_price': ml1['next_price']
            }
        }
    
    return {'status':'success','timestamp':time.time(),'interval':interval,'results':results}
