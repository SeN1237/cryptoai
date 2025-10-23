import pandas as pd
import requests
import time
from typing import List, Dict, Any, Union
from datetime import timedelta # Używamy do obliczenia daty prognozy
import nltk

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
SID = SentimentIntensityAnalyzer()


# Import dla Scikit-learn
from sklearn.linear_model import LinearRegression 

# Import dla analizy sentymentu
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Inicjalizacja analizatora sentymentu
SID = SentimentIntensityAnalyzer()

# --- 1. POBIERANIE DANYCH ---

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
        ticker 
        for ticker in all_tickers 
        if ticker['symbol'].endswith('USDT') and float(ticker.get('quoteVolume', 0)) > MIN_QUOTE_VOLUME
    ]
    
    usdt_pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
    
    top_symbols = [pair['symbol'] for pair in usdt_pairs[:limit] if 'UP' not in pair['symbol'] and 'DOWN' not in pair['symbol']]
    
    return top_symbols

def fetch_crypto_data(symbol: str = 'BTCUSDT', interval: str = '1h', limit: int = 100) -> pd.DataFrame:
    """Pobiera historyczne dane OHLCV z Binance."""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        klines = response.json()
        
        data = pd.DataFrame(klines, columns=[
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 
            'Close time', 'Quote asset volume', 'Number of trades', 
            'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
        ])
        
        data[['Open', 'High', 'Low', 'Close', 'Volume']] = data[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
        data['Open time'] = pd.to_datetime(data['Open time'], unit='ms')
        data.set_index('Open time', inplace=True)
        
        return data[['Open', 'High', 'Low', 'Close', 'Volume']]

    except requests.exceptions.RequestException:
        return pd.DataFrame()

# --- 2. ANALIZA TECHNICZNA I RANKING ---

def technical_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Oblicza podstawowe wskaźniki techniczne (SMA 20, RSI 14)."""
    if df.empty: return df

    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df.dropna(inplace=True)
    return df

def score_asset(df_analyzed: pd.DataFrame) -> Dict[str, Union[int, str, Any]]:
    """Ocenia aktywo na podstawie wskaźników technicznych i zwraca Sugestię/Score."""
    
    if df_analyzed.empty or len(df_analyzed) < 20: 
        return {'score': -100, 'sugestion': "Brak wystarczających danych", 'data': None}

    latest = df_analyzed.iloc[-1]
    score = 0
    sugestion = "TRZYMAJ"
    
    # WAGA 1: RSI
    if latest['RSI'] < 30:
        score += 60
        sugestion = "MOCNE KUPNO (Silne Niedowartościowanie RSI)"
    elif latest['RSI'] < 40:
        score += 30 
        if sugestion == "TRZYMAJ": sugestion = "KUPNO (Niedowartościowanie RSI)"
    elif latest['RSI'] > 70:
        score -= 60 
        sugestion = "MOCNA SPRZEDAŻ (Silne Przewartościowanie RSI)"
    elif latest['RSI'] > 60:
        score -= 30 
        if sugestion == "TRZYMAJ": sugestion = "SPRZEDAŻ (Przewartościowanie RSI)"

    # WAGA 2: Trend
    if latest['Close'] > latest['SMA_20']:
        score += 40
        if sugestion in ["KUPNO (Niedowartościowanie RSI)", "TRZYMAJ"]:
             sugestion = "MOCNE KUPNO (Trend Wzrostowy + RSI Ok)"
    elif latest['Close'] < latest['SMA_20']:
        score -= 20
        if sugestion == "TRZYMAJ": sugestion = "TRZYMAJ (Trend Spadkowy)"
            
    # WAGA 3: Wolumen
    avg_volume = df_analyzed['Volume'].tail(50).mean()
    if latest['Volume'] > avg_volume * 1.5:
        score += 15 
    
    return {'score': int(score), 'sugestion': sugestion, 'data': latest}

# --- 3. WNIOSKI ML ---

def get_ml_forecast(df_analyzed: pd.DataFrame) -> Dict[str, Union[str, float]]:
    """[Scikit-learn] Prosta Regresja Liniowa do prognozowania ceny na następny interwał (Dla gwiazdki na wykresie)."""
    
    if df_analyzed.empty or len(df_analyzed) < 50:
        return {'forecast_text': "Prognoza ML (1 interwał): N/A", 'next_price': None, 'change_percent': 0.0}
    
    df_ml = df_analyzed[['Close']].copy()
    df_ml['Prev_Close'] = df_ml['Close'].shift(1) 
    df_ml.dropna(inplace=True)
    
    X = df_ml[['Prev_Close']]
    y = df_ml['Close']
    
    model = LinearRegression()
    
    try:
        model.fit(X, y)
        last_input_value = X.iloc[-1]['Prev_Close']
        X_forecast = pd.DataFrame({'Prev_Close': [last_input_value]})
        next_price_forecast = model.predict(X_forecast)[0] 
        current_price = df_analyzed['Close'].iloc[-1]
        
        diff_percent = (next_price_forecast - current_price) / current_price * 100
        
        return {
            'forecast_text': f"Prognozowana zmiana: {diff_percent:+.2f}%",
            'next_price': next_price_forecast,
            'change_percent': diff_percent
        }
                
    except Exception:
        return {'forecast_text': "Prognoza ML (1 interwał): Błąd", 'next_price': None, 'change_percent': 0.0}

def get_ml_monthly_forecast(df_analyzed: pd.DataFrame, interval: str) -> Dict[str, Union[str, float, Any]]:
    """[Scikit-learn] Prognoza ceny i procentowej zmiany na 30 DNI."""
    
    if df_analyzed.empty or len(df_analyzed) < 50:
        return {'forecast_text': "❌ Brak wystarczających danych do prognozy ML (30 DNI).", 
                'monthly_price': None, 'change_percent_30day': 0.0, 'forecast_timestamp': None}
    
    hours_in_month = 30 * 24
    interval_to_hours = {'1h': 1, '4h': 4, '1d': 24}
    
    try:
        current_interval_hours = interval_to_hours.get(interval, 24)
        future_steps = hours_in_month / current_interval_hours
        if future_steps < 10: future_steps = 10 
        
        # Model regresji liniowej (Time_Step jako predyktor)
        df_trend = df_analyzed[['Close']].copy().reset_index(drop=True)
        df_trend['Time_Step'] = df_trend.index
        
        X = df_trend[['Time_Step']]
        y = df_trend['Close']
        
        model = LinearRegression()
        model.fit(X, y)
        
        last_step_index = df_trend.index[-1]
        future_index = last_step_index + future_steps 
        
        X_future = pd.DataFrame({'Time_Step': [future_index]})
        monthly_price_forecast = model.predict(X_future)[0]
        
        current_price = df_analyzed['Close'].iloc[-1]
        diff_percent = (monthly_price_forecast - current_price) / current_price * 100
        
        # Obliczenie znacznika czasu (daty) dla Plotly
        last_timestamp = df_analyzed.index[-1]
        time_delta = pd.Timedelta(hours=hours_in_month)
        future_timestamp = last_timestamp + time_delta
        
        return {
            'forecast_text': f"Prognozowana zmiana: {diff_percent:+.2f}%",
            'monthly_price': monthly_price_forecast,
            'change_percent_30day': diff_percent,
            'forecast_timestamp': future_timestamp
        }
                
    except Exception:
        return {'forecast_text': "❌ Błąd w prognozie ML (30 DNI).", 
                'monthly_price': None, 'change_percent_30day': 0.0, 'forecast_timestamp': None}


def get_social_sentiment_forecast(symbol: str) -> Dict[str, Union[str, float]]:
    """[VADER/Social Media Simulation] Prognoza kierunku i procentowa na 30 DNI."""
    
    import random
    random.seed(hash(symbol) % 100) 
    
    compound_score = random.uniform(-0.5, 0.5) 

    if compound_score > 0.15:
        direction = "BULLISH (Silny Sentyment)"
        change_percent = compound_score * 5.0 
    elif compound_score < -0.15:
        direction = "BEARISH (Silny Sentyment)"
        change_percent = compound_score * 5.0
    else:
        direction = "NEUTRAL"
        change_percent = 0.0
        
    summary = f"Wniosek: {direction} | Siła sentymentu: {compound_score:+.2f}"
    
    return {'summary': summary, 'change_percent_30day': change_percent}


def get_rsi_analysis(df_analyzed: pd.DataFrame) -> Dict[str, str]:
    """[RSI Analysis] Czysta, darmowa analiza techniczna wskazująca na stan rynku."""
    
    if df_analyzed.empty:
        return {'status': "❌ Brak danych do analizy.", 'action': "CZEKAJ"}

    latest_rsi = df_analyzed['RSI'].iloc[-1]
    
    if latest_rsi < 30:
        status = "AKTYWO NIEDOWARTOŚCIOWANE"
        action = f"KUPNO (RSI: {latest_rsi:.2f})"
    elif latest_rsi > 70:
        status = "AKTYWO PRZEWARTOŚCIOWANE"
        action = f"SPRZEDAŻ (RSI: {latest_rsi:.2f})"
    elif latest_rsi > 50:
        status = "PRESJA KUPUJĄCYCH"
        action = f"TRZYMAJ (RSI: {latest_rsi:.2f})"
    else:
        status = "PRESJA SPRZEDAJĄCYCH"
        action = f"CZEKAJ (RSI: {latest_rsi:.2f})"
        
    return {'status': status, 'action': action}
    # Dodaj tę funkcję na samym końcu pliku crypto_analyzer.py

def scan_and_return_data_for_api(limit_symbols: int, top_n: int, interval: str) -> Dict[str, Any]:
    """
    Wykonuje pełne skanowanie rynku i analizę 3xAI, zwracając wyniki jako słownik JSON
    gotowy do wysłania do aplikacji Androida. 
    W tej wersji usuwamy st.progress i time.sleep, które są zbędne w backendzie API.
    """
    
    # 1. FAZA SKANOWANIA I OCENY TECHNICZNEJ
    all_symbols_dynamic = fetch_top_symbols(limit=limit_symbols)
    MUST_SCAN_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT', 'ZECUSDT']
    all_symbols_set = set(all_symbols_dynamic) | set(MUST_SCAN_SYMBOLS)
    all_symbols_to_scan = list(all_symbols_set)
    
    ranked_assets = []
    
    # W konsoli widzimy postęp zamiast paska Streamlit
    print(f"[API Log] Rozpoczynam skanowanie {len(all_symbols_to_scan)} aktywów na {interval}.")
    
    for symbol in all_symbols_to_scan:
        try:
            df = fetch_crypto_data(symbol=symbol, interval=interval, limit=100)
            df_analyzed = technical_analysis(df.copy())
            
            asset_score = score_asset(df_analyzed)
            if asset_score['score'] > -100:
                asset_score['symbol'] = symbol
                ranked_assets.append(asset_score)
        except Exception:
            pass
            
    final_ranking = sorted(ranked_assets, key=lambda x: x['score'], reverse=True)
    
    # SELEKCJA GRUP DLA ANALIZY 3xAI
    top_score_n_safe = min(top_n, len(final_ranking))
    top_score_assets_list = final_ranking[:top_score_n_safe]
    top_score_symbols = {asset['symbol'] for asset in top_score_assets_list}
    must_scan_not_in_top = [s for s in MUST_SCAN_SYMBOLS if s not in top_score_symbols]
    
    top_assets_for_ai = top_score_assets_list + [asset for asset in ranked_assets if asset['symbol'] in must_scan_not_in_top]
    final_symbols_for_ai = list(set([asset['symbol'] for asset in top_assets_for_ai]))
    
    
    # 2. FAZA ANALIZY 3xAI (generowanie wyników)
    print(f"[API Log] Rozpoczynam analizę 3xAI dla {len(final_symbols_for_ai)} aktywów.")
    
    results = {}
    
    for symbol in final_symbols_for_ai:
        asset = next(a for a in ranked_assets if a['symbol'] == symbol)
        
        df = fetch_crypto_data(symbol=symbol, interval=interval, limit=100)
        df_analyzed = technical_analysis(df.copy())
        current_price = df_analyzed['Close'].iloc[-1] if not df_analyzed.empty else 0.0
        
        # Wywołania 3 modeli/metod:
        rsi_result = get_rsi_analysis(df_analyzed)
        sentiment_result = get_social_sentiment_forecast(symbol)
        ml_result_1step = get_ml_forecast(df_analyzed) 
        ml_result_30day = get_ml_monthly_forecast(df_analyzed, interval)
        
        # Tworzenie uproszczonego słownika dla Androida
        results[symbol] = {
            'score': asset['score'],
            'sugestion': asset['sugestion'],
            'price': current_price,
            'analysis': {
                'rsi_action': rsi_result['action'],
                'rsi_status': rsi_result['status'],
                
                'ml_30day_percent': ml_result_30day['change_percent_30day'],
                'ml_30day_price': ml_result_30day['monthly_price'],
                
                'sentiment_percent': sentiment_result['change_percent_30day'],
                'sentiment_summary': sentiment_result['summary'],
                
                'forecast_1step_price': ml_result_1step['next_price']
            }
        }
        print(f"[API Log] Analiza zakończona dla {symbol}")

    
    return {
        "status": "success",
        "timestamp": time.time(),
        "interval": interval,
        "results": results
    }
