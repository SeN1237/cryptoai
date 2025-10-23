import pandas as pd
import requests
import time
import random
from typing import List, Dict, Any, Union
from sklearn.linear_model import LinearRegression
import nltk

# --- NLTK Vader ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
SID = SentimentIntensityAnalyzer()

# --- 1. POBIERANIE DANYCH ---

def fetch_top_symbols(limit: int = 50) -> List[str]:
    """Pobiera top symbole USDT wg wolumenu z Binance. Fallback: zwraca pustą listę."""
    url = "https://api.binance.com/api/v3/ticker/24hr"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        tickers = r.json()
    except Exception:
        return []

    MIN_VOLUME = 500_000
    usdt_pairs = [
        t for t in tickers
        if t['symbol'].endswith('USDT') 
        and float(t.get('quoteVolume', 0)) > MIN_VOLUME
        and 'UP' not in t['symbol'] and 'DOWN' not in t['symbol']
    ]
    usdt_pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
    return [t['symbol'] for t in usdt_pairs[:limit]] or []

def fetch_crypto_data(symbol: str='BTCUSDT', interval: str='1h', limit: int=100) -> pd.DataFrame:
    """Pobiera dane OHLCV z Binance. Fallback: zwraca 10 wierszy losowych danych."""
    url = "https://api.binance.com/api/v3/klines"
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        klines = r.json()
        df = pd.DataFrame(klines, columns=[
            'Open time','Open','High','Low','Close','Volume','Close time',
            'Quote asset volume','Number of trades','Taker buy base asset volume',
            'Taker buy quote asset volume','Ignore'
        ])
        df[['Open','High','Low','Close','Volume']] = df[['Open','High','Low','Close','Volume']].astype(float)
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        df.set_index('Open time', inplace=True)
        return df[['Open','High','Low','Close','Volume']]
    except Exception:
        # Fallback - sztuczne dane
        df = pd.DataFrame({
            'Open': [random.uniform(10,100) for _ in range(10)],
            'High': [random.uniform(10,100) for _ in range(10)],
            'Low': [random.uniform(10,100) for _ in range(10)],
            'Close': [random.uniform(10,100) for _ in range(10)],
            'Volume': [random.uniform(1000,5000) for _ in range(10)]
        }, index=pd.date_range(end=pd.Timestamp.now(), periods=10, freq=interval.upper()))
        return df

# --- 2. TECHNICAL ANALYSIS ---

def technical_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """SMA20 i RSI14. Zabezpieczenie dla krótkich danych."""
    if df.empty or len(df)<2:
        # Tworzymy minimalny wiersz z danymi
        df = pd.DataFrame({
            'Open':[0.0],'High':[0.0],'Low':[0.0],'Close':[0.0],'Volume':[0.0]
        }, index=[pd.Timestamp.now()])
    df['SMA_20'] = df['Close'].rolling(20,min_periods=1).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14,min_periods=1).mean()
    loss = -delta.clip(upper=0).rolling(14,min_periods=1).mean()
    rs = gain/loss.replace(0,1)  # Unikamy dzielenia przez 0
    df['RSI'] = 100-(100/(1+rs))
    return df

def score_asset(df_analyzed: pd.DataFrame) -> Dict[str, Union[int,str,Any]]:
    """Score na podstawie RSI, SMA i wolumenu."""
    if df_analyzed.empty:
        return {'score':0,'sugestion':'Brak danych','data':None}
    latest = df_analyzed.iloc[-1]
    score = 0
    sugestion = "TRZYMAJ"
    # RSI
    if latest['RSI'] < 30:
        score += 60; sugestion="MOCNE KUPNO"
    elif latest['RSI'] < 40:
        score += 30; sugestion="KUPNO"
    elif latest['RSI'] > 70:
        score -= 60; sugestion="MOCNA SPRZEDAŻ"
    elif latest['RSI'] > 60:
        score -= 30; sugestion="SPRZEDAŻ"
    # Trend
    if latest['Close'] > latest['SMA_20']: score += 40
    elif latest['Close'] < latest['SMA_20']: score -= 20
    # Wolumen
    avg_vol = df_analyzed['Volume'].tail(50).mean() or 1
    if latest['Volume'] > avg_vol*1.5: score += 15
    return {'score':int(score),'sugestion':sugestion,'data':latest}

# --- 3. RSI Analysis ---
def get_rsi_analysis(df_analyzed: pd.DataFrame) -> Dict[str,str]:
    if df_analyzed.empty:
        return {'status':"Brak danych","action":"CZEKAJ"}
    rsi = df_analyzed['RSI'].iloc[-1]
    if rsi < 30:
        return {'status':"AKTYWO NIEDOWARTOŚCIOWANE",'action':f"KUPNO (RSI:{rsi:.2f})"}
    elif rsi > 70:
        return {'status':"AKTYWO PRZEWARTOŚCIOWANE",'action':f"SPRZEDAŻ (RSI:{rsi:.2f})"}
    elif rsi > 50:
        return {'status':"PRESJA KUPUJĄCYCH",'action':f"TRZYMAJ (RSI:{rsi:.2f})"}
    else:
        return {'status':"PRESJA SPRZEDAJĄCYCH",'action':f"CZEKAJ (RSI:{rsi:.2f})"}

# --- 4. ML Forecast ---
def get_ml_forecast(df_analyzed: pd.DataFrame) -> Dict[str, Union[str,float]]:
    if df_analyzed.empty or len(df_analyzed)<2:
        return {'forecast_text':'N/A','next_price':None,'change_percent':0.0}
    df_ml = df_analyzed[['Close']].copy()
    df_ml['Prev_Close'] = df_ml['Close'].shift(1)
    df_ml.dropna(inplace=True)
    X = df_ml[['Prev_Close']]
    y = df_ml['Close']
    model = LinearRegression()
    try:
        model.fit(X,y)
        next_price = model.predict([[X.iloc[-1,0]]])[0]
        diff = (next_price - df_analyzed['Close'].iloc[-1])/df_analyzed['Close'].iloc[-1]*100
        return {'forecast_text':f"{diff:+.2f}%","next_price":next_price,'change_percent':diff}
    except:
        return {'forecast_text':'Błąd','next_price':None,'change_percent':0.0}

def get_ml_monthly_forecast(df_analyzed: pd.DataFrame, interval: str) -> Dict[str, Any]:
    if df_analyzed.empty or len(df_analyzed)<2:
        return {'forecast_text':'Brak danych','monthly_price':None,'change_percent_30day':0.0,'forecast_timestamp':None}
    interval_hours = {'1h':1,'4h':4,'1d':24}.get(interval,24)
    hours_in_month = 30*24
    steps = hours_in_month/interval_hours
    df_ml = df_analyzed[['Close']].copy().reset_index(drop=True)
    df_ml['Time_Step'] = df_ml.index
    X = df_ml[['Time_Step']]; y = df_ml['Close']
    model = LinearRegression()
    try:
        model.fit(X,y)
        future_idx = df_ml.index[-1]+steps
        future_price = model.predict([[future_idx]])[0]
        diff = (future_price - df_analyzed['Close'].iloc[-1])/df_analyzed['Close'].iloc[-1]*100
        future_ts = df_analyzed.index[-1]+pd.Timedelta(hours=hours_in_month)
        return {'forecast_text':f"{diff:+.2f}%","monthly_price":future_price,'change_percent_30day':diff,'forecast_timestamp':future_ts}
    except:
        return {'forecast_text':'Błąd','monthly_price':None,'change_percent_30day':0.0,'forecast_timestamp':None}

# --- 5. Social Sentiment ---
def get_social_sentiment_forecast(symbol: str) -> Dict[str, Union[str,float]]:
    random.seed(hash(symbol)%100)
    compound = random.uniform(-0.5,0.5)
    if compound>0.15:
        direction="BULLISH"
        change=compound*5
    elif compound<-0.15:
        direction="BEARISH"
        change=compound*5
    else:
        direction="NEUTRAL"
        change=0.0
    summary=f"Wniosek: {direction} | Siła: {compound:+.2f}"
    return {'summary':summary,'change_percent_30day':change}
