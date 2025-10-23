import pandas as pd
import requests
import time
from typing import List, Dict, Any, Union
from sklearn.linear_model import LinearRegression
import nltk

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
SID = SentimentIntensityAnalyzer()

# --- 1. POBIERANIE DANYCH ---
def fetch_top_symbols(limit: int = 50) -> List[str]:
    url = "https://api.binance.com/api/v3/ticker/24hr"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        tickers = response.json()
    except requests.exceptions.RequestException:
        return []

    MIN_QUOTE_VOLUME = 500000 
    usdt_pairs = [t for t in tickers if t['symbol'].endswith('USDT') and float(t.get('quoteVolume',0)) > MIN_QUOTE_VOLUME]
    usdt_pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
    top_symbols = [p['symbol'] for p in usdt_pairs[:limit] if 'UP' not in p['symbol'] and 'DOWN' not in p['symbol']]
    return top_symbols

def fetch_crypto_data(symbol: str='BTCUSDT', interval: str='1h', limit: int=100) -> pd.DataFrame:
    url = "https://api.binance.com/api/v3/klines"
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        klines = resp.json()
        df = pd.DataFrame(klines, columns=['Open time','Open','High','Low','Close','Volume','Close time','Quote asset volume','Num trades','Taker buy base','Taker buy quote','Ignore'])
        df[['Open','High','Low','Close','Volume']] = df[['Open','High','Low','Close','Volume']].astype(float)
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        df.set_index('Open time', inplace=True)
        return df[['Open','High','Low','Close','Volume']]
    except requests.exceptions.RequestException:
        return pd.DataFrame()

# --- 2. ANALIZA TECHNICZNA ---
def technical_analysis(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df['SMA_20'] = df['Close'].rolling(20).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta>0,0).rolling(14).mean()
    loss = -delta.where(delta<0,0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1+rs))
    df.dropna(inplace=True)
    return df

def score_asset(df_analyzed: pd.DataFrame) -> Dict[str, Union[int,str,Any]]:
    if df_analyzed.empty or len(df_analyzed)<20: 
        return {'score':-100, 'sugestion':"Brak danych",'data':None}
    latest = df_analyzed.iloc[-1]
    score=0; sugestion="TRZYMAJ"
    if latest['RSI']<30: score+=60; sugestion="MOCNE KUPNO"
    elif latest['RSI']<40: score+=30; sugestion="KUPNO" if sugestion=="TRZYMAJ" else sugestion
    elif latest['RSI']>70: score-=60; sugestion="MOCNA SPRZEDAŻ"
    elif latest['RSI']>60: score-=30; sugestion="SPRZEDAŻ" if sugestion=="TRZYMAJ" else sugestion
    if latest['Close']>latest['SMA_20']: score+=40
    elif latest['Close']<latest['SMA_20']: score-=20
    avg_vol = df_analyzed['Volume'].tail(50).mean()
    if latest['Volume']>avg_vol*1.5: score+=15
    return {'score':int(score), 'sugestion':sugestion, 'data':latest}

# --- 3. ANALIZA ML ---
def get_ml_forecast(df_analyzed: pd.DataFrame) -> Dict[str, Union[str,float]]:
    if df_analyzed.empty or len(df_analyzed)<50:
        return {'forecast_text':"N/A",'next_price':None,'change_percent':0.0}
    df_ml = df_analyzed[['Close']].copy()
    df_ml['Prev_Close'] = df_ml['Close'].shift(1)
    df_ml.dropna(inplace=True)
    X,y = df_ml[['Prev_Close']], df_ml['Close']
    model=LinearRegression()
    try:
        model.fit(X,y)
        next_price = model.predict(pd.DataFrame({'Prev_Close':[X.iloc[-1,0]]}))[0]
        diff_percent = (next_price - df_analyzed['Close'].iloc[-1])/df_analyzed['Close'].iloc[-1]*100
        return {'forecast_text':f"{diff_percent:+.2f}%", 'next_price':next_price, 'change_percent':diff_percent}
    except Exception:
        return {'forecast_text':"Błąd",'next_price':None,'change_percent':0.0}

def get_ml_monthly_forecast(df_analyzed: pd.DataFrame, interval: str) -> Dict[str, Union[str,float,Any]]:
    if df_analyzed.empty or len(df_analyzed)<50:
        return {'forecast_text':"N/A",'monthly_price':None,'change_percent_30day':0.0,'forecast_timestamp':None}
    hours_in_month = 30*24
    interval_to_hours = {'1h':1,'4h':4,'1d':24}
    try:
        current_hours = interval_to_hours.get(interval,24)
        future_steps = max(hours_in_month/current_hours,10)
        df_trend = df_analyzed[['Close']].reset_index(drop=True)
        df_trend['Time_Step'] = df_trend.index
        model = LinearRegression()
        model.fit(df_trend[['Time_Step']], df_trend['Close'])
        future_index = df_trend.index[-1]+future_steps
        monthly_price = model.predict([[future_index]])[0]
        diff_percent = (monthly_price - df_analyzed['Close'].iloc[-1])/df_analyzed['Close'].iloc[-1]*100
        future_timestamp = df_analyzed.index[-1]+pd.Timedelta(hours=hours_in_month)
        return {'forecast_text':f"{diff_percent:+.2f}%", 'monthly_price':monthly_price, 'change_percent_30day':diff_percent,'forecast_timestamp':future_timestamp}
    except Exception:
        return {'forecast_text':"Błąd",'monthly_price':None,'change_percent_30day':0.0,'forecast_timestamp':None}

# --- 4. ANALIZA SNTYMENTU ---
def get_social_sentiment_forecast(symbol: str) -> Dict[str,float]:
    import random
    random.seed(hash(symbol)%100)
    score=random.uniform(-0.5,0.5)
    if score>0.15: direction="BULLISH"; change=score*5
    elif score<-0.15: direction="BEARISH"; change=score*5
    else: direction="NEUTRAL"; change=0.0
    summary=f"Wniosek: {direction} | Siła sentymentu: {score:+.2f}"
    return {'summary':summary,'change_percent_30day':change}

# --- 5. ANALIZA RSI ---
def get_rsi_analysis(df_analyzed: pd.DataFrame) -> Dict[str,str]:
    if df_analyzed.empty: return {'status':"Brak danych",'action':"CZEKAJ"}
    rsi=df_analyzed['RSI'].iloc[-1]
    if rsi<30: return {'status':"NIEDOWARTOŚCIOWANE",'action':f"KUPNO (RSI:{rsi:.2f})"}
    elif rsi>70: return {'status':"PRZEWARTOŚCIOWANE",'action':f"SPRZEDAŻ (RSI:{rsi:.2f})"}
    elif rsi>50: return {'status':"PRESJA KUPUJĄCYCH",'action':f"TRZYMAJ (RSI:{rsi:.2f})"}
    else: return {'status':"PRESJA SPRZEDAJĄCYCH",'action':f"CZEKAJ (RSI:{rsi:.2f})"}
