# features_news.py
import pandas as pd
import requests
from datetime import datetime, timedelta

# --- KONFIGURACJA NEWS ---
NEWSAPI_KEY = "42bab953546d4558be2a73815e8eae92"
FINNHUB_KEY = "d2re62pr01qlk22sttf0d2re62pr01qlk22sttfg"

TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","META","TSLA","NVDA","NFLX","ADBE","INTC",
    "PYPL","CRM","ORCL","CSCO","IBM","INTU","PEP","TXN","QCOM","AVGO",
    "SBUX","BABA","AMD","UBER","SHOP","TWLO","SPOT","HON","V","BKNG",
    "MDLZ","DOCU","ISRG","ADI","MU","CRWD","NVDA","ZM","ROKU","AMAT",
    "OKTA","TSM","GILD","SNOW","GOOG","PLTR","IBM","MSFT","AAPL","NFLX"
]

def get_news_sentiment(ticker, days=365):
    """
    Pobiera newsy z NewsAPI i Finnhub i zwraca średni sentyment dla ticker w ostatnich 'days' dniach.
    """
from datetime import datetime, timedelta

def build_news_features(tickers, days=365):
    end_date = datetime.today()  # <-- dzisiaj
    start_date = end_date - timedelta(days=days)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    # --- NewsAPI ---
    url_newsapi = f"https://newsapi.org/v2/everything?q={ticker}&from={start_str}&to={end_str}&language=en&apiKey={NEWSAPI_KEY}"
    try:
        res = requests.get(url_newsapi).json()
        articles = res.get("articles", [])
    except:
        articles = []

    # --- Finnhub ---
    url_finnhub = f"https://finnhub.io/api/v1/news-sentiment?symbol={ticker}&token={FINNHUB_KEY}"
    try:
        res2 = requests.get(url_finnhub).json()
        finnhub_sentiment = res2.get("score", {}).get("avg", 0)
    except:
        finnhub_sentiment = 0

    # Prosta heurystyka NewsAPI: +1 jeśli "up"/"gain"/"bull", -1 jeśli "down"/"loss"/"bear"
    sentiment_sum = 0
    for article in articles:
        title = article.get('title') or ""
        desc  = article.get('description') or ""
        text = (title + " " + desc).lower()
        score = 0
        if any(w in text for w in ['up','gain','rise','bull']):
            score += 1
        if any(w in text for w in ['down','loss','fall','bear']):
            score -= 1
        sentiment_sum += score

    newsapi_sentiment = sentiment_sum / len(articles) if articles else 0
    sentiment = 0.5 * newsapi_sentiment + 0.5 * finnhub_sentiment
    return sentiment

def build_news_features(tickers=TICKERS, days=365):
    """
    Buduje DataFrame: ticker | date | sentiment
    """
    rows = []
    today = datetime.today().date()
    for t in tickers:
        sentiment = get_news_sentiment(t, days=days)
        rows.append({'ticker': t, 'date': today, 'sentiment': sentiment})
    df = pd.DataFrame(rows)
    return df

