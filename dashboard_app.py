import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
from crypto_analyzer import (
    fetch_top_symbols, fetch_crypto_data, technical_analysis, score_asset,
    get_rsi_analysis, get_social_sentiment_forecast, get_ml_forecast, get_ml_monthly_forecast
)

# --- KONFIGURACJA STRONY ---
st.set_page_config(layout="wide", page_title="Crypto AI Scanner PRO 🚀", initial_sidebar_state="expanded")

# --- CSS ---
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDataFrame {font-size: 14px; line-height:1.2;}
.stContainer {border-radius:10px; border:1px solid rgba(255,255,255,0.1); padding:10px; margin-bottom:10px; box-shadow:2px 2px 10px rgba(0,0,0,0.2);}
.stTextArea [data-baseweb="textarea"] {background-color:#262730;border-radius:5px;font-family:monospace;}
</style>
""", unsafe_allow_html=True)

# --- Stałe tokeny ---
MUST_SCAN_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
                     'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT', 'ZECUSDT']

# --- FUNKCJA GŁÓWNA SKANU ---
@st.cache_data(ttl=60*15)
def run_auto_scan_and_analysis(limit_symbols_scan, top_score_n, interval, add_delay):
    # 1. Pobranie listy symboli
    dynamic_symbols = fetch_top_symbols(limit=limit_symbols_scan)
    all_symbols = list(set(dynamic_symbols) | set(MUST_SCAN_SYMBOLS))

    st.info(f"Skanuję {len(all_symbols)} par (stałe + dynamiczne) na interwale {interval}...")

    ranked_assets = []

    for i, sym in enumerate(all_symbols):
        try:
            df = fetch_crypto_data(sym, interval, limit=100)
            df_analyzed = technical_analysis(df.copy())
            asset_score = score_asset(df_analyzed)
            asset_score['symbol'] = sym
            ranked_assets.append(asset_score)
        except Exception:
            # fallback: minimalny wynik
            ranked_assets.append({'symbol': sym, 'score':0, 'sugestion':'Brak danych', 'data':None})

    final_ranking = sorted(ranked_assets, key=lambda x: x['score'], reverse=True)

    # Top N + must scan
    top_score_assets = final_ranking[:top_score_n]
    top_symbols = {a['symbol'] for a in top_score_assets}
    must_scan_missing = [s for s in MUST_SCAN_SYMBOLS if s not in top_symbols]
    final_symbols_for_ai = list({a['symbol'] for a in top_score_assets + [a for a in ranked_assets if a['symbol'] in must_scan_missing]})

    # --- 2. Analiza 3xAI ---
    results = {}
    for i, sym in enumerate(final_symbols_for_ai):
        asset = next(a for a in ranked_assets if a['symbol']==sym)
        try:
            df = fetch_crypto_data(sym, interval, limit=100)
            df_analyzed = technical_analysis(df.copy())

            rsi_res = get_rsi_analysis(df_analyzed)
            sentiment_res = get_social_sentiment_forecast(sym)
            ml_1step = get_ml_forecast(df_analyzed)
            ml_30day = get_ml_monthly_forecast(df_analyzed, interval)

            results[sym] = {
                'data': df_analyzed,
                'analysis_rsi': rsi_res,
                'forecast_ml_percent': ml_30day['change_percent_30day'],
                'forecast_ml_price_30day': ml_30day['monthly_price'],
                'forecast_ml_text': ml_30day['forecast_text'],
                'forecast_sentiment_percent': sentiment_res['change_percent_30day'],
                'forecast_sentiment_text': sentiment_res['summary'],
                'forecast_1step_price': ml_1step['next_price'],
                'forecast_monthly_timestamp': ml_30day['forecast_timestamp'],
                'score': asset['score'],
                'sugestion': asset['sugestion']
            }
        except Exception:
            # fallback minimalny
            results[sym] = {
                'data': pd.DataFrame(),
                'analysis_rsi': {'status':'Brak danych','action':'CZEKAJ'},
                'forecast_ml_percent':0,
                'forecast_ml_price_30day':None,
                'forecast_ml_text':'Brak danych',
                'forecast_sentiment_percent':0,
                'forecast_sentiment_text':'Brak danych',
                'forecast_1step_price':None,
                'forecast_monthly_timestamp':None,
                'score': asset['score'],
                'sugestion': asset['sugestion']
            }

        if add_delay and i+1<len(final_symbols_for_ai):
            time.sleep(1)  # krótsza przerwa, żeby nie blokować w demo

    return results, final_ranking


# --- PANEL BOCZNY ---
with st.sidebar:
    st.header("Konfiguracja Skanera")
    limit_symbols_scan = st.slider("Ilość aktywów do wstępnego skanowania", 20, 200, 50, step=10)
    top_score_n = st.slider("Top X wg SCORE", 1, 15, 10)
    interval = st.selectbox("Interwał czasowy", ['1h','4h','1d'], index=1)
    add_delay = st.checkbox("Dodaj krótką przerwę między analizami", value=True)

    if st.button("Uruchom / Odśwież 🔄"):
        st.cache_data.clear()
        st.experimental_rerun()


# --- URUCHOMIENIE ANALIZY ---
analysis_results, full_ranking = run_auto_scan_and_analysis(limit_symbols_scan, top_score_n, interval, add_delay)

st.header(f"📊 Aktualny Skan Rynku ({interval})")
st.write("🧩 DEBUG: liczba elementów w analysis_results =", len(analysis_results))

if not analysis_results:
    st.error("❌ Brak danych do wyświetlenia po wstępnym skanowaniu. Sprawdź połączenie z Binance API lub spróbuj ponownie.")
    st.stop()


# --- Tworzenie DataFrame dla tabel ---
df_full = pd.DataFrame([
    {
        'Symbol': s.replace('USDT',''),
        'Score': r.get('score',0),
        'Sugestia': r.get('sugestion','Brak'),
        'ML Prognoza %': f"{r.get('forecast_ml_percent',0):+.2f}%",
        'ML Cena Prog.': f"${r.get('forecast_ml_price_30day',0):,.2f}" if r.get('forecast_ml_price_30day') else "N/A",
        'RSI Akcja': r.get('analysis_rsi',{}).get('action','Brak'),
        'Sentyment %': f"{r.get('forecast_sentiment_percent',0):+.2f}%",
        'ID': s
    } for s,r in analysis_results.items()
])

df_top = df_full.sort_values('Score',ascending=False).head(top_score_n).reset_index(drop=True)
df_top.index += 1
popular_symbols_short = [s.replace('USDT','') for s in MUST_SCAN_SYMBOLS]
df_popular = df_full[df_full['Symbol'].isin(popular_symbols_short)].sort_values('Score',ascending=False).reset_index(drop=True)
df_popular.index += 1

# --- WYŚWIETLANIE TABEL ---
col1,col2 = st.columns(2)

with col1:
    st.markdown(f"**🚀 Top {top_score_n} Zyskowne Okazje**")
    st.dataframe(df_top[['Symbol','Score','ML Prognoza %','RSI Akcja']], use_container_width=True)

with col2:
    st.markdown("**⭐ Popularne Aktywa**")
    st.dataframe(df_popular[['Symbol','Score','ML Prognoza %','RSI Akcja']], use_container_width=True)

# --- WYKRESY ---
st.markdown("---")
st.header("🧠 Szczegółowa Analiza 3xAI")
selected_symbol = st.selectbox("Wybierz aktywo do analizy wykresu", options=df_full['Symbol'].tolist())

if selected_symbol:
    full_symbol = f"{selected_symbol}USDT"
    res = analysis_results.get(full_symbol)
    df_data = res['data'] if res['data'] is not None else pd.DataFrame({'Close':[0],'Open':[0],'High':[0],'Low':[0],'Volume':[0]}, index=[pd.Timestamp.now()])

    fig = go.Figure(data=[
        go.Candlestick(x=df_data.index, open=df_data['Open'], high=df_data['High'], low=df_data['Low'], close=df_data['Close'], name='Cena'),
        go.Scatter(x=df_data.index, y=df_data['SMA_20'] if 'SMA_20' in df_data.columns else df_data['Close'], mode='lines', name='SMA20', line=dict(color='orange'))
    ])
    # ML punkty
    if res.get('forecast_1step_price'):
        fig.add_trace(go.Scatter(x=[df_data.index[-1]], y=[res['forecast_1step_price']], mode='markers', name='Prognoza 1 Interwał', marker=dict(color='red', size=12, symbol='star')))
    if res.get('forecast_ml_price_30day') and res.get('forecast_monthly_timestamp'):
        fig.add_trace(go.Scatter(x=[res['forecast_monthly_timestamp']], y=[res['forecast_ml_price_30day']], mode='markers', name='Prognoza 30 Dni', marker=dict(color='green', size=12, symbol='circle')))

    fig.update_layout(height=500, xaxis_rangeslider_visible=False, title=f"{selected_symbol} - {interval} z prognozą ML", yaxis_title="Cena (USDT)")
    st.plotly_chart(fig, use_container_width=True)

    # Wnioski
    col_ml,col_sent,col_rsi = st.columns(3)
    with col_ml:
        st.markdown("**Prognoza ML (30 dni)**")
        st.text_area(label="", value=res['forecast_ml_text'], height=150, label_visibility='collapsed')
    with col_sent:
        st.markdown("**Sentyment Społeczności**")
        st.text_area(label="", value=res['forecast_sentiment_text'], height=150, label_visibility='collapsed')
    with col_rsi:
        st.markdown("**Wnioski Techniczne (RSI)**")
        st.text_area(label="", value=f"{res['analysis_rsi']['action']}\n{res['analysis_rsi']['status']}", height=150, label_visibility='collapsed')

st.markdown(f"_Ostatnia aktualizacja: {time.strftime('%H:%M:%S', time.localtime())}_")
