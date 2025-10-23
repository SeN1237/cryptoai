import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time

from crypto_analyzer import (
    fetch_top_symbols,
    fetch_crypto_data,
    technical_analysis,
    score_asset,
    get_rsi_analysis,
    get_social_sentiment_forecast,
    get_ml_forecast,
    get_ml_monthly_forecast
)

# ------------------------
# KONFIGURACJA STRONY
# ------------------------
st.set_page_config(
    layout="wide",
    page_title="Crypto AI Scanner PRO 🚀",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDataFrame {font-size: 14px; line-height: 1.2;}
.stContainer {border-radius: 10px; border:1px solid rgba(255,255,255,0.1); padding:10px; margin-bottom:10px; box-shadow:2px 2px 10px rgba(0,0,0,0.2);}
.stTextArea [data-baseweb="textarea"] {background-color: #262730; border-radius:5px; font-family: monospace;}
</style>
""", unsafe_allow_html=True)

MUST_SCAN_SYMBOLS = ['BTCUSDT','ETHUSDT','BNBUSDT','SOLUSDT','XRPUSDT','ADAUSDT','DOGEUSDT','AVAXUSDT','DOTUSDT','LINKUSDT','ZECUSDT']

# ------------------------
# FUNKCJA GŁÓWNA
# ------------------------
@st.cache_data(ttl=60*15)
def run_auto_scan(limit_symbols_scan, top_score_n, interval, add_delay=True):
    all_symbols_dynamic = fetch_top_symbols(limit=limit_symbols_scan)
    all_symbols_set = set(all_symbols_dynamic) | set(MUST_SCAN_SYMBOLS)
    all_symbols_to_scan = list(all_symbols_set)
    
    st.info(f"Skanuję {len(all_symbols_to_scan)} symboli ({len(MUST_SCAN_SYMBOLS)} stałych + dynamiczne) na interwale {interval}...")
    
    ranked_assets = []
    for i, symbol in enumerate(all_symbols_to_scan):
        try:
            df = fetch_crypto_data(symbol=symbol, interval=interval, limit=100)
            df_an = technical_analysis(df.copy())
            sentiment = get_social_sentiment_forecast(symbol)
            scored = score_asset(df_an, sentiment['change_percent_30day'])
            scored['symbol'] = symbol
            if scored['score'] > -100:
                ranked_assets.append(scored)
        except Exception:
            pass
    
    ranked_assets.sort(key=lambda x: x['score'], reverse=True)
    
    # Top assets + MUST_SCAN missing
    top_assets = ranked_assets[:top_score_n]
    top_symbols = {a['symbol'] for a in top_assets}
    must_missing = [s for s in MUST_SCAN_SYMBOLS if s not in top_symbols]
    for a in ranked_assets:
        if a['symbol'] in must_missing:
            top_assets.append(a)
    
    final_symbols = [a['symbol'] for a in top_assets]
    
    # 3xAI analysis
    results = {}
    for symbol in final_symbols:
        try:
            df = fetch_crypto_data(symbol=symbol, interval=interval, limit=100)
            df_an = technical_analysis(df.copy())
            rsi = get_rsi_analysis(df_an)
            ml1 = get_ml_forecast(df_an)
            ml30 = get_ml_monthly_forecast(df_an, interval)
            sentiment = get_social_sentiment_forecast(symbol)
            price = df['Close'].iloc[-1] if not df.empty else None
            
            asset = next((a for a in ranked_assets if a['symbol']==symbol), None)
            results[symbol] = {
                'data': df_an,
                'score': asset['score'] if asset else 0,
                'sugestion': asset['sugestion'] if asset else "Brak",
                'analysis_rsi': rsi,
                'forecast_1step_price': ml1['next_price'],
                'forecast_ml_price_30day': ml30['monthly_price'],
                'forecast_ml_percent': ml30['change_percent_30day'],
                'forecast_monthly_timestamp': ml30['forecast_timestamp'],
                'forecast_sentiment_percent': sentiment['change_percent_30day'],
                'forecast_sentiment_text': sentiment['summary']
            }
            
            if add_delay:
                time.sleep(0.5)  # krótszy delay, nie blokuje UI
        except Exception:
            continue
    
    return results, ranked_assets

# ------------------------
# PANEL BOCZNY
# ------------------------
with st.sidebar:
    st.header("Konfiguracja Skanera")
    limit_symbols_scan = st.slider("Liczba aktywów do wstępnego skanowania", 20, 200, 200, step=20)
    top_score_n = st.slider("Top X wg SCORE", 1, 15, 10)
    interval = st.selectbox("Interwał:", ['4h','1d','1h'], index=0)
    add_delay = st.checkbox("Dodaj krótką przerwę między analizami", True)
    
    if st.button("Uruchom / Odśwież 🔄"):
        st.cache_data.clear()
        st.experimental_rerun()

# ------------------------
# URUCHOMIENIE ANALIZY
# ------------------------
analysis_results, full_ranking = run_auto_scan(limit_symbols_scan, top_score_n, interval, add_delay)

if not analysis_results:
    st.error("❌ Brak danych do wyświetlenia. Spróbuj ponownie uruchomić skan.")
    st.stop()

# ------------------------
# TABELA TOP I POPULAR
# ------------------------
df_full = pd.DataFrame([
    {
        'Symbol': s.replace("USDT",""),
        'Score': res.get('score',0),
        'Sugestia': res.get('sugestion','Brak'),
        'ML Prognoza %': f"{res.get('forecast_ml_percent',0):+.2f}%",
        'ML Cena Prog.': f"${res.get('forecast_ml_price_30day',0):,.2f}" if res.get('forecast_ml_price_30day') else "N/A",
        'RSI Akcja': res.get('analysis_rsi',{}).get('action','Brak'),
        'Sentyment %': f"{res.get('forecast_sentiment_percent',0):+.2f}%",
        'ID': s
    }
    for s,res in analysis_results.items()
])

df_top = df_full.sort_values('Score',ascending=False).head(top_score_n).reset_index(drop=True)
df_top.index = df_top.index + 1

popular_symbols_short = [s.replace("USDT","") for s in MUST_SCAN_SYMBOLS]
df_pop = df_full[df_full['Symbol'].isin(popular_symbols_short)].sort_values('Score',ascending=False).reset_index(drop=True)
df_pop.index = df_pop.index + 1

col1, col2 = st.columns([1,1])
with col1:
    st.markdown(f"**🚀 Top {top_score_n} Zyskowne Okazje**")
    st.dataframe(df_top[['Symbol','Score','ML Prognoza %','RSI Akcja']], use_container_width=True, hide_index=True)
with col2:
    st.markdown("**⭐ Popularne Aktywa**")
    st.dataframe(df_pop[['Symbol','Score','ML Prognoza %','RSI Akcja']], use_container_width=True, hide_index=True)

# ------------------------
# WYKRESY I ANALIZA 3xAI
# ------------------------
st.header("🧠 Szczegółowa Analiza 3xAI")
selected_symbol_short = st.selectbox("Wybierz aktywo do wykresu:", df_full['Symbol'].tolist())
full_symbol = selected_symbol_short + "USDT"
res = analysis_results.get(full_symbol)

if res:
    df_data = res['data']
    prog1 = res['forecast_1step_price']
    prog30 = res['forecast_ml_price_30day']
    prog30_time = res['forecast_monthly_timestamp']

    # --- Wnioski metryki ---
    st.subheader(f"Wnioski dla {selected_symbol_short} (SCORE: {res['score']})")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("📈 Prognoza ML 30 dni %", res['forecast_ml_percent'], f"{res['forecast_ml_price_30day']:.2f}" if prog30 else "N/A")
    col_m2.metric("❤️ Sentyment 30 dni %", res['forecast_sentiment_percent'], res['forecast_sentiment_text'])
    col_m3.metric("🟢 RSI Akcja", res['analysis_rsi']['action'])
    col_m4.metric("Sugestia Algorytmu", res['sugestion'])

    # --- Wykres
    st.markdown("### Wykres świecowy + prognozy ML")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_data.index,
        open=df_data['Open'],
        high=df_data['High'],
        low=df_data['Low'],
        close=df_data['Close'],
        name='Cena'
    ))
    fig.add_trace(go.Scatter(x=df_data.index, y=df_data['SMA_20'], mode='lines', name='SMA20', line=dict(color='orange')))
    
    if prog1:
        fig.add_trace(go.Scatter(x=[df_data.index[-1]], y=[prog1], mode='markers+text', name='Prognoza 1 Interwał', marker=dict(color='red', size=15, symbol='star')))
    if prog30 and prog30_time:
        fig.add_trace(go.Scatter(x=[prog30_time], y=[prog30], mode='markers+text', name='Prognoza 30 dni', marker=dict(color='green', size=15, symbol='circle')))
        fig.add_trace(go.Scatter(x=[df_data.index[-1], prog30_time], y=[df_data['Close'].iloc[-1], prog30], mode='lines', line=dict(dash='dash', color='green'), name='Linia trendu 30 dni'))

    fig.update_layout(height=550, xaxis_rangeslider_visible=False, title=f"{selected_symbol_short} - Wykres świecowy ({interval})")
    st.plotly_chart(fig, use_container_width=True)

    # --- Tekstowe wnioski ---
    st.markdown("### Surowe wnioski analityczne")
    col_ml, col_sentiment, col_rsi = st.columns(3)
    with col_ml:
        st.text_area("Prognoza ML 30 dni", res['forecast_ml_percent'], height=150)
    with col_sentiment:
        st.text_area("Sentyment społeczności", res['forecast_sentiment_text'], height=150)
    with col_rsi:
        st.text_area("RSI i wnioski techniczne", res['analysis_rsi']['action'] + "\n" + res['analysis_rsi']['status'], height=150)

st.markdown(f"_Ostatnia aktualizacja: {time.strftime('%H:%M:%S', time.localtime())}_")
