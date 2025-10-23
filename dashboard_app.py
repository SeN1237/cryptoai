import streamlit as st
import pandas as pd
import plotly.graph_objects as go 
from crypto_analyzer import (
    fetch_top_symbols, fetch_crypto_data, technical_analysis, score_asset, 
    get_rsi_analysis, get_social_sentiment_forecast, get_ml_forecast, get_ml_monthly_forecast
)
import time 

# --- KONFIGURACJA STRONY ---
st.set_page_config(layout="wide", page_title="Crypto AI Scanner PRO 🚀", initial_sidebar_state="expanded")

# Styl CSS dla nowoczesnego wyglądu
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDataFrame {font-size: 14px; line-height: 1.2;}
    .stContainer {border-radius: 10px; border: 1px solid rgba(255,255,255,0.1); padding: 10px; margin-bottom: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.2);}
    .stTextArea [data-baseweb="textarea"] {background-color: #262730; border-radius: 5px; font-family: monospace;}
    </style>
""", unsafe_allow_html=True)

# Stała lista tokenów
MUST_SCAN_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT', 'ZECUSDT']

# --- LOGIKA APLIKACJI ---
@st.cache_data(ttl=60*15)
def run_auto_scan_and_analysis(limit_symbols_scan, top_score_n, interval, add_delay): 
    all_symbols_dynamic = fetch_top_symbols(limit=limit_symbols_scan)
    all_symbols_set = set(all_symbols_dynamic) | set(MUST_SCAN_SYMBOLS)
    all_symbols_to_scan = list(all_symbols_set)
    
    st.info(f"Skanuję łącznie {len(all_symbols_to_scan)} par ({len(MUST_SCAN_SYMBOLS)} stałych + dynamiczne) na interwale {interval}...")
    
    ranked_assets = []
    progress_bar_scan = st.progress(0, text="Faza 1/2: Wstępne skanowanie wskaźników...")
    
    for i, symbol in enumerate(all_symbols_to_scan):
        progress_bar_scan.progress((i + 1) / len(all_symbols_to_scan), text=f"Faza 1/2: Skanowanie: {symbol}...")
        try:
            df = fetch_crypto_data(symbol=symbol, interval=interval, limit=100)
            df_analyzed = technical_analysis(df.copy())
            asset_score = score_asset(df_analyzed)
            if asset_score['score'] > -100:
                asset_score['symbol'] = symbol
                ranked_assets.append(asset_score)
        except Exception:
            continue
            
    progress_bar_scan.empty()
    
    final_ranking = sorted(ranked_assets, key=lambda x: x['score'], reverse=True)
    
    top_score_assets_list = final_ranking[:top_score_n]
    top_score_symbols = {asset['symbol'] for asset in top_score_assets_list}
    must_scan_not_in_top = [s for s in MUST_SCAN_SYMBOLS if s not in top_score_symbols]
    top_assets_for_ai = top_score_assets_list + [asset for asset in ranked_assets if asset['symbol'] in must_scan_not_in_top]
    final_symbols_for_ai = list(set([asset['symbol'] for asset in top_assets_for_ai]))
    
    st.subheader(f"Pobieranie szczegółowej analizy 3xAI dla {len(final_symbols_for_ai)} aktywów...")
    results = {}
    total_top_n = len(final_symbols_for_ai)
    
    progress_container = st.container()
    progress_bar_ai = progress_container.progress(0)
    progress_text_ai = progress_container.empty()
    
    for i, symbol in enumerate(final_symbols_for_ai):
        asset = next(a for a in ranked_assets if a['symbol'] == symbol)
        progress_bar_ai.progress((i + 1)/total_top_n)
        progress_text_ai.text(f"Faza 2/2: Analiza AI dla {symbol} ({i + 1}/{total_top_n})")
        
        df = fetch_crypto_data(symbol=symbol, interval=interval, limit=100)
        df_analyzed = technical_analysis(df.copy())
        
        rsi_result = get_rsi_analysis(df_analyzed)
        sentiment_result = get_social_sentiment_forecast(symbol)
        ml_result_1step = get_ml_forecast(df_analyzed)
        ml_result_30day = get_ml_monthly_forecast(df_analyzed, interval)
        
        results[symbol] = {
            'data': df_analyzed,
            'analysis_rsi': rsi_result,
            'forecast_ml_percent': ml_result_30day['change_percent_30day'], 
            'forecast_ml_price_30day': ml_result_30day['monthly_price'],
            'forecast_ml_text': ml_result_30day['forecast_text'],
            'forecast_sentiment_percent': sentiment_result['change_percent_30day'],
            'forecast_sentiment_text': sentiment_result['summary'],
            'forecast_1step_price': ml_result_1step['next_price'],
            'forecast_monthly_timestamp': ml_result_30day['forecast_timestamp'],
            'score': asset['score'],
            'sugestion': asset['sugestion']
        }
        
        if add_delay and (i + 1) < total_top_n:
            time.sleep(30) 

    progress_bar_ai.empty()
    progress_text_ai.empty()
    
    return results, final_ranking

# --- PANEL BOCZNY ---
with st.sidebar:
    st.header("Konfiguracja Skanera")
    limit_symbols_scan = st.slider("Liczba Aktywów do Wstępnego Skanowania:", 20, 200, 200, 20)
    top_score_n = st.slider("Pokaż Top X Zyskownych Okazji:", 1, 15, 10, 1)
    interval = st.selectbox("Interwał Czasowy:", ('4h', '1d', '1h'), index=0)
    add_delay = st.checkbox("Dodaj 30s przerwę między analizami", value=True)
    
    if st.button("Uruchom Skan Rynku / Odśwież teraz 🔄"):
        st.cache_data.clear() 
        st.experimental_rerun()

# --- URUCHOMIENIE ANALIZY ---
analysis_results, full_ranking = run_auto_scan_and_analysis(limit_symbols_scan, top_score_n, interval, add_delay) 

st.header(f"📊 Aktualny Skan Rynku (Interwał: {interval})")
st.write("🧩 DEBUG: liczba elementów w analysis_results =", len(analysis_results))

df_full_analysis = pd.DataFrame([
    {
        'Symbol': s.replace("USDT", ""), 
        'Score': res.get('score', 0), 
        'Sugestia': res.get('sugestion', 'Brak'), 
        'ML Prognoza %': f"{res.get('forecast_ml_percent', 0):+.2f}%", 
        'ML Cena Prog.': f"${res.get('forecast_ml_price_30day', 0):,.2f}" if res.get('forecast_ml_price_30day') is not None else "N/A",
        'RSI Akcja': res.get('analysis_rsi', {}).get('action', 'Brak'),
        'Sentyment %': f"{res.get('forecast_sentiment_percent', 0):+.2f}%",
        'ID': s
    } 
    for s, res in analysis_results.items()
])

if df_full_analysis.empty:
    st.error("❌ Brak danych do wyświetlenia — żadna analiza nie zwróciła wyników.")
    st.info("Spróbuj ponownie uruchomić skan.")
    st.stop()

# --- TOP i Popularne ---
df_top_score = df_full_analysis.sort_values('Score', ascending=False).head(top_score_n).reset_index(drop=True)
df_top_score.index += 1
popular_symbols_short = [s.replace('USDT', '') for s in MUST_SCAN_SYMBOLS]
df_popular = df_full_analysis[df_full_analysis['Symbol'].isin(popular_symbols_short)].sort_values('Score', ascending=False).reset_index(drop=True)
df_popular.index += 1

col_zyski, col_popularne = st.columns(2)
with col_zyski:
    st.markdown(f"**🚀 Top {top_score_n} Zyskowne Okazje**")
    st.dataframe(df_top_score[['Symbol', 'Score', 'ML Prognoza %', 'RSI Akcja']], use_container_width=True, hide_index=True)

with col_popularne:
    st.markdown("**⭐ Popularne Aktywa**")
    st.dataframe(df_popular[['Symbol', 'Score', 'ML Prognoza %', 'RSI Akcja']], use_container_width=True, hide_index=True)
