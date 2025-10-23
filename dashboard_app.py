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
    .stContainer {border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.1); padding: 10px; margin-bottom: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.2);}
    .stTextArea [data-baseweb="textarea"] {background-color: #262730; border-radius: 5px; font-family: monospace;}
    </style>
""", unsafe_allow_html=True)

# Stała lista tokenów
MUST_SCAN_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 
                     'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT', 'ZECUSDT']

# --- FUNKCJA GŁÓWNA Z DEBUGIEM ---
@st.cache_data(ttl=60*15)
def run_auto_scan_and_analysis(limit_symbols_scan, top_score_n, interval, add_delay):
    
    all_symbols_dynamic = fetch_top_symbols(limit=limit_symbols_scan)
    all_symbols_set = set(all_symbols_dynamic) | set(MUST_SCAN_SYMBOLS)
    all_symbols_to_scan = list(all_symbols_set)
    
    st.info(f"Skanuję łącznie {len(all_symbols_to_scan)} par ({len(MUST_SCAN_SYMBOLS)} stałych + dynamiczne) na interwale {interval}...")
    
    ranked_assets = []
    
    # Faza 1: Skanowanie techniczne
    progress_bar_scan = st.progress(0, text="Faza 1/2: Wstępne skanowanie wskaźników...")
    
    for i, symbol in enumerate(all_symbols_to_scan):
        progress_bar_scan.progress((i+1)/len(all_symbols_to_scan), text=f"Skanowanie: {symbol} ({i+1}/{len(all_symbols_to_scan)})")
        try:
            df = fetch_crypto_data(symbol=symbol, interval=interval, limit=100)
            if df.empty:
                print(f"[WARN] Brak danych dla {symbol}")
                continue
            df_analyzed = technical_analysis(df.copy())
            scored = score_asset(df_analyzed)
            if scored['score'] > -100:
                scored['symbol'] = symbol
                ranked_assets.append(scored)
            else:
                print(f"[INFO] {symbol} odrzucone (score={scored['score']})")
        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")
            continue
    
    progress_bar_scan.empty()
    
    if not ranked_assets:
        st.error("❌ Brak danych do wyświetlenia po wstępnym skanowaniu. Sprawdź połączenie z Binance API.")
        return {}, []
    
    final_ranking = sorted(ranked_assets, key=lambda x: x['score'], reverse=True)
    
    # Faza 2: Selekcja top + MUST_SCAN
    top_score_assets_list = final_ranking[:top_score_n]
    top_score_symbols = {a['symbol'] for a in top_score_assets_list}
    must_scan_not_in_top = [s for s in MUST_SCAN_SYMBOLS if s not in top_score_symbols]
    top_assets_for_ai = top_score_assets_list + [a for a in ranked_assets if a['symbol'] in must_scan_not_in_top]
    final_symbols_for_ai = list(set([a['symbol'] for a in top_assets_for_ai]))
    
    st.subheader(f"Pobieranie szczegółowej analizy 3xAI dla {len(final_symbols_for_ai)} aktywów...")
    results = {}
    progress_container = st.container()
    progress_bar_ai = progress_container.progress(0)
    progress_text_ai = progress_container.empty()
    
    # Faza 3: Analiza 3xAI
    for i, symbol in enumerate(final_symbols_for_ai):
        asset = next(a for a in ranked_assets if a['symbol'] == symbol)
        progress_bar_ai.progress((i+1)/len(final_symbols_for_ai))
        progress_text_ai.text(f"Analiza AI dla {symbol} ({i+1}/{len(final_symbols_for_ai)})")
        try:
            df = fetch_crypto_data(symbol=symbol, interval=interval, limit=100)
            if df.empty:
                print(f"[WARN] Brak danych dla {symbol} w fazie AI")
                continue
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
            
            if add_delay and (i+1)<len(final_symbols_for_ai):
                next_symbol = final_symbols_for_ai[i+1]
                progress_text_ai.text(f"Analiza dla {symbol} zakończona. Wstrzymuję 3 sekundy (test delay). Następny: {next_symbol}")
                time.sleep(3)
                
        except Exception as e:
            print(f"[ERROR] 3xAI dla {symbol}: {e}")
            continue
    
    progress_bar_ai.empty()
    progress_text_ai.empty()
    
    return results, final_ranking


# --- PANEL BOCZNY ---
with st.sidebar:
    st.header("Konfiguracja Skanera")
    limit_symbols_scan = st.slider("Liczba Aktywów do Wstępnego Skanowania:", min_value=20, max_value=200, value=50, step=10)
    top_score_n = st.slider("Top X Zyskownych Okazji (wg SCORE):", min_value=1, max_value=15, value=5)
    interval = st.selectbox("Interwał Czasowy:", ('4h', '1d', '1h'), index=0)
    add_delay = st.checkbox("Dodaj krótką przerwę między analizami", value=True)
    
    if st.button("Uruchom Skan / Odśwież 🔄"):
        st.cache_data.clear()
        st.experimental_rerun()

# --- URUCHOMIENIE FUNKCJI ---
analysis_results, full_ranking = run_auto_scan_and_analysis(limit_symbols_scan, top_score_n, interval, add_delay)

# --- TWORZENIE TABEL ---
st.header(f"📊 Aktualny Skan Rynku ({interval})")
st.write("🧩 DEBUG: liczba elementów w analysis_results =", len(analysis_results))

df_full_analysis = pd.DataFrame([
    {
        'Symbol': s.replace("USDT",""),
        'Score': res.get('score',0),
        'Sugestia': res.get('sugestion','Brak'),
        'ML Prognoza %': f"{res.get('forecast_ml_percent',0):+.2f}%",
        'ML Cena Prog.': f"${res.get('forecast_ml_price_30day',0):,.2f}" if res.get('forecast_ml_price_30day') is not None else "N/A",
        'RSI Akcja': res.get('analysis_rsi',{}).get('action','Brak'),
        'Sentyment %': f"{res.get('forecast_sentiment_percent',0):+.2f}%",
        'ID': s
    } for s,res in analysis_results.items()
])

if df_full_analysis.empty:
    st.error("❌ Brak danych do wyświetlenia — żadna analiza nie zwróciła wyników.")
    st.stop()

# --- Tabele Top i Popular ---
df_top_score = df_full_analysis.sort_values(by='Score', ascending=False).head(top_score_n).reset_index(drop=True)
df_top_score.index += 1
popular_symbols_short = [s.replace('USDT','') for s in MUST_SCAN_SYMBOLS]
df_popular = df_full_analysis[df_full_analysis['Symbol'].isin(popular_symbols_short)].sort_values(by='Score', ascending=False).reset_index(drop=True)
df_popular.index += 1

col_zyski, col_popularne = st.columns([1,1])

with col_zyski:
    st.markdown(f"**🚀 Top {top_score_n} Zyskowne Okazje**")
    st.dataframe(df_top_score[['Symbol','Score','ML Prognoza %','RSI Akcja']], use_container_width=True, hide_index=True)

with col_popularne:
    st.markdown("**⭐ Popularne Aktywa**")
    st.dataframe(df_popular[['Symbol','Score','ML Prognoza %','RSI Akcja']], use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown(f"_Ostatnia aktualizacja: **{time.strftime('%H:%M:%S')}**_")
