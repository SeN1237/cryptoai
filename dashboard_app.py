import streamlit as st
import pandas as pd
import plotly.graph_objects as go 
from crypto_analyzer import (
    fetch_top_symbols, fetch_crypto_data, technical_analysis, score_asset, 
    get_rsi_analysis, get_social_sentiment_forecast, get_ml_forecast, get_ml_monthly_forecast # POPRAWIONY IMPORT
)
import time 

# --- KONFIGURACJA STRONY ---
st.set_page_config(layout="wide", page_title="Crypto AI Scanner PRO 🚀", initial_sidebar_state="expanded")

# Styl CSS dla nowoczesnego wyglądu
st.markdown("""
    <style>
    /* Ukrycie menu Streamlit i stopki */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Nadanie stylu tabeli w kontenerach */
    .stDataFrame {
        font-size: 14px;
        line-height: 1.2;
    }
    
    /* Globalne cienie i zaokrąglenia */
    .stContainer {
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 10px;
        margin-bottom: 10px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
    }
    
    /* Styl dla text_area (czyste fakty) */
    .stTextArea [data-baseweb="textarea"] {
        background-color: #262730; 
        border-radius: 5px;
        font-family: monospace;
    }
    </style>
""", unsafe_allow_html=True)


# Stała lista tokenów (Popularne Top 10 + ZCASH)
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
            pass
            
    progress_bar_scan.empty()
    
    final_ranking = sorted(ranked_assets, key=lambda x: x['score'], reverse=True)
    
    # SELEKCJA GRUP DLA ANALIZY 3xAI
    top_score_assets_list = final_ranking[:top_score_n]
    top_score_symbols = {asset['symbol'] for asset in top_score_assets_list}
    must_scan_not_in_top = [s for s in MUST_SCAN_SYMBOLS if s not in top_score_symbols]
    
    top_assets_for_ai = top_score_assets_list + [asset for asset in ranked_assets if asset['symbol'] in must_scan_not_in_top]
    final_symbols_for_ai = list(set([asset['symbol'] for asset in top_assets_for_ai]))
    
    
    # 3. FAZA ANALIZY 3xAI
    st.subheader(f"Pobieranie szczegółowej analizy 3xAI dla {len(final_symbols_for_ai)} aktywów...")
    
    results = {}
    total_top_n = len(final_symbols_for_ai)
    
    progress_container = st.container()
    progress_bar_ai = progress_container.progress(0)
    progress_text_ai = progress_container.empty()
    
    for i, symbol in enumerate(final_symbols_for_ai):
        asset = next(a for a in ranked_assets if a['symbol'] == symbol)
        
        progress = (i + 1) / total_top_n
        progress_bar_ai.progress(progress)
        progress_text_ai.text(f"Faza 2/2: Analiza AI dla {symbol} ({i + 1} z {total_top_n} | Pozostało: {total_top_n - (i + 1)})")
        
        df = fetch_crypto_data(symbol=symbol, interval=interval, limit=100)
        df_analyzed = technical_analysis(df.copy())
        
        # ⚠️ Wywołania 3 modeli/metod:
        rsi_result = get_rsi_analysis(df_analyzed)
        sentiment_result = get_social_sentiment_forecast(symbol)
        ml_result_1step = get_ml_forecast(df_analyzed) # Prognoza 1 interwał
        ml_result_30day = get_ml_monthly_forecast(df_analyzed, interval) # Prognoza 30 dni
        
        results[symbol] = {
            'data': df_analyzed,
            'analysis_rsi': rsi_result,
            
            # WYNIKI 30 DNI (dla metryk)
            'forecast_ml_percent': ml_result_30day['change_percent_30day'], 
            'forecast_ml_price_30day': ml_result_30day['monthly_price'],
            'forecast_ml_text': ml_result_30day['forecast_text'],
            'forecast_sentiment_percent': sentiment_result['change_percent_30day'],
            'forecast_sentiment_text': sentiment_result['summary'],
            
            # WYNIKI DLA WYKRESU
            'forecast_1step_price': ml_result_1step['next_price'],
            'forecast_monthly_timestamp': ml_result_30day['forecast_timestamp'],
            
            'score': asset['score'],
            'sugestion': asset['sugestion']
        }
        
        if add_delay and (i + 1) < total_top_n:
            next_symbol = final_symbols_for_ai[i+1]
            progress_text_ai.text(f"Analiza dla {symbol} zakończona. Wstrzymuję na 30 sekund. Następny: {next_symbol}")
            time.sleep(30) 

    progress_bar_ai.empty()
    progress_text_ai.empty()
    
    return results, final_ranking


# ----------------------------------------------------------------------------------
# PANEL BOCZNY (WYWOŁANIE I WIZUALIZACJA)
# ----------------------------------------------------------------------------------

# PANEL BOCZNY - Konfiguracja
with st.sidebar:
    st.header("Konfiguracja Skanera")
    limit_symbols_scan = st.slider("Liczba Aktywów do Wstępnego Skanowania:", min_value=20, max_value=200, value=200, step=20)
    top_score_n = st.slider("Pokaż Top X Zyskownych Okazji (wg SCORE):", min_value=1, max_value=15, value=10, step=1)
    interval = st.selectbox("Interwał Czasowy:", ('4h', '1d', '1h'), index=0)
    add_delay = st.checkbox("Dodaj 30s przerwę między analizami", value=True) 
    
    if st.button("Uruchom Skan Rynku / Odśwież teraz 🔄"):
        st.cache_data.clear() 
        st.experimental_rerun()


# Uruchomienie głównej funkcji analizy
analysis_results, full_ranking = run_auto_scan_and_analysis(limit_symbols_scan, top_score_n, interval, add_delay) 

# --- TWORZENIE TABEL DANYCH (Z DODANYM DEBUGEM I OBSŁUGĄ BRAKU DANYCH) ---
st.header("📊 Aktualny Skan Rynku (Interwał: " + interval + ")")

st.write("🧩 DEBUG: liczba elementów w analysis_results =", len(analysis_results))

df_full_analysis = pd.DataFrame([
    {
        'Symbol': s.replace("USDT", ""), 
        'Score': res.get('score', 0), 
        'Sugestia': res.get('sugestion', 'Brak'), 
        'ML Prognoza %': f"{res.get('forecast_ml_percent', 0):+.2f}%", 
        'ML Cena Prog.': (
            f"${res.get('forecast_ml_price_30day', 0):,.2f}"
            if res.get('forecast_ml_price_30day') is not None else "N/A"
        ),
        'RSI Akcja': res.get('analysis_rsi', {}).get('action', 'Brak'),
        'Sentyment %': f"{res.get('forecast_sentiment_percent', 0):+.2f}%",
        'ID': s
    } 
    for s, res in analysis_results.items()
])

if df_full_analysis.empty:
    st.error("❌ Brak danych do wyświetlenia — żadna analiza nie zwróciła wyników.")
    st.info("Spróbuj ponownie uruchomić skan (kliknij 'Uruchom Skan Rynku / Odśwież teraz 🔄').")
    st.stop()


# --- KONTYNUACJA NORMALNEGO DZIAŁANIA ---
df_top_score = df_full_analysis.sort_values(by='Score', ascending=False).head(top_score_n).reset_index(drop=True)
df_top_score.index = df_top_score.index + 1
popular_symbols_short = [s.replace('USDT', '') for s in MUST_SCAN_SYMBOLS]
df_popular = df_full_analysis[df_full_analysis['Symbol'].isin(popular_symbols_short)].sort_values(by='Score', ascending=False).reset_index(drop=True)
df_popular.index = df_popular.index + 1
chart_symbols = df_full_analysis['Symbol'].tolist()

col_zyski, col_popularne = st.columns([1, 1])

with col_zyski:
    with st.container():
        st.markdown(f"**🚀 Top {top_score_n} Zyskowne Okazje** | _Prognoza 30 Dni_")
        st.dataframe(df_top_score[['Symbol', 'Score', 'ML Prognoza %', 'RSI Akcja']], use_container_width=True, hide_index=True)
    
with col_popularne:
    with st.container():
        st.markdown(f"**⭐ Popularne Aktywa** | _Stała Lista wg Potencjału 30 Dni_")
        st.dataframe(df_popular[['Symbol', 'Score', 'ML Prognoza %', 'RSI Akcja']], use_container_width=True, hide_index=True)



# --- UKŁAD GŁÓWNY: DWIE KOLUMNY Z TABELAMI ---

st.header("📊 Aktualny Skan Rynku (Interwał: " + interval + ")")

col_zyski, col_popularne = st.columns([1, 1])

with col_zyski:
    with st.container():
        st.markdown(f"**🚀 Top {top_score_n} Zyskowne Okazje** | _Prognoza 30 Dni_")
        st.dataframe(df_top_score[['Symbol', 'Score', 'ML Prognoza %', 'RSI Akcja']], use_container_width=True, hide_index=True)
    
with col_popularne:
    with st.container():
        st.markdown(f"**⭐ Popularne Aktywa** | _Stała Lista wg Potencjału 30 Dni_")
        st.dataframe(df_popular[['Symbol', 'Score', 'ML Prognoza %', 'RSI Akcja']], use_container_width=True, hide_index=True)

st.markdown("---")

# --- DYNAMICZNY WYKRES PLOTLY Z PROGNOZĄ ML (1 INTERWAŁ I 30 DNI) ---

st.header("🧠 Szczegółowa Analiza 3xAI")
selected_symbol_short = st.selectbox("Wybierz Aktywo do Analizy Wykresu i Wniosków:", options=chart_symbols)

if selected_symbol_short:
    full_symbol = f"{selected_symbol_short}USDT"
    result = analysis_results.get(full_symbol)
    df_row = df_full_analysis[df_full_analysis['Symbol'] == selected_symbol_short].iloc[0]
    
    df_data = result['data'] 
    prog_price_1step = result['forecast_1step_price'] # Cena dla gwiazdki (następny interwał)
    prog_price_30day = result['forecast_ml_price_30day'] # Cena dla kropki (30 dni)
    prog_time_30day = result['forecast_monthly_timestamp'] # Czas dla kropki (30 dni)
    
    # --- WSKAŹNIKI KLUCZOWE NA GÓRZE ---
    st.subheader(f"Wnioski dla {selected_symbol_short} (SCORE: {result['score']})")
    
    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)

    with col_metric1:
        st.metric(
            label="📈 Prognoza ML (30 Dni) %",
            value=df_row['ML Prognoza %'],
            delta=df_row['ML Cena Prog.'],
            delta_color="normal" if result['forecast_ml_percent'] >= 0 else "inverse"
        )
    with col_metric2:
        st.metric(
            label="❤️ Sentyment (30 Dni) %",
            value=df_row['Sentyment %'],
            delta=result['forecast_sentiment_text'].split('|')[0].strip().split(':')[1].strip() if '|' in result['forecast_sentiment_text'] else 'Brak',
            delta_color="normal" if result['forecast_sentiment_percent'] >= 0 else "inverse"
        )
    with col_metric3:
        st.metric(
            label="🟢 RSI Akcja",
            value=result['analysis_rsi']['action'],
            delta=result['analysis_rsi']['status'],
            delta_color="normal" if 'KUPNO' in result['analysis_rsi']['action'] else "inverse"
        )
    with col_metric4:
        st.metric(
            label="Sugestia Algorytmu",
            value=result['sugestion'],
            delta=f"RSI: {result['data']['RSI'].iloc[-1]:.2f}",
            delta_color="off"
        )
    
    st.markdown("---")

    # --- GENEROWANIE WYKRESU PLOTLY ---
    st.markdown("### Wykres Cenowy z Prognozą ML (1 Interwał i 30 Dni)")

    fig = go.Figure(data=[
        go.Candlestick(
            x=df_data.index,
            open=df_data['Open'],
            high=df_data['High'],
            low=df_data['Low'],
            close=df_data['Close'],
            name='Cena'
        ),
        go.Scatter(
            x=df_data.index, 
            y=df_data['SMA_20'], 
            mode='lines', 
            name='SMA 20', 
            line=dict(color='orange')
        )
    ])
    
    # DODANIE PUNKTU PROGNOZY ML (Gwiazdka - Następny Interwał)
    if prog_price_1step is not None:
        last_time = df_data.index[-1]
        fig.add_trace(go.Scatter(
            x=[last_time], 
            y=[prog_price_1step], 
            mode='markers+text', 
            name='Prognoza Nast. Interwał', 
            marker=dict(size=15, color='red', symbol='star'),
            text=['Prognoza (Następny Interwał)'],
            textposition="top center"
        ))
        
    # DODANIE PUNKTU PROGNOZY MIESIĘCZNEJ (Duża Kropka i Linia)
    if prog_price_30day is not None and prog_time_30day is not None:
        
        last_close_time = df_data.index[-1]
        last_close_price = df_data['Close'].iloc[-1]
        
        # Punkt na osi X w przyszłości
        fig.add_trace(go.Scatter(
            x=[prog_time_30day], 
            y=[prog_price_30day], 
            mode='markers', 
            name='Prognoza 30 Dni', 
            marker=dict(size=15, color='green', symbol='circle'),
            text=[f"Prognoza 30 dni: ${prog_price_30day:,.2f}"],
            hoverinfo='text'
        ))
        
        # Linia łącząca ostatnią cenę zamknięcia z prognozą miesięczną
        fig.add_trace(go.Scatter(
            x=[last_close_time, prog_time_30day],
            y=[last_close_price, prog_price_30day],
            mode='lines',
            line=dict(dash='dash', color='green', width=1),
            name='Linia Trendu (30 dni)'
        ))

    fig.update_layout(
        height=550, 
        xaxis_rangeslider_visible=False,
        title=f"{selected_symbol_short} - Wykres świecowy ({interval}) z Ekstrapolacją ML",
        yaxis_title="Cena (USDT)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Wnioski Analityczne (Surowe Dane)")
    
    # 3. Wnioski w 3 kolumnach na dole
    col_ml_txt, col_sentiment_txt, col_rsi_txt = st.columns(3)
    
    with col_ml_txt:
        st.markdown("**Scikit-learn (Prognoza 30 Dni)** 🧠")
        st.text_area(label="Prognoza ML:", value=result['forecast_ml_text'], height=150, label_visibility="collapsed", key=f"ml_{full_symbol}_final")

    with col_sentiment_txt:
        st.markdown("**Sentyment Społecznościowy (VADER Sim.)** 💬")
        st.text_area(label="Analiza Sentymentu:", value=result['forecast_sentiment_text'], height=150, label_visibility="collapsed", key=f"sentiment_{full_symbol}_final")
    
    with col_rsi_txt:
        st.markdown("**Wniosek Techniczny (Czyste RSI)** 📊")
        st.text_area(label="Wnioski RSI:", value=result['analysis_rsi']['action'] + "\n" + result['analysis_rsi']['status'], height=150, label_visibility="collapsed", key=f"rsi_{full_symbol}_final")


# --- LOGIKA AUTOMATYCZNEGO ODŚWIEŻANIA ---

st.markdown("---")
st.markdown(f"_Ostatnia aktualizacja: **{time.strftime('%H:%M:%S', time.localtime())}**_")
st.info("System automatycznie odświeży dane i ponowi skanowanie za około 60 sekund. Zapewnia to zawsze świeże prognozy.")

