import streamlit as st
import pandas as pd
import os
import time
from datetime import datetime

# --- KONFIGURACJA KRYPTOWALUT ---
AVG_FILE = "top_results_crypto/average_top_crypto.csv"
REFRESH_INTERVAL_SECONDS = 3600 # Odświeżanie dashboardu co 1 godzinę
NUM_SIMULATIONS = 10 

# --- WIZUALIZACJA ---
st.set_page_config(layout="wide", page_title="AI Crypto Advisor - Aggregated Results")
st.title("₿ Agregacja Prognoz AI (Top Kryptowaluty)")
st.markdown("Dashboard wczytuje uśrednione wyniki z pliku **average_top_crypto.csv**, generowanego cyklicznie co 6 godzin przez Zadanie Cron.")
st.markdown("---")

try:
    # Wczytujemy uśrednione wyniki
    avg_df = pd.read_csv(AVG_FILE)
    
    # Formatowanie kolumny procentowej
    avg_df['Średnia Prognoza 21 Dni'] = avg_df['pred_%'].apply(lambda x: f"{x:.2f}%")
    avg_df = avg_df.rename(columns={'ticker': 'Kryptowaluta'})
    avg_df = avg_df[['Kryptowaluta', 'Średnia Prognoza 21 Dni']]
    
    st.subheader(f"🚀 Top {len(avg_df)} Aktywów (Średnia z {NUM_SIMULATIONS} Symulacji)")
    st.markdown("Wysoki procent oznacza większy oczekiwany zwrot w ciągu najbliższych 21 dni. Pamiętaj, że dane opierają się na historycznych cenach i newsach.")
    
    # Wyświetlenie tabeli
    st.dataframe(avg_df, use_container_width=True, hide_index=True)
    
    # Dodanie informacji o ostatniej aktualizacji
    if os.path.exists(AVG_FILE):
        timestamp = os.path.getmtime(AVG_FILE)
        dt_object = datetime.fromtimestamp(timestamp)
        st.caption(f"Ostatnia aktualizacja danych (plik average_top_crypto.csv): {dt_object.strftime('%Y-%m-%d %H:%M:%S')}")
        
except FileNotFoundError:
    st.error(f"Plik wyników {AVG_FILE} nie został jeszcze wygenerowany przez Zadanie Cron. Sprawdź, czy run_multiple_simulations_crypto.py działa poprawnie.")
except Exception as e:
    st.error(f"Wystąpił błąd podczas wczytywania danych: {e}")

# Automatyczne odświeżanie strony Streamlit
time.sleep(REFRESH_INTERVAL_SECONDS)
st.rerun()
