import os
import pandas as pd
import subprocess
import time
from datetime import datetime

# Stała liczba symulacji
NUM_SIMULATIONS = 10
# Zmieniono nazwy folderów i plików na wersję dla KRYPTOWALUT
RESULTS_DIR = "top_results_crypto"
AVG_FILE = f"{RESULTS_DIR}/average_top_crypto.csv"

def run_and_aggregate_simulations(num_simulations=NUM_SIMULATIONS):
    
    dfs = []
    
    # Tworzymy folder, jeśli nie istnieje
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print(f"--- ROZPOCZĘTO WIELOKROTNĄ SYMULACJĘ KRYPTOWALUT ({num_simulations} cykli) ---")

    for sim in range(1, num_simulations + 1):
        print(f"=== Symulacja {sim}/{num_simulations} ===")
        
        # 1. Uruchamiamy train_model_crypto.py
        env = os.environ.copy()
        env["SIMULATION_NUMBER"] = str(sim)
        
        try:
             # ⚠️ ZMIENIONO: Wywołujemy nowy plik train_model_crypto.py
             subprocess.run(["python", "train_model_crypto.py"], check=True, env=env)
        except subprocess.CalledProcessError as e:
             print(f"BŁĄD: train_model_crypto.py zakończył się niepowodzeniem w symulacji {sim}. Szczegóły: {e}")
             continue
        
        # 2. Wczytujemy zapisany plik CSV
        top_file = f"{RESULTS_DIR}/last_top_crypto_{sim}.csv"
        try:
            df = pd.read_csv(top_file)
            df['simulation_id'] = sim
            dfs.append(df)
        except FileNotFoundError:
            print(f"Ostrzeżenie: Plik {top_file} nie został znaleziony (Symulacja {sim}).")
        
        # ⚠️ OBOWIĄZKOWA PAUZA 60 SEKUND
        if sim < num_simulations:
            print(f"Pauza 60 sekund przed kolejną symulacją...")
            time.sleep(60)

    # 3. Łączymy i agregujemy wyniki
    if not dfs:
        print("Brak wyników do agregacji. Anulowanie zapisu średnich.")
        return

    all_runs = pd.concat(dfs)
    avg_df = all_runs.groupby("ticker")["pred_%"].mean().reset_index()
    avg_df = avg_df.sort_values("pred_%", ascending=False)
    
    # 4. Zapisujemy do pliku
    avg_df.to_csv(AVG_FILE, index=False)
    
    print(f"Zapisano plik średnich: {AVG_FILE}")
    print(f"Najlepsze aktywa (średnia prognoza z {len(dfs)} symulacji):")
    print(avg_df.head(10))

if __name__ == "__main__":
    run_and_aggregate_simulations()
