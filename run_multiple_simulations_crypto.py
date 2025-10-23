import os
import pandas as pd
import subprocess
import time
from datetime import datetime

# Stała liczba symulacji
NUM_SIMULATIONS = 5 
RESULTS_DIR = "top_results_crypto"
AVG_FILE = f"{RESULTS_DIR}/average_top_crypto.csv"


def git_push_results():
    """Wypycha zaktualizowany plik CSV do repozytorium GitHub, wymuszając operacje Git."""
    print("--- ROZPOCZĘTO OPERACJĘ ZAPISU GIT ---")
    
    # 1. Konfiguracja Gita w środowisku GitHub Actions
    try:
        subprocess.run(["git", "config", "--global", "user.email", "github-actions[bot]@users.noreply.github.com"], check=True)
        subprocess.run(["git", "config", "--global", "user.name", "github-actions[bot]"], check=True)
        # Używamy tokenu do pulla i commitów
        subprocess.run(["git", "pull"], check=True) # Najpierw pobieramy zmiany, by uniknąć konfliktów
        print("✅ Git skonfigurowany i pobrano najnowsze zmiany.")
    except Exception as e:
        print(f"❌ BŁĄD GIT: Nie udało się skonfigurować lub pobrać zmian: {e}")
        return

    # 2. DODATKOWY DEBUG: Sprawdzenie, czy plik wynikowy istnieje
    if not os.path.exists(AVG_FILE):
        print(f"❌ BŁĄD KRYTYCZNY: Plik wynikowy {AVG_FILE} NIE ISTNIEJE po agregacji! Nie można zapisać.")
        return

    # 3. Dodanie pliku do stage'a
    try:
        # Dodanie pliku do śledzenia przez Git
        subprocess.run(["git", "add", AVG_FILE], check=True)
        print(f"✅ Dodano plik {AVG_FILE} do stage'a.")
    except Exception as e:
        print(f"❌ BŁĄD w 'git add': {e}")
        return
        
    # 4. Wymuszony Commit
    try:
        commit_message = f"🤖 [CRON] Nowe wyniki z symulacji ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
        
        # Używamy --allow-empty, aby ominąć błąd "nothing to commit"
        subprocess.run(["git", "commit", "-m", commit_message, "--allow-empty"], check=True) 
        print(f"✅ Commit wykonany.")
    except subprocess.CalledProcessError as e:
        print(f"❌ BŁĄD GIT COMMIT: {e}")
        return

    # 5. Push
    try:
        # Push zmian do gałęzi 'main'
        subprocess.run(["git", "push"], check=True) 
        print("✅ Pomyślnie zapisano wyniki na GitHub. Sprawdź repozytorium.")
    except Exception as e:
        print(f"❌ BŁĄD GIT PUSH: {e}")
        print("Prawdopodobnie błąd autoryzacji (Krok 2 - 'Read and write permissions').")


def run_and_aggregate_simulations(num_simulations=NUM_SIMULATIONS):
    
    dfs = []
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print(f"--- ROZPOCZĘTO WIELOKROTNĄ SYMULACJĘ KRYPTOWALUT ({num_simulations} cykli) ---")

    for sim in range(1, num_simulations + 1):
        print(f"=== Symulacja {sim}/{num_simulations} ===")
        
        env = os.environ.copy()
        env["SIMULATION_NUMBER"] = str(sim)
        
        try:
             # Uruchamiamy train_model_crypto.py
             subprocess.run(["python", "train_model_crypto.py"], check=True, env=env)
        except subprocess.CalledProcessError as e:
             print(f"BŁĄD: train_model_crypto.py zakończył się niepowodzeniem. {e}")
             continue
        
        top_file = f"{RESULTS_DIR}/last_top_crypto_{sim}.csv"
        try:
            df = pd.read_csv(top_file)
            df['simulation_id'] = sim
            dfs.append(df)
        except FileNotFoundError:
            print(f"Ostrzeżenie: Plik {top_file} nie został znaleziony.")
        
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
    
    # Zapisujemy do pliku
    avg_df.to_csv(AVG_FILE, index=False)
    
    print(f"Zapisano plik średnich: {AVG_FILE}")

    # 4. WYKONANIE ZAPISU NA GITHUB
    git_push_results()


if __name__ == "__main__":
    run_and_aggregate_simulations()
