import os
import pandas as pd
import subprocess
import time
from datetime import datetime

# Stała liczba symulacji (zoptymalizowana dla darmowego tieru)
NUM_SIMULATIONS = 5 
RESULTS_DIR = "top_results_crypto"
AVG_FILE = f"{RESULTS_DIR}/average_top_crypto.csv"


def git_push_results():
    """Wypycha zaktualizowany plik CSV do repozytorium GitHub, wymuszając operacje Git."""
    print("--- ROZPOCZĘTO OPERACJĘ ZAPISU GIT ---")
    
    # 1. Konfiguracja Gita
    subprocess.run(["git", "config", "--global", "user.email", "github-actions[bot]@users.noreply.github.com"], check=True)
    subprocess.run(["git", "config", "--global", "user.name", "github-actions[bot]"], check=True)
    
    # 2. Musimy pobrać najnowszy stan repozytorium (unikamy błędów non-fast-forward)
    try:
        subprocess.run(["git", "pull", "--rebase"], check=True)
        print("✅ Pomyślnie pobrano najnowsze zmiany.")
    except subprocess.CalledProcessError:
        print("Brak zmian do pobrania lub błąd 'pull'. Kontynuuję.")
    
    # 3. Dodanie plików (w tym folderu, jeśli nie jest jeszcze śledzony)
    try:
        subprocess.run(["git", "add", RESULTS_DIR], check=True)
        subprocess.run(["git", "add", AVG_FILE], check=True)
        print("✅ Dodano folder i plik do stage'a.")
    except Exception as e:
        print(f"❌ BŁĄD w 'git add': {e}")
        return
        
    # 4. Sprawdzenie, czy są faktyczne zmiany do commita
    status_output = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout

    if not status_output:
        print("ℹ️ Brak faktycznych zmian w pliku. Pomijam commit i push.")
        return
    
    # 5. Commit
    try:
        commit_message = f"🤖 [CRON] Nowe wyniki z symulacji ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
        subprocess.run(["git", "commit", "-m", commit_message], check=True) 
        print(f"✅ Commit wykonany.")
    except subprocess.CalledProcessError as e:
         print("ℹ️ Commit pominięty: brak zmian w pliku.")
         return

    # 6. Push
    try:
        subprocess.run(["git", "push"], check=True) 
        print("✅ Pomyślnie zapisano wyniki na GitHub. SYSTEM JEST AKTYWNY.")
    except Exception as e:
        print(f"❌ BŁĄD GIT PUSH: {e}")
        print("Błąd autoryzacji. MUSISZ SPRAWDZIĆ USTAWIENIA GITHUB (Read and write permissions).")


def run_and_aggregate_simulations(num_simulations=NUM_SIMULATIONS):
    
    dfs = []
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print(f"--- ROZPOCZĘTO WIELOKROTNĄ SYMULACJĘ KRYPTOWALUT ({num_simulations} cykli) ---")

    for sim in range(1, num_simulations + 1):
        print(f"=== Symulacja {sim}/{num_simulations} ===")
        
        env = os.environ.copy()
        env["SIMULATION_NUMBER"] = str(sim)
        
        try:
             # 🚨 KRYTYCZNA ZMIANA: Użycie 'python' jako pierwszego argumentu.
             # Ta konstrukcja jest najbardziej niezawodna w środowiskach CI/CD.
             subprocess.run(["python", "train_model_crypto.py"], check=True, env=env)
        except subprocess.CalledProcessError as e:
             # Zatrzymujemy działanie całego Workflow, jeśli trenowanie zawiedzie.
             print(f"BŁĄD: train_model_crypto.py zakończył się niepowodzeniem w symulacji {sim}. Szczegóły: {e}")
             print("Prawdopodobnie błąd pobierania danych z zewnętrznego API. Przerywam.")
             return # Przerywamy działanie, jeśli model nie jest trenowany

        
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
