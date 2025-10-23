# Używamy oficjalnego obrazu Pythona
FROM python:3.10-slim

# Ustawienie katalogu roboczego
WORKDIR /app

# Kopiowanie plików wymagań i instalacja zależności
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pobranie danych NLTK do analizy sentymentu (VADER)
RUN python -m nltk.downloader vader_lexicon

# Kopiowanie wszystkich plików projektu
COPY . .

# Definicja polecenia uruchamiającego serwer Uvicorn
# Ustawienie liczby workerów na 1 dla darmowych tierów
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
