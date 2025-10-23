# Plik: api_server.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
from crypto_analyzer import scan_and_return_data_for_api 
import uvicorn

# Inicjalizacja aplikacji FastAPI
app = FastAPI(
    title="Crypto AI Advisor API",
    description="Serwer analityczny dla aplikacji Android."
)

# Definicja modelu danych wejściowych (to, co aplikacja Android wyśle)
class ScanRequest(BaseModel):
    limit_symbols: int = Field(default=200, ge=20, le=200, description="Głębokość wstępnego skanowania.")
    top_n: int = Field(default=10, ge=1, le=15, description="Liczba aktywów z najlepszym SCORE do szczegółowej analizy.")
    interval: str = Field(default='4h', pattern='^(1h|4h|1d)$', description="Interwał czasowy do analizy (1h, 4h, 1d).")

# Endpoint API
@app.post("/scan-and-advice", response_model=Dict[str, Any])
def get_ai_scan(request: ScanRequest):
    """
    Uruchamia pełny skaner 3xAI (ML, Sentyment, RSI) i zwraca szczegółowe dane dla wybranych aktywów.
    """
    try:
        # Wywołanie głównej funkcji analitycznej
        results = scan_and_return_data_for_api(
            limit_symbols=request.limit_symbols,
            top_n=request.top_n,
            interval=request.interval
        )
        return results
    
    except Exception as e:
        print(f"BŁĄD KRYTYCZNY SERWERA: {e}")
        raise HTTPException(status_code=500, detail=f"Wystąpił błąd serwera podczas analizy: {e}")

# Endpoint testowy
@app.get("/")
def read_root():
    return {"message": "Crypto AI Advisor API jest aktywne. Użyj POST /scan-and-advice."}

if __name__ == '__main__':
    # ⚠️ Użycie host="0.0.0.0" jest KLUCZOWE, aby serwer był dostępny w sieci lokalnej (przez telefon)
    print("🚀 Uruchamiam serwer API na http://0.0.0.0:8000")
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000)
