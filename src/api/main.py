from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import date, timedelta, datetime # <--- 'datetime' eklendi
import pandas as pd
import pandas_ta as ta
import ccxt
import os
import traceback
from fastapi.middleware.cors import CORSMiddleware
import sys
import json # <--- 'json' eklendi

# --- Projenin ana dizinini sistem yoluna ekleyerek diğer modülleri import edilebilir hale getiriyoruz ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.append(project_root)

from src.models.model_utils import load_all_models

# --- YENİ: Arena Modüllerini Import Etme ---
ARENA_SIMULATOR_DIR = os.path.join(project_root, 'arena_simulator')
sys.path.append(ARENA_SIMULATOR_DIR) # Arena klasörünü path'e ekle
ARENA_DATA_OUTPUT_DIR = ARENA_SIMULATOR_DIR # JSON dosyalarının okunacağı yer

try:
    # arena_simulator paketinden gerekli şeyleri import et
    from run_arena_backtest import (
        execute_backtest,
        DEFAULT_THRESHOLD, 
        DEFAULT_CAPITAL    
    )
    print("Arena backtest modülü başarıyla import edildi.")
except ImportError as e:
    print(f"HATA: run_arena_backtest import edilemedi. __init__.py var mı? Hata: {e}")
    # Hata durumunda API'nin çökmemesi için varsayılan değerler ve sahte fonksiyon
    def execute_backtest(*args, **kwargs):
        raise HTTPException(status_code=500, detail="Arena backtest modülü yüklenemedi.")
    DEFAULT_THRESHOLD = 1.5 
    DEFAULT_CAPITAL = 10000.0
except Exception as e: # Diğer olası import hataları için
     print(f"Beklenmedik import hatası: {e}")
     def execute_backtest(*args, **kwargs):
        raise HTTPException(status_code=500, detail=f"Arena import hatası: {e}")
     DEFAULT_THRESHOLD = 1.5; DEFAULT_CAPITAL = 10000.0
# --- YENİ IMPORT BÖLÜMÜ BİTTİ ---


# --- UYGULAMA BAŞLANGICI ---
app = FastAPI(title="Kripto Tahmin API")

# --- CORS Ayarları ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS = load_all_models()

# --- STANDART ÖZELLİK LİSTESİ (Eğitim script'i ile %100 aynı) ---
MODEL_FEATURES = [
    'open', 'high', 'low', 'close', 'volume', 'fear_greed_value', 'RSI_14',
    'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'BBL_20_2.0',
    'BBM_20_2.0', 'BBU_20_2.0', 'BBP_20_2.0',
    'daily_return', 'MA7', 'MA30', 'reddit_sentiment'
]

# --- Pydantic Modelleri ---
class PastPredictionRequest(BaseModel):
    coin_pair: str = "BTC/USDT"
    target_date: date = Field(..., description="YYYY-MM-DD formatında geçmiş bir tarih")

# --- YENİ: Arena Backtest İsteği Modeli ---
class BacktestParams(BaseModel):
    start_date_str: str
    rule: str
    risk_ratio: float
# --- YENİ MODEL BİTTİ ---

# --- YARDIMCI FONKSİYONLAR ---
# (Senin process_features fonksiyonun burada - Değişiklik Yok)
def process_features(df: pd.DataFrame) -> pd.DataFrame:
    df_merged = df.copy()

    fg_path = os.path.join(project_root, 'data', 'raw', 'fear_greed_index.csv')
    if os.path.exists(fg_path):
        df_fg = pd.read_csv(fg_path, index_col='date', parse_dates=True)[['fear_greed_value']]
        df_merged = df_merged.join(df_fg, how='left')
        df_merged['fear_greed_value'] = df_merged['fear_greed_value'].ffill().bfill()
    else:
        df_merged['fear_greed_value'] = 50 

    df_merged['daily_return'] = df_merged['close'].pct_change()
    df_merged['MA7'] = df_merged.ta.sma(length=7, append=False)
    df_merged['MA30'] = df_merged.ta.sma(length=30, append=False)
    
    df_merged = df_merged.join(df_merged.ta.rsi(length=14))
    df_merged = df_merged.join(df_merged.ta.macd(fast=12, slow=26, signal=9))
    df_merged = df_merged.join(df_merged.ta.bbands(length=20, std=2))
    
    # API'nin yanlış ürettiği sütun adlarını, modelin beklediği doğru adlara çevir
    rename_map = {
        'BBL_20_2.0_2.0': 'BBL_20_2.0', 'BBM_20_2.0_2.0': 'BBM_20_2.0',
        'BBU_20_2.0_2.0': 'BBU_20_2.0', 'BBP_20_2.0_2.0': 'BBP_20_2.0'
    }
    df_merged.rename(columns=rename_map, inplace=True)

    df_merged['reddit_sentiment'] = 0
    df_merged.dropna(inplace=True)
    
    return df_merged

# --- API ENDPOINTS ---
@app.get("/")
def read_root():
    return {"message": f"Kripto Tahmin API'sine hoş geldiniz! {len(MODELS)} adet model yüklendi: {list(MODELS.keys())}"}

# (Senin make_prediction fonksiyonun burada - Değişiklik Yok)
def make_prediction(model, features_df):
    """Ortak tahmin mantığını yürüten fonksiyon"""
    # Sadece modelin bildiği sütunları, doğru sırada seçerek tüm uyumsuzlukları giderir.
    final_features = features_df.reindex(columns=MODEL_FEATURES).fillna(0)

    prediction = model.predict(final_features)
    prediction_proba = model.predict_proba(final_features)
    result = "Yükselir" if prediction[0] == 1 else "Düşer"
    confidence = float(prediction_proba[0][prediction[0]])
    
    return result, confidence, final_features

# (Senin /predict endpoint'in burada - Değişiklik Yok)
@app.post("/predict")
def predict_future(coin_pair: str = "BTC/USDT"):
    try:
        coin_symbol = coin_pair.split('/')[0].upper()
        if coin_symbol not in MODELS:
            raise HTTPException(status_code=404, detail=f"{coin_symbol} için eğitilmiş bir model bulunamadı.")
        
        exchange = ccxt.binance()
        bars = exchange.fetch_ohlcv(coin_pair, timeframe='1d', limit=100)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('date', inplace=True)
        
        df_processed = process_features(df.drop(columns=['timestamp']))
        
        if df_processed.empty:
            raise ValueError("Özellikler hesaplandıktan sonra veri kalmadı.")
        
        result, confidence, final_features = make_prediction(MODELS[coin_symbol], df_processed.iloc[[-1]])

        return { 
            "coin_pair": coin_pair, 
            "prediction_for": (final_features.index[0] + timedelta(days=1)).strftime('%Y-%m-%d'), 
            "prediction": result, 
            "confidence": f"{confidence:.2%}",
            "processed_features": final_features.to_dict(orient='records')[0] 
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Gelecek tahmini sırasında bir hata oluştu: {e}")

# (Senin /predict_past endpoint'in burada - Değişiklik Yok)
@app.post("/predict_past")
def predict_past(request: PastPredictionRequest):
    try:
        coin_symbol = request.coin_pair.split('/')[0].upper()
        if coin_symbol not in MODELS:
            raise HTTPException(status_code=404, detail=f"{coin_symbol} için eğitilmiş bir model bulunamadı.")

        target_date = pd.to_datetime(request.target_date)

        exchange = ccxt.binance()
        since = exchange.parse8601((target_date - timedelta(days=100)).isoformat())
        bars = exchange.fetch_ohlcv(request.coin_pair, timeframe='1d', since=since, limit=102)
        
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('date', inplace=True)
        df.index = df.index.normalize()

        if target_date not in df.index or (target_date + timedelta(days=1)) not in df.index:
            raise HTTPException(status_code=404, detail="Belirtilen tarih veya sonraki gün için veri bulunamadı.")
        
        actual_close_today = df.loc[target_date]['close']
        actual_close_tomorrow = df.loc[target_date + timedelta(days=1)]['close']
        actual_result_numeric = 1 if actual_close_tomorrow > actual_close_today * 1.005 else 0
        actual_result_text = "Yükseldi" if actual_result_numeric == 1 else "Düştü"

        df_for_features = df.loc[:target_date].drop(columns=['timestamp'])
        df_processed = process_features(df_for_features.copy())
        
        if df_processed.empty:
            raise ValueError("Özellikler hesaplandıktan sonra veri kalmadı.")
        
        result, confidence, final_features = make_prediction(MODELS[coin_symbol], df_processed.iloc[[-1]])

        return { 
            "coin_pair": request.coin_pair, 
            "prediction_for_date": target_date.strftime('%Y-%m-%d'), 
            "model_prediction": result, 
            "confidence": f"{confidence:.2%}",
            "actual_result": actual_result_text, 
            "was_correct": bool((1 if result == 'Yükselir' else 0) == actual_result_numeric),
            "processed_features": final_features.to_dict(orient='records')[0] 
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Geçmiş tahmini sırasında bir hata oluştu: {e}")

# --- YENİ ARENA ENDPOINTLERİ ---

@app.get("/arena/results")
async def get_arena_results():
    """Önceden çalıştırılmış varsayılan backtest sonuçlarını JSON dosyalarından okur."""
    print("GET /arena/results isteği alındı.")
    results = {}
    files_to_read = {"summary": "summary.json", "equity_curve": "equity_curve.json", "trade_log": "trade_log.json"}
    for key, filename in files_to_read.items():
        path = os.path.join(ARENA_DATA_OUTPUT_DIR, filename)
        try:
            with open(path, 'r', encoding='utf-8') as f: results[key] = json.load(f)
            print(f" - {filename} başarıyla okundu.")
        except FileNotFoundError:
            print(f" HATA: {filename} bulunamadı."); results[key] = {"error": f"{filename} not found."}
        except Exception as e:
            print(f" HATA: {filename} okunurken hata: {e}"); results[key] = {"error": f"Error reading {filename}: {e}"}
    return results

@app.post("/arena/run_backtest")
async def run_custom_backtest(params: BacktestParams):
    """Frontend'den gelen parametrelerle dinamik olarak backtest çalıştırır."""
    print(f"POST /arena/run_backtest isteği alındı: {params}")
    try:
        end_date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        results = execute_backtest( # Import edilen fonksiyonu çağır
            start_date_str=params.start_date_str,
            end_date_str=end_date_str,
            rule=params.rule,
            risk_ratio=params.risk_ratio,
            threshold=DEFAULT_THRESHOLD, # En başta import edilen sabiti kullan
            capital=DEFAULT_CAPITAL      # En başta import edilen sabiti kullan
        )
        print("Özel backtest başarıyla çalıştırıldı, sonuçlar döndürülüyor.")
        return results
    except FileNotFoundError as fnf: print(f"HATA: Backtest dosya bulunamadı: {fnf}"); raise HTTPException(status_code=404, detail=f"Gerekli dosya bulunamadı: {fnf}")
    except ValueError as ve: print(f"HATA: Backtest geçersiz değer: {ve}"); raise HTTPException(status_code=400, detail=f"Geçersiz parametre/veri: {ve}")
    except HTTPException as http_exc: raise http_exc
    except Exception as e:
        print(f"KRİTİK HATA: Özel backtest hatası: {e}"); 
        traceback.print_exc(); # Hata detayını logla
        raise HTTPException(status_code=500, detail=f"Backtest sunucu hatası: {e}")

# --- API Çalıştırma Komutu (Terminalden Çalıştırılacak) ---
# D:\crypto-predictor\src içindeyken:
# uvicorn api.main:app --reload --host 127.0.0.1 --port 8000