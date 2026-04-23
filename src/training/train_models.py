import pandas as pd
import os
import sys
import pandas_ta as ta
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib 
from sklearn.ensemble import RandomForestRegressor
import numpy as np 

# Proje ana dizinini sistem yoluna ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.append(project_root)

from src.models.model_utils import save_model

# --- Ayarlar ---
COINS_TO_TRAIN = ['BTC', 'ETH', 'DOGE', 'LTC', 'XRP', 'SOL', 'ADA']
OPTUNA_TRIALS = 25 

# --- YENİ: Arena Model Yolu ---
ARENA_MODEL_DIR = os.path.join(project_root, 'models', 'arena_models')

# --- STANDART ÖZELLİK LİSTELERİ (TÜM SİSTEMDE AYNI OLACAK) ---
MODEL_FEATURES = [
    'open', 'high', 'low', 'close', 'volume', 'fear_greed_value', 'RSI_14',
    'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'BBL_20_2.0',
    'BBM_20_2.0', 'BBU_20_2.0', 'BBP_20_2.0',
    'daily_return', 'MA7', 'MA30', 'reddit_sentiment'
]
# Arena (TP/SL) modelleri 'reddit_sentiment' özelliğini kullanmıyordu
RISK_MODEL_FEATURES = [f for f in MODEL_FEATURES if f != 'reddit_sentiment']


def train_for_coin(coin_symbol: str):
    """
    Belirtilen tek bir coin için tüm veri işleme ve 3 modelin (Signal, TP, SL) eğitim sürecini yürütür.
    --- YENİ: Başarılı olursa sinyal modelinin doğruluğunu döndürür. ---
    """
    print(f"\n{'='*50}\nİşlem Başlıyor: {coin_symbol}\n{'='*50}")
    
    try:
        # 1. Veri Yükleme
        market_path = os.path.join(project_root, 'data', 'raw', f'{coin_symbol}_USDT_1d_market_data.csv')
        df_market = pd.read_csv(market_path, index_col=0, parse_dates=True)

        fg_path = os.path.join(project_root, 'data', 'raw', 'fear_greed_index.csv')
        df_fg = pd.read_csv(fg_path, index_col='date', parse_dates=True)[['fear_greed_value']]

        # 2. Veri İşleme ve Birleştirme
        df_merged = df_market.join(df_fg, how='inner')
        df_merged['reddit_sentiment'] = 0

        # 3. Teknik Özellikler
        df_merged['daily_return'] = df_merged['close'].pct_change()
        df_merged['MA7'] = df_merged.ta.sma(length=7, append=False)
        df_merged['MA30'] = df_merged.ta.sma(length=30, append=False)
        df_merged = df_merged.join(df_merged.ta.rsi(length=14))
        df_merged = df_merged.join(df_merged.ta.macd(fast=12, slow=26, signal=9))
        df_merged = df_merged.join(df_merged.ta.bbands(length=20, std=2))
        
        rename_map = {
            'BBL_20_2.0_2.0': 'BBL_20_2.0', 'BBM_20_2.0_2.0': 'BBM_20_2.0',
            'BBU_20_2.0_2.0': 'BBU_20_2.0', 'BBP_20_2.0_2.0': 'BBP_20_2.0'
        }
        df_merged.rename(columns=rename_map, inplace=True)

        # 4. Hedef Değişkenleri (Target)
        
        # --- Sinyal Modeli Hedefi (UP/DOWN) ---
        df_merged['target_signal'] = (df_merged['close'].shift(-1) > df_merged['close'] * 1.005).astype(int)

        # --- YENİ: Arena Modelleri Hedefleri (TP/SL Yüzdeleri) ---
        next_high = df_merged['high'].shift(-1)
        next_low = df_merged['low'].shift(-1)
        df_merged['target_tp_pct'] = (next_high - df_merged['close']) / df_merged['close']
        df_merged['target_sl_pct'] = (next_low - df_merged['close']) / df_merged['close']

        df_merged.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_merged.dropna(inplace=True)

        # --- 5. Model Eğitimi (Sinyal Modeli - XGBoost) ---
        print(f"[{coin_symbol}] Sinyal Modeli (XGBoost) eğitiliyor...")
        
        X_signal = df_merged.reindex(columns=MODEL_FEATURES).fillna(0) 
        y_signal = df_merged['target_signal']
        
        split_index = int(len(X_signal) * 0.8)
        X_train, X_test = X_signal.iloc[:split_index], X_signal.iloc[split_index:]
        y_train, y_test = y_signal.iloc[:split_index], y_signal.iloc[split_index:]

        def objective(trial):
            param = { 'objective': 'binary:logistic','eval_metric': 'logloss', 'n_estimators': trial.suggest_int('n_estimators', 100, 500), 'max_depth': trial.suggest_int('max_depth', 3, 8), }
            model = XGBClassifier(**param, random_state=42)
            model.fit(X_train, y_train)
            return accuracy_score(y_test, model.predict(X_test))

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=OPTUNA_TRIALS)
        print(f"[{coin_symbol}] Sinyal Modeli - En iyi doğruluk: {study.best_value:.4f}")

        final_signal_model = XGBClassifier(**study.best_params, random_state=42)
        final_signal_model.fit(X_signal, y_signal)
        save_model(final_signal_model, model_name=f"{coin_symbol.lower()}_xgboost_v1")
        print(f"[{coin_symbol}] Sinyal Modeli kaydedildi.")

        # --- 6. YENİ: Model Eğitimi (Arena Modelleri - RandomForest) ---
        
        X_risk = df_merged.reindex(columns=RISK_MODEL_FEATURES).fillna(0)
        y_tp = df_merged['target_tp_pct']
        y_sl = df_merged['target_sl_pct']

        print(f"[{coin_symbol}] Arena TP Modeli (RandomForest) eğitiliyor...")
        tp_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=10)
        tp_model.fit(X_risk, y_tp)
        
        print(f"[{coin_symbol}] Arena SL Modeli (RandomForest) eğitiliyor...")
        sl_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=10)
        sl_model.fit(X_risk, y_sl)

        os.makedirs(ARENA_MODEL_DIR, exist_ok=True) 
        
        tp_path = os.path.join(ARENA_MODEL_DIR, f"{coin_symbol.lower()}_tp_model.pkl")
        joblib.dump(tp_model, tp_path)
        
        sl_path = os.path.join(ARENA_MODEL_DIR, f"{coin_symbol.lower()}_sl_model.pkl")
        joblib.dump(sl_model, sl_path)
        
        print(f"[{coin_symbol}] Arena TP/SL modelleri kaydedildi.")

        # --- YENİ: Fonksiyondan en iyi doğruluk değerini döndür ---
        return study.best_value

    except Exception as e:
        print(f"HATA: {coin_symbol} için model eğitilirken bir hata oluştu: {e}")
        # --- YENİ: Hata durumunda None döndür ---
        return None

if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # --- YENİ: Coin doğrulamalarını saklamak için bir sözlük ---
    coin_accuracies = {}

    for coin in COINS_TO_TRAIN:
        # --- YENİ: Fonksiyondan dönen doğruluk değerini yakala ---
        accuracy = train_for_coin(coin)
        
        # --- YENİ: Sadece başarılı olanları (None olmayanları) listeye ekle ---
        if accuracy is not None:
            coin_accuracies[coin] = accuracy

    print("\nTüm modellerin (Sinyal + Arena) eğitimi tamamlandı!")

    # --- YENİ: Toplu doğruluk sonuçlarını en sonda yazdır ---
    print("\n" + "="*50)
    print(" SİNYAL MODELİ DOĞRULUK ÖZETİ (Test Seti)")
    print("="*50)

    if not coin_accuracies:
        print("Hesaplanan bir doğruluk sonucu bulunamadı.")
    else:
        # Sonuçları doğruluk oranına göre büyükten küçüğe sırala
        sorted_accuracies = sorted(coin_accuracies.items(), key=lambda item: item[1], reverse=True)
        
        for coin, acc in sorted_accuracies:
            print(f" - {coin:5s} : {acc * 100:.2f}%")
            
    print("="*50)