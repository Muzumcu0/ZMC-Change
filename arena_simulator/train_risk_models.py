import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.ensemble import RandomForestRegressor
from joblib import dump
import os
import sys
import traceback # Hata ayıklama için

# --- 1. Proje Kök Dizinini Ayarlama ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# --- 2. Konfigürasyon ve Dosya Yolları ---
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, 'models', 'arena_models')
FEAR_GREED_PATH = os.path.join(RAW_DATA_DIR, 'fear_greed_index.csv') # F&G dosyası ortak

# --- İşlenecek Coin Listesi ---
COINS = ['BTC', 'ETH', 'ADA', 'SOL', 'LTC', 'XRP', 'DOGE']

# Risk modeli ne kadar ileriyi tahmin edecek (Günlük veri)
LOOK_FORWARD_PERIOD = 24

# TP/SL Modelleri için Kullanılacak Özellikler (reddit_sentiment HARİÇ)
RISK_MODEL_FEATURES = [
   'open', 'high', 'low', 'close', 'volume', 'fear_greed_value', 'RSI_14',
   'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'BBL_20_2.0',
   'BBM_20_2.0', 'BBU_20_2.0', 'BBP_20_2.0',
   'daily_return', 'MA7', 'MA30'
]

# --- 3. Veri Yükleme ve Hazırlama Fonksiyonu (Tek Coin İçin - Return Düzeltildi) ---
def load_prepare_single_coin_data_for_training(coin):
    """Belirtilen TEK coin için veriyi yükler, F&G ile birleştirir, özellikleri hesaplar."""
    print(f"\n--- {coin} için Veri Hazırlanıyor ---")
    market_data_path = os.path.join(RAW_DATA_DIR, f'{coin.upper()}_USDT_1d_market_data.csv')

    print(f"Market verisi yükleniyor: {market_data_path}")
    try:
        df_market = pd.read_csv(market_data_path, parse_dates=['timestamp'], index_col='timestamp')
        df_market.sort_index(inplace=True)
        df_market.columns = [col.lower() for col in df_market.columns]
        if not all(col in df_market.columns for col in ['open', 'high', 'low', 'close', 'volume']):
             raise ValueError("Market verisinde temel OHLCV sütunları eksik.")
        print(f"Market verisi yüklendi: {len(df_market)} satır.")
    except Exception as e:
        print(f"HATA: {coin} market verisi okunamadı: {e}")
        return None

    print(f"{coin}: Teknik göstergeler hesaplanıyor...")
    try:
        cols_to_drop = [f for f in RISK_MODEL_FEATURES if f not in ['open','high','low','close','volume','fear_greed_value']]
        df_market.drop(columns=cols_to_drop, errors='ignore', inplace=True)
        indicators = pd.DataFrame(index=df_market.index)
        indicators['daily_return'] = df_market['close'].pct_change()
        indicators['MA7'] = ta.sma(df_market['close'], length=7)
        indicators['MA30'] = ta.sma(df_market['close'], length=30)
        indicators['RSI_14'] = ta.rsi(df_market['close'], length=14)
        macd = ta.macd(df_market['close'], fast=12, slow=26, signal=9)
        bbands = ta.bbands(df_market['close'], length=20, std=2.0)
        bbands_rename_map = {col: f'BBL_20_2.0' for col in bbands.columns if col.startswith("BBL")}
        bbands_rename_map.update({col: f'BBM_20_2.0' for col in bbands.columns if col.startswith("BBM")})
        bbands_rename_map.update({col: f'BBU_20_2.0' for col in bbands.columns if col.startswith("BBU")})
        bbands_rename_map.update({col: f'BBP_20_2.0' for col in bbands.columns if col.startswith("BBP")})
        bbands.rename(columns=bbands_rename_map, inplace=True)
        bb_cols_needed = ['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBP_20_2.0']
        bb_cols_available = [col for col in bb_cols_needed if col in bbands.columns]
        df_with_indicators = pd.concat([df_market, indicators, macd, bbands[bb_cols_available]], axis=1)
        print(f"{coin}: Teknik göstergeler hesaplandı.")
    except Exception as e:
        print(f"HATA: {coin} teknik göstergeler hesaplanırken hata oluştu: {e}")
        return None

    print(f"{coin}: Fear & Greed Index verisi yükleniyor...")
    try:
        df_fg = pd.read_csv(FEAR_GREED_PATH, parse_dates=['date'], index_col='date')
        df_fg.sort_index(inplace=True)
        df_fg.index.name = 'timestamp'
        if 'fear_greed_value' not in df_fg.columns: raise KeyError("'fear_greed_value'")
        df_fg = df_fg[['fear_greed_value']]
    except Exception as e:
        print(f"Uyarı: Fear & Greed okunamadı ({e}). Varsayılan değer (50) kullanılacak.")
        df_fg = pd.DataFrame(index=df_with_indicators.index, columns=['fear_greed_value'])
        df_fg['fear_greed_value'] = 50

    print(f"{coin}: Veriler birleştiriliyor...")
    df_final = df_with_indicators.join(df_fg, how='left')
    df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f"{coin}: NaN değerler dolduruluyor...")
    df_final = df_final.ffill()
    df_final.dropna(inplace=True)
    print(f"{coin}: Ön işleme tamamlandı. Kullanılabilir veri: {len(df_final)} satır.")
        
    # --- DÜZELTİLMİŞ RETURN BLOĞU ---
    # Etiketleme ('high', 'low', 'close') ve eğitim ('RISK_MODEL_FEATURES') için gereken tüm sütunları alalım.
    required_cols_for_labeling = ['high', 'low', 'close']
    all_needed_cols = list(set(RISK_MODEL_FEATURES + required_cols_for_labeling)) # set() mükerrerleri kaldırır

    missing_needed = [col for col in all_needed_cols if col not in df_final.columns]
    if missing_needed:
        print(f"HATA: {coin} için son veride etiketleme/eğitim için gerekli sütunlar eksik: {missing_needed}")
        return None

    print(f"{coin}: Etiketleme ve eğitim için {len(all_needed_cols)} sütun seçildi.")
    return df_final[all_needed_cols].copy()
    # --- DÜZELTİLMİŞ RETURN BLOĞU BİTTİ ---


# --- 4. Etiketleme Fonksiyonu (Değişiklik Yok) ---
def create_regression_labels(df, period=LOOK_FORWARD_PERIOD):
    # ... (İçerik aynı) ...
    print(f"Regresyon etiketleri oluşturuluyor (Sonraki {period} gün)...")
    epsilon = 1e-10
    rolling_high = df['high'].rolling(window=period).max().shift(-period)
    rolling_low = df['low'].rolling(window=period).min().shift(-period)
    df['target_high_pct'] = (rolling_high - df['close']) / (df['close'] + epsilon)
    df['target_low_pct'] = (rolling_low - df['close']) / (df['close'] + epsilon)
    df = df.dropna(subset=['target_high_pct', 'target_low_pct'])
    print("Etiketleme tamamlandı.")
    return df

# --- 5. Model Eğitimi Fonksiyonu (Tek Coin İçin - Değişiklik Yok) ---
def train_single_coin_models(df, coin):
    # ... (İçerik aynı) ...
    features = RISK_MODEL_FEATURES
    print(f"\n--- {coin} için Modeller Eğitiliyor ---")
    print(f"Kullanılacak özellikler: {features}")
    df_train = df.dropna(subset=features + ['target_high_pct', 'target_low_pct'])
    if df_train.empty:
         print(f"HATA: {coin} için NaN temizleme sonrası model eğitimi verisi boş kaldı.")
         return False
    print(f"Eğitim için kullanılacak satır sayısı: {len(df_train)}")
    try:
        print(f"{coin}: Kâr Al (TP) modeli eğitiliyor...")
        X_tp = df_train[features]
        y_tp = df_train['target_high_pct']
        tp_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, max_features='sqrt')
        tp_model.fit(X_tp, y_tp)
        print(f"{coin}: TP Modeli eğitildi.")
        print(f"{coin}: Zarar Durdur (SL) modeli eğitiliyor...")
        X_sl = df_train[features]
        y_sl = df_train['target_low_pct']
        sl_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, max_features='sqrt')
        sl_model.fit(X_sl, y_sl)
        print(f"{coin}: SL Modeli eğitildi.")
        if not os.path.exists(MODEL_SAVE_DIR): os.makedirs(MODEL_SAVE_DIR)
        tp_filename = f"{coin.lower()}_tp_model.pkl"
        sl_filename = f"{coin.lower()}_sl_model.pkl"
        dump(tp_model, os.path.join(MODEL_SAVE_DIR, tp_filename))
        dump(sl_model, os.path.join(MODEL_SAVE_DIR, sl_filename))
        print(f"{coin}: Modeller başarıyla kaydedildi ({tp_filename}, {sl_filename}).")
        last_features_data = df_train[features].iloc[-1]
        last_features_df = pd.DataFrame([last_features_data])
        predicted_tp = tp_model.predict(last_features_df)[0]
        predicted_sl = sl_model.predict(last_features_df)[0]
        print(f"  Test Tahmini ({df_train.index[-1]}): TP: +{predicted_tp * 100:.2f}%, SL: {predicted_sl * 100:.2f}%")
        return True
    except Exception as e:
        print(f"HATA: {coin} için model eğitimi/kaydetme hatası: {e}")
        return False

# --- 6. Ana Çalıştırıcı (Değişiklik Yok) ---
if __name__ == "__main__":
    # ... (İçerik aynı) ...
    print("*"*50)
    print(" Coin Bazlı Risk Modeli Eğitimi Başlatıldı ")
    print("*"*50)
    successful_coins = []
    failed_coins = []
    for coin in COINS:
        try:
            df_coin_processed = load_prepare_single_coin_data_for_training(coin)
            if df_coin_processed is None or df_coin_processed.empty:
                failed_coins.append(coin)
                continue
            df_coin_labeled = create_regression_labels(df_coin_processed)
            if df_coin_labeled.empty:
                print(f"HATA: {coin} için etiketleme sonrası veri boş kaldı.")
                failed_coins.append(coin)
                continue
            success = train_single_coin_models(df_coin_labeled, coin)
            if success: successful_coins.append(coin)
            else: failed_coins.append(coin)
        except Exception as e:
            print(f"KRİTİK HATA: {coin} işlenirken beklenmedik bir hata oluştu: {e}")
            failed_coins.append(coin)
    print("\n" + "*"*50)
    print(" Eğitim Tamamlandı ")
    print("*"*50)
    print(f"Başarıyla Eğitilen Coinler ({len(successful_coins)}): {', '.join(successful_coins)}")
    if failed_coins: print(f"Başarısız Olan Coinler ({len(failed_coins)}): {', '.join(failed_coins)}")
    print(f"Modeller şuraya kaydedildi: {MODEL_SAVE_DIR}")