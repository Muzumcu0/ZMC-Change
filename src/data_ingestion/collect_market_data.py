import ccxt
import pandas as pd
from datetime import datetime, timedelta, UTC
import os
import time

def fetch_ohlcv(symbol='BTC/USDT', timeframe='1d', years=3):
    """
    Belirtilen sembol için geçmiş OHLCV verilerini çeker.

    Args:
        symbol (str): Çekilecek kripto para çifti (örn: 'BTC/USDT').
        timeframe (str): Zaman aralığı (örn: '1d', '4h', '1h').
        years (int): Kaç yıllık veri çekileceği.

    Returns:
        pandas.DataFrame: OHLCV verilerini içeren DataFrame.
    """
    exchange = ccxt.binance()

    start_date = datetime.now(UTC) - timedelta(days=years*365)
    since = exchange.parse8601(start_date.isoformat())
    
    all_ohlcv = []
    
    print(f"-> {symbol} için son {years} yıllık veri çekiliyor...")

    while since < exchange.milliseconds():
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since)
            if len(ohlcv):
                since = ohlcv[-1][0] + exchange.parse_timeframe(timeframe) * 1000 
                all_ohlcv.extend(ohlcv)
                print(f"   {len(all_ohlcv)} adet mum verisi çekildi...")
            else:
                break
        except Exception as e:
            print(f"   Bir hata oluştu: {e}")
            break
            
    print(f"-> Veri çekme işlemi tamamlandı. {len(all_ohlcv)} satır veri bulundu.")

    if not all_ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    return df

if __name__ == "__main__":
    # --- AYARLAR ---
    # Artık tek bir coin yerine, verisini çekmek istediğimiz coin'lerin bir listesini tanımlıyoruz.
    # Buraya istediğin kadar coin ekleyebilirsin.
    COIN_PAIRS = [
        'BTC/USDT', 
        'ETH/USDT', 
        'DOGE/USDT', 
        'LTC/USDT',
        'XRP/USDT', # Eklendi
        'SOL/USDT', # Eklendi
        'ADA/USDT'  # Eklendi
    ] 
    TIMEFRAME = '1d'
    YEARS_OF_DATA = 3

    # Veriyi kaydetmek için klasör yolunu oluştur
    output_folder = os.path.join('data', 'raw')
    os.makedirs(output_folder, exist_ok=True)

    # Listedeki her bir coin için döngü başlat
    for coin_pair in COIN_PAIRS:
        print("-" * 50)
        print(f"İşlem başlıyor: {coin_pair}")
        
        # Dosya adını her coin için dinamik olarak oluştur
        filename = f"{coin_pair.replace('/', '_')}_{TIMEFRAME}_market_data.csv"
        output_path = os.path.join(output_folder, filename)

        # Veriyi çek
        market_data_df = fetch_ohlcv(symbol=coin_pair, timeframe=TIMEFRAME, years=YEARS_OF_DATA)

        if not market_data_df.empty:
            # Veriyi ilgili CSV dosyasına kaydet
            market_data_df.to_csv(output_path)
            print(f"BAŞARILI: Veri şu dosyaya kaydedildi -> {output_path}")
        else:
            print(f"BAŞARISIZ: {coin_pair} için veri çekilemedi veya veri boş geldi.")
        
        # Binance API'sine çok sık istek atmamak için istekler arasına küçük bir bekleme ekliyoruz.
        time.sleep(1) 

    print("-" * 50)
    print("Tüm coinler için veri çekme işlemi başarıyla tamamlandı.")



