import pandas as pd
from pytrends.request import TrendReq
import time
import os
from datetime import datetime, timedelta

def fetch_google_trends(keywords, years=3, retries=3, backoff_factor=10):
    """
    Belirtilen anahtar kelimeler için Google Trends verilerini çeker.
    Hata durumunda bekleyip tekrar dener.

    Args:
        keywords (list): Aranacak anahtar kelimelerin listesi.
        years (int): Kaç yıllık veri çekileceği.
        retries (int): Hata durumunda kaç kez tekrar deneneceği.
        backoff_factor (int): Denemeler arasında beklenecek saniye.

    Returns:
        pandas.DataFrame: Google Trends verilerini içeren DataFrame.
    """
    pytrends = TrendReq(hl='en-US', tz=360)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    timeframe = f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"

    print(f"-> Google Trends verisi '{timeframe}' aralığı için çekiliyor...")
    print(f"   Anahtar Kelimeler: {keywords}")

    for attempt in range(retries):
        try:
            pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo='', gprop='')
            trends_df = pytrends.interest_over_time()
            
            if trends_df.empty:
                print("UYARI: Google Trends'den veri çekilemedi. DataFrame boş.")
                return pd.DataFrame()

            if 'isPartial' in trends_df.columns:
                trends_df.drop(columns=['isPartial'], inplace=True)
                
            print(f"-> Veri çekme tamamlandı. {len(trends_df)} satır veri bulundu.")
            return trends_df

        except Exception as e:
            # Google'dan gelen 429 kodlu rate limit hatasını kontrol et
            if '429' in str(e):
                wait_time = backoff_factor * (attempt + 1)
                print(f"   HATA: Google API rate limit'e takıldı (Kod 429).")
                if attempt < retries - 1:
                    print(f"   {wait_time} saniye beklenip tekrar denenecek... ({attempt + 1}/{retries})")
                    time.sleep(wait_time)
                else:
                    print("   Maksimum deneme sayısına ulaşıldı. İşlem durduruluyor.")
                    break
            else:
                # Diğer beklenmedik hataları yakala
                print(f"Google Trends'den veri çekerken beklenmedik bir hata oluştu: {e}")
                break
                
    return pd.DataFrame()


if __name__ == "__main__":
    KEYWORDS = ['Bitcoin', 'Ethereum', 'Dogecoin', 'Litecoin', 'Ripple XRP']

    output_folder = os.path.join('data', 'raw')
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, 'google_trends_data.csv')

    # Dosyanın zaten var olup olmadığını kontrol et
    if os.path.exists(output_path):
        print(f"BİLGİ: '{output_path}' dosyası zaten mevcut. İşlem atlanıyor.")
    else:
        # Google'ın rate limit'i çok agresif olabildiği için,
        # bekleme süresini (backoff_factor) artırarak tekrar deniyoruz.
        # Artık denemeler arasında 60, 120, 180 saniye bekleyecek.
        trends_data_df = fetch_google_trends(KEYWORDS, years=3, retries=3, backoff_factor=60)

        if not trends_data_df.empty:
            trends_data_df.to_csv(output_path)
            print(f"BAŞARILI: Veri şu dosyaya kaydedildi -> {output_path}")
        else:
            print("BAŞARISIZ: Google Trends verisi çekilemedi veya kaydedilemedi.")

