import requests
import pandas as pd
import os
from datetime import datetime

def fetch_fear_greed_index(days=1095):
    """
    alternative.me API'sinden Korku ve Açgözlülük Endeksi verilerini çeker.

    Args:
        days (int): Kaç günlük veri çekileceği.

    Returns:
        pandas.DataFrame: Endeks verilerini içeren DataFrame.
    """
    print(f"-> Son {days} günlük Korku ve Açgözlülük Endeksi verisi çekiliyor...")
    try:
        # API'den veriyi çek
        response = requests.get(f"https://api.alternative.me/fng/?limit={days}&format=json")
        response.raise_for_status()  # HTTP hataları için kontrol
        data = response.json()['data']
        
        # Gelen veriyi DataFrame'e çevir
        df = pd.DataFrame(data)
        
        # Sütun adlarını daha anlaşılır ve standart hale getiriyoruz
        df.rename(columns={
            'value': 'fear_greed_value',
            'value_classification': 'fear_greed_classification',
            'timestamp': 'date'
        }, inplace=True)

        # API'den gelen 'timestamp' saniye cinsinden, bunu normal tarihe çeviriyoruz
        # .dt.date ile sadece tarih kısmını alarak saat bilgisinden kurtuluyoruz
        df['date'] = pd.to_datetime(df['date'], unit='s').dt.date
        df['date'] = pd.to_datetime(df['date']) # Diğer verilerle birleştirebilmek için tekrar datetime objesine çeviriyoruz
        
        # Sadece gerekli sütunları seçip, indeksi tarih yapıyoruz
        df = df[['date', 'fear_greed_value', 'fear_greed_classification']]
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)  # Veriyi tarihe göre sıralıyoruz
        
        print(f"-> Veri çekme tamamlandı. {len(df)} satır veri bulundu.")
        return df

    except requests.exceptions.RequestException as e:
        print(f"API isteği sırasında bir hata oluştu: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Veri işlenirken bir hata oluştu: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Veriyi kaydetmek için klasör yolunu oluştur
    output_folder = os.path.join('data', 'raw')
    os.makedirs(output_folder, exist_ok=True)
    
    output_path = os.path.join(output_folder, 'fear_greed_index.csv')
    
    # Piyasa verileriyle aynı zaman aralığını kapsamak için 3 yıllık veri çekiyoruz
    days_of_data = 3 * 365 
    
    index_df = fetch_fear_greed_index(days=days_of_data)
    
    if not index_df.empty:
        # Çekilen veriyi CSV dosyasına kaydet
        index_df.to_csv(output_path)
        print(f"BAŞARILI: Veri şu dosyaya kaydedildi -> {output_path}")
    else:
        print("BAŞARISIZ: Korku ve Açgözlülük Endeksi verisi çekilemedi.")
