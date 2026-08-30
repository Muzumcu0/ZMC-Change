# ZMC-Change — Yapay Zeka Destekli Finansal Tahmin ve Simülasyon Platformu

ZMC-Change, makine öğrenmesi modelleri ve piyasa verilerini bir araya getirerek
finansal tahmin ve strateji testi (backtesting) yapan bir platformdur. Proje;
tahmin motoru, geçmiş veri üzerinde strateji simülasyonu ve haber/sosyal medya
kaynaklarından duygu analizini tek bir sistemde birleştirmeyi hedefler.

## Özellikler

- **Tahmin Motoru** — XGBoost ve LightGBM modelleri, Optuna ile hiperparametre
  optimizasyonu kullanılarak eğitilir.
- **Arena Simülasyonu** — Geliştirilen stratejilerin geçmiş veriler üzerinde
  test edilmesini sağlayan bir backtesting ortamı.
- **Veri Füzyonu** — CCXT üzerinden alınan borsa verileri, Reddit ve haber
  kaynaklarından NLP/duygu analiziyle çıkarılan sinyallerle birleştirilir.
- **API & Görselleştirme** — FastAPI tabanlı asenkron bir sunucu üzerinden
  Plotly ile interaktif grafikler sunulur.

## Klasör Yapısı

```
ZMC-Change/
├── src/                     # API ve uygulama kaynak kodu
├── arena_simulator/         # Backtesting / strateji simülasyon modülü
├── models/arena_models/     # Eğitilmiş model dosyaları
├── notebooks/               # Araştırma ve deney defterleri
├── launcher.py               # Uygulamayı başlatan yardımcı script
├── requirements.txt          # Python bağımlılıkları
├── start_api.bat              # API sunucusunu başlatır (Windows)
└── update_data.bat            # Veri güncelleme script'i (Windows)
```

## Kurulum

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Proje, Reddit API'sinden veri çekebilmek için kimlik bilgilerine ihtiyaç duyar.
Kök dizinde bir `.env` dosyası oluşturup aşağıdaki değerleri kendi Reddit API
bilgilerinle doldur (bu dosya `.gitignore` ile repoya dahil edilmez):

```
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=
```

## Çalıştırma

```bash
start_api.bat
```

veya elle:

```bash
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

## Durum

Proje aktif geliştirme aşamasındadır; bazı modüller (ör. arena simülasyonu,
veri füzyonu pipeline'ı) deneysel niteliktedir ve zamanla genişletilmektedir.

## Teknolojiler

Python · FastAPI · XGBoost · LightGBM · Optuna · CCXT · Plotly
