@echo off
echo ============================================
echo      KRIPTO VERI GUNCELLEME BASLATILIYOR     
echo ============================================
echo.

REM Sanal ortami aktif et
echo Aktif ediliyor: Sanal ortam (venv)...
call .\venv\Scripts\activate.bat
IF %ERRORLEVEL% NEQ 0 (
    echo HATA: Sanal ortam aktif edilemedi. Script durduruluyor.
    pause
    exit /b 1
)
echo Sanal ortam aktif edildi.
echo.

REM Piyasa verilerini guncelle
echo Calistiriliyor: Piyasa Verisi Guncelleme (market_data)...
python src/data_ingestion/collect_market_data.py
IF %ERRORLEVEL% NEQ 0 (
    echo UYARI: Piyasa verisi guncellenirken hata olustu.
) ELSE (
    echo Basarili: Piyasa verileri guncellendi.
)
echo.

REM Korku & Acgozluluk Endeksini guncelle
echo Calistiriliyor: Korku & Acgozluluk Endeksi Guncelleme (fear_greed)...
python src/data_ingestion/collect_fear_greed_index.py
IF %ERRORLEVEL% NEQ 0 (
    echo UYARI: Korku & Acgozluluk Endeksi guncellenirken hata olustu.
) ELSE (
    echo Basarili: Korku & Acgozluluk Endeksi guncellendi.
)
echo.

REM Reddit verilerini guncelle
echo Calistiriliyor: Reddit Verisi Guncelleme (reddit_data)...
python src/data_ingestion/collect_reddit_data.py
IF %ERRORLEVEL% NEQ 0 (
    echo UYARI: Reddit verisi guncellenirken hata olustu.
) ELSE (
    echo Basarili: Reddit verileri guncellendi.
)
echo.

REM Google Trends (Opsiyonel, API limitleri nedeniyle yorumda)
REM echo Calistiriliyor: Google Trends Verisi Guncelleme (google_trends)...
REM python src/data_ingestion/collect_google_trends.py
REM IF %ERRORLEVEL% NEQ 0 (
REM     echo UYARI: Google Trends verisi guncellenirken hata olustu (API limiti olabilir).
REM ) ELSE (
REM     echo Basarili: Google Trends verisi guncellendi.
REM )
REM echo.

echo ============================================
echo      VERI GUNCELLEME TAMAMLANDI           
echo ============================================
echo.
pause
