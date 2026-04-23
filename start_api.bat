@echo off
TITLE ZMC-Change API Server
ECHO API Sunucusu baslatiliyor...
ECHO.
ECHO VENV (Sanal Ortam) Aktiflestiriliyor...
CALL venv\Scripts\activate

ECHO.
ECHO FastAPI (Uvicorn) sunucusu --reload moduyla baslatiliyor...
ECHO.
ECHO *** API'yi durdurmak icin bu pencereyi kapatmaniz (X) yeterlidir. ***
ECHO.

uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000