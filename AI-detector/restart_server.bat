@echo off
echo Restarting AI Detector Server...
taskkill /F /IM python.exe /T 2>nul
taskkill /F /IM uvicorn.exe /T 2>nul
timeout /t 2 /nobreak >nul
cd /d "%~dp0"
start /b uvicorn app.main:app --host 127.0.0.1 --port 8000
echo Server is starting at http://127.0.0.1:8000
echo Please wait 5-10 seconds before refreshing the browser.
pause
