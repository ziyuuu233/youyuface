@echo off
cd /d "%~dp0"

echo Checking Python...
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.11 not found. Please install Python 3.11 with "Add to PATH" checked.
    pause
    exit /b 1
)

if not exist "venv" (
    echo Creating virtual environment...
    py -3.11 -m venv venv
)

echo Installing dependencies (first time may take a while)...
venv\Scripts\python.exe -m pip install --upgrade pip -q
venv\Scripts\pip.exe install -r requirements.txt

echo.
echo Downloading model files...
venv\Scripts\python.exe scripts\download_models.py

echo.
echo ========================================
echo   Done! Double-click 启动.bat to use.
echo ========================================
pause
