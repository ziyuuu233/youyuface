@echo off
cd /d "%~dp0"
py -3.11 --version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python 3.11 not found. Install Python 3.11 with Add to PATH.
  pause
  exit /b 1
)
if not exist venv py -3.11 -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip -q
venv\Scripts\pip.exe install -r requirements.txt
venv\Scripts\python.exe scripts\download_models.py
echo Done. Run start.bat now.
pause
