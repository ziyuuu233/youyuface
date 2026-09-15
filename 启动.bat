@echo off
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
  echo [ERROR] venv not found, please run install.bat first.
  pause
  exit /b 1
)
venv\Scripts\python.exe main.py
if errorlevel 1 pause
