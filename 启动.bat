@echo off
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found. Please run 安装.bat first.
    pause
    exit /b 1
)

echo Starting FaceSwap app...
venv\Scripts\python.exe main.py

if errorlevel 1 (
    echo.
    echo Program exited with error. Press any key to close...
    pause >nul
)
