# 实时换脸 - PowerShell 启动脚本
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not (Test-Path "venv\Scripts\python.exe")) {
    Write-Host "[错误] 找不到虚拟环境，请先运行 安装.bat" -ForegroundColor Red
    Read-Host "按回车键退出"
    exit 1
}

Write-Host "正在启动换脸程序..." -ForegroundColor Green
& "venv\Scripts\python.exe" "main.py"
