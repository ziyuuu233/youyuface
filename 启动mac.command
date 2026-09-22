#!/bin/bash
# macOS 启动脚本：双击即可运行
# 使用方法：
#   1. 先运行: chmod +x 启动mac.command
#   2. 双击此文件启动程序
#   或者终端执行: bash 启动mac.command

cd "$(dirname "$0")"

# 检查 Python 版本
if command -v python3 &> /dev/null; then
    PY=python3
elif command -v python &> /dev/null; then
    PY=python
else
    echo "[ERROR] 未找到 Python，请先安装 Python 3.10+ (https://www.python.org/downloads/)"
    read -p "按回车键退出..."
    exit 1
fi

# 首次运行：创建虚拟环境并安装依赖
if [ ! -d "venv_mac" ]; then
    echo "首次运行，正在创建虚拟环境..."
    $PY -m venv venv_mac
    echo "正在安装依赖（首次约需 5-10 分钟，请耐心等待）..."
    venv_mac/bin/pip install --upgrade pip
    venv_mac/bin/pip install -r requirements-mac.txt
    echo "依赖安装完成！"
fi

# 启动程序
venv_mac/bin/python main.py

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] 程序异常退出，请检查上方错误信息"
    read -p "按回车键退出..."
fi
