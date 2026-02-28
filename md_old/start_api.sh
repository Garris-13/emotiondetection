#!/bin/bash

# ======================================================================
# 表情识别与健康建议系统 - Linux 启动脚本
# ======================================================================

set -e

echo "======================================================================"
echo "          表情识别与健康建议系统 - 一键启动脚本"
echo "======================================================================"
echo

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ======================== 配置 ========================
VENV_DIR="$SCRIPT_DIR/.venv"
API_PORT=5000
WEB_PORT=8000

# ======================== 函数定义 ========================

check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        echo "[错误] 未检测到 Python，请先安装 Python 3.8+"
        exit 1
    fi
    echo "[OK] Python: $($PYTHON_CMD --version)"
}

setup_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        echo "[初始化] 创建虚拟环境..."
        $PYTHON_CMD -m venv "$VENV_DIR"
        echo "[OK] 虚拟环境创建成功: $VENV_DIR"
    else
        echo "[OK] 虚拟环境已存在"
    fi
    
    # 激活虚拟环境
    source "$VENV_DIR/bin/activate"
    PYTHON_CMD="$VENV_DIR/bin/python"
    echo "[OK] 虚拟环境已激活"
}

install_deps() {
    echo "[安装] 升级 pip..."
    $PYTHON_CMD -m pip install --upgrade pip -q
    
    echo "[安装] 安装项目依赖..."
    if [ -f "requirements.txt" ]; then
        $PYTHON_CMD -m pip install -r requirements.txt -q
        echo "[OK] 依赖安装完成"
    else
        echo "[警告] requirements.txt 不存在，安装核心依赖..."
        $PYTHON_CMD -m pip install flask flask-cors torch torchvision pillow numpy opencv-python-headless openai -q
    fi
}

check_model() {
    echo
    echo "[检查] 模型文件..."
    if [ -f "best_model.pth" ]; then
        MODEL_SIZE=$(du -h best_model.pth | cut -f1)
        echo "[OK] 模型文件已找到: best_model.pth ($MODEL_SIZE)"
    else
        echo "[警告] 模型文件未找到，系统将使用模拟模式"
    fi
}

check_env() {
    echo
    echo "[检查] 环境变量..."
    if [ -n "$DASHSCOPE_API_KEY" ]; then
        echo "[OK] DASHSCOPE_API_KEY 已设置"
    else
        echo "[提示] 未设置 DASHSCOPE_API_KEY，大模型功能不可用"
        echo "       设置方法: export DASHSCOPE_API_KEY='your-api-key'"
    fi
}

check_device() {
    echo
    echo "[检查] 计算设备..."
    $PYTHON_CMD -c "import torch; print('[OK] GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU 模式')" 2>/dev/null || echo "[OK] 使用 CPU"
}

create_dirs() {
    echo
    echo "[初始化] 创建目录结构..."
    mkdir -p data/monitor_results/images
    mkdir -p data/monitor_results/results
    mkdir -p data/comprehensive_results
    echo "[OK] 目录结构已创建"
}

kill_old_processes() {
    echo
    echo "[清理] 检查端口占用..."
    
    # 清理端口 5000
    if lsof -i :$API_PORT &>/dev/null; then
        echo "       端口 $API_PORT 被占用，正在清理..."
        fuser -k $API_PORT/tcp 2>/dev/null || true
        sleep 1
    fi
    
    # 清理端口 8000
    if lsof -i :$WEB_PORT &>/dev/null; then
        echo "       端口 $WEB_PORT 被占用，正在清理..."
        fuser -k $WEB_PORT/tcp 2>/dev/null || true
        sleep 1
    fi
    echo "[OK] 端口清理完成"
}

start_api_server() {
    echo
    echo "[启动] API 服务器..."
    cd "$SCRIPT_DIR"
    
    if [ -f "api/api_server.py" ]; then
        $PYTHON_CMD api/api_server.py &
        API_PID=$!
        echo "[OK] API 服务器已启动 (PID: $API_PID)"
    else
        echo "[错误] api/api_server.py 不存在"
        exit 1
    fi
}

start_web_server() {
    echo "[启动] Web 文件服务器..."
    cd "$SCRIPT_DIR"
    $PYTHON_CMD -m http.server $WEB_PORT &
    WEB_PID=$!
    echo "[OK] Web 服务器已启动 (PID: $WEB_PID)"
}

show_info() {
    echo
    echo "======================================================================"
    echo "  系统启动完成!"
    echo "======================================================================"
    echo
    echo "  API 服务器:  http://localhost:$API_PORT"
    echo "  健康检查:    http://localhost:$API_PORT/health"
    echo "  Web 界面:    http://localhost:$WEB_PORT/examples/emotion_ui.html"
    echo
    echo "  测试命令:"
    echo "    curl http://localhost:$API_PORT/health"
    echo "    curl http://localhost:$API_PORT/emotions"
    echo
    echo "  按 Ctrl+C 停止所有服务"
    echo "======================================================================"
}

cleanup() {
    echo
    echo "[停止] 正在关闭服务..."
    [ -n "$API_PID" ] && kill $API_PID 2>/dev/null
    [ -n "$WEB_PID" ] && kill $WEB_PID 2>/dev/null
    echo "[OK] 服务已停止"
    exit 0
}

# ======================== 主流程 ========================

# 捕获 Ctrl+C 信号
trap cleanup SIGINT SIGTERM

echo "[1/8] 检查 Python 环境..."
check_python

echo
echo "[2/8] 设置虚拟环境..."
setup_venv

echo
echo "[3/8] 安装依赖..."
install_deps

echo
echo "[4/8] 检查模型文件..."
check_model

echo
echo "[5/8] 检查环境变量..."
check_env

echo
echo "[6/8] 检查计算设备..."
check_device

echo
echo "[7/8] 初始化目录..."
create_dirs
kill_old_processes

echo
echo "[8/8] 启动服务..."
start_api_server
sleep 3
start_web_server

show_info

# 等待进程
wait
