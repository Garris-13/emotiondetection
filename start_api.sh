#!/bin/bash

# ========================================
#  表情识别系统 - Linux/macOS 启动脚本
# ========================================

# 设置项目根目录为当前脚本所在目录
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_DIR"

# ========================================================
#  配置虚拟环境路径
# ========================================================
VENV_NAME=".venv"
VENV_DIR="$PROJECT_DIR/$VENV_NAME"
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

echo "========================================"
echo " 正在初始化表情识别系统..."
echo "========================================"

# 检查虚拟环境是否存在，不存在则创建
if [ ! -f "$VENV_PYTHON" ]; then
    echo "[初始化] 未检测到虚拟环境，正在当前目录下创建..."

    # 检查系统是否安装了 python3
    if ! command -v python3 &> /dev/null; then
        echo "❌ 错误: 未找到 python3 命令，请先安装 Python 3。"
        exit 1
    fi

    # 创建虚拟环境
    python3 -m venv "$VENV_DIR"

    if [ $? -ne 0 ]; then
        echo "❌ 创建虚拟环境失败！您可能需要安装 python3-venv 包 (例如: sudo apt install python3-venv)。"
        exit 1
    fi
    echo "✅ 虚拟环境创建成功！"
else
    echo "[初始化] 检测到现有虚拟环境，准备启动..."
fi

echo "[1/7] 使用Python: $VENV_PYTHON"

# 1. 清理进程
echo "[2/7] 清理旧的服务进程..."
# 使用 pkill 精准清理当前项目的 Python 进程，避免误杀系统其他 Python 进程
pkill -f "api/api_server.py" || true
pkill -f "python -m http.server" || true
sleep 1

# 2. 安装依赖
echo "[3/7] 检查并安装项目依赖..."
echo "正在升级pip..."
"$VENV_PIP" install --upgrade pip --quiet

echo "正在安装兼容性依赖..."
"$VENV_PIP" install flask==2.3.3 werkzeug==2.3.7 --quiet
"$VENV_PIP" install "flask-cors>=4.0.0" --upgrade --quiet

if [ -f "requirements.txt" ]; then
    echo "从 requirements.txt 安装其他依赖 (这可能需要一些时间)..."
    "$VENV_PIP" install -r requirements.txt --quiet --ignore-installed flask flask-cors werkzeug
fi

# 3. 启动API服务器
echo "[4/7] 启动 API 服务器..."
# 在后台运行，并将日志输出到 api_server.log
nohup "$VENV_PYTHON" api/api_server.py > api_server.log 2>&1 &
API_PID=$!
echo "✅ API 服务器已在后台启动 (PID: $API_PID)，日志详见 api_server.log"

echo "等待 API 服务器初始化（5秒）..."
sleep 5

# 4. 启动HTTP服务器
echo "[5/7] 启动 HTTP 文件服务器..."
# 在后台运行，端口默认使用 8000
nohup "$VENV_PYTHON" -m http.server 8000 > http_server.log 2>&1 &
HTTP_PID=$!
echo "✅ HTTP 服务器已在后台启动 (PID: $HTTP_PID)，日志详见 http_server.log"

echo ""
echo "========================================"
echo "🎉 系统启动完成！"
echo "👉 请在浏览器中访问: http://localhost:8000/emotion_ui.html"
echo "========================================"
echo "提示: 如需停止服务，请运行以下命令："
echo "kill $API_PID $HTTP_PID"