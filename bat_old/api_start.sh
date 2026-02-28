#!/bin/bash

# ========================================
# 表情识别系统 - Linux启动脚本
# ========================================

echo -e "\033[1;32m========================================\033[0m"
echo -e "\033[1;32m  表情识别系统 - Linux启动脚本\033[0m"
echo -e "\033[1;32m========================================\033[0m"
echo ""

# 设置颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 使用虚拟环境的Python
VENV_PYTHON="./.venv1/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    echo -e "${RED}错误: 未找到虚拟环境 Python 可执行文件${NC}"
    echo "请确保虚拟环境位于 .venv1 目录中"
    exit 1
fi

echo -e "[1/6] ${BLUE}使用Python: $VENV_PYTHON${NC}"

# 1. 清理进程
echo -e "[2/6] ${BLUE}清理旧的Python进程...${NC}"
pkill -f "python.*api_server.py" 2>/dev/null
pkill -f "python.*http.server" 2>/dev/null
sleep 2

# 2. 检查端口占用
echo -e "[3/6] ${BLUE}检查端口占用...${NC}"

# 检查端口5000
if lsof -ti:5000 >/dev/null 2>&1; then
    echo -e "${YELLOW}端口5000被占用，正在清理...${NC}"
    lsof -ti:5000 | xargs kill -9 2>/dev/null
    sleep 1
fi

# 检查端口8000
if lsof -ti:8000 >/dev/null 2>&1; then
    echo -e "${YELLOW}端口8000被占用，正在清理...${NC}"
    lsof -ti:8000 | xargs kill -9 2>/dev/null
    sleep 1
fi

# 3. 创建目录结构
echo -e "[4/6] ${BLUE}创建目录结构...${NC}"
mkdir -p "data/monitor_results/images"
mkdir -p "data/monitor_results/results"
mkdir -p "data/camera_test"

# 4. 启动API服务器
echo -e "[5/6] ${BLUE}启动API服务器...${NC}"
gnome-terminal --title="表情识别API服务器" -- bash -c "cd '$PWD' && '$VENV_PYTHON' api/api_server.py; exec bash" &
# 或者使用 xterm（如果 gnome-terminal 不可用）:
# xterm -title "表情识别API服务器" -e "cd '$PWD' && '$VENV_PYTHON' api/api_server.py; bash" &

echo -e "${GREEN}等待API服务器启动（5秒）...${NC}"
sleep 5

# 5. 启动HTTP服务器
echo -e "[6/6] ${BLUE}启动HTTP文件服务器...${NC}"
gnome-terminal --title="HTTP文件服务器" -- bash -c "cd '$PWD' && '$VENV_PYTHON' -m http.server 8000; exec bash" &
# 或者使用 xterm:
# xterm -title "HTTP文件服务器" -e "cd '$PWD' && '$VENV_PYTHON' -m http.server 8000; bash" &

echo ""
echo -e "\033[1;32m========================================\033[0m"
echo -e "\033[1;32m✅ 系统启动完成！\033[0m"
echo ""
echo -e "\033[1;34m📍 访问地址：\033[0m"
echo -e "    API服务器:    ${GREEN}http://localhost:5000${NC}"
echo -e "    前端界面:     ${GREEN}http://127.0.0.1:8000/examples/emotion_ui.html${NC}"
echo ""
echo -e "\033[1;34m🎯 摄像头状态：\033[0m"

# 检测摄像头（Linux版本）
echo -e "${GREEN}检测摄像头设备...${NC}"
CAM_COUNT=0
if command -v v4l2-ctl &> /dev/null; then
    CAM_COUNT=$(v4l2-ctl --list-devices | grep -c "/dev/video")
    echo -e "    ✅ 检测到 ${CAM_COUNT} 个摄像头设备"
    echo -e "    📷 可用摄像头索引: $(seq 0 $((CAM_COUNT-1)) | tr '\n' ' ')"
    echo -e "    🔧 建议使用摄像头索引: 0"
else
    echo -e "    ${YELLOW}⚠️  无法检测摄像头（请安装 v4l-utils）${NC}"
    echo -e "    运行: sudo apt-get install v4l-utils"
fi

echo ""
echo -e "\033[1;34m🚀 使用步骤：\033[0m"
echo -e "    1. 打开前端界面"
echo -e "    2. 检查API连接状态"
echo -e "    3. 设置摄像头索引为 0"
echo -e "    4. 设置抓拍间隔为 5秒"
echo -e "    5. 点击\"开始监测\""
echo -e "    6. 查看 data/monitor_results 目录"
echo -e "\033[1;32m========================================\033[0m"
echo ""

# 自动打开浏览器
echo -e "${BLUE}正在打开浏览器...${NC}"
if command -v xdg-open &> /dev/null; then
    xdg-open "http://127.0.0.1:8000/examples/emotion_ui.html" &
elif command -v firefox &> /dev/null; then
    firefox "http://127.0.0.1:8000/examples/emotion_ui.html" &
elif command -v google-chrome &> /dev/null; then
    google-chrome "http://127.0.0.1:8000/examples/emotion_ui.html" &
else
    echo -e "${YELLOW}无法自动打开浏览器，请手动访问上述URL${NC}"
fi

echo ""
read -p "按 Enter 键继续..."