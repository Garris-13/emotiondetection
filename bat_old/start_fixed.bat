@echo off
chcp 65001 >nul
echo ========================================
echo  表情识别系统 - 完全修复版启动脚本
echo ========================================
echo.

color 0A

REM 切换到项目目录
cd /d "D:\deployment"

REM 1. 清理所有可能冲突的Python进程
echo [1/6] 清理旧的Python进程...
taskkill /F /IM python.exe /T 2>nul
timeout /t 2 /nobreak >nul

REM 2. 检查端口占用并清理
echo [2/6] 检查端口占用...
netstat -ano | findstr :5000 >nul
if %errorlevel% equ 0 (
    echo 端口5000被占用，正在清理...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5000') do taskkill /F /PID %%a
    timeout /t 1 /nobreak >nul
)

REM 3. 检查依赖包
echo [3/6] 检查Python依赖...
python -c "import flask" 2>nul
if errorlevel 1 (
    echo 安装Flask...
    pip install flask flask-cors
)

python -c "import torch" 2>nul
if errorlevel 1 (
    echo 安装PyTorch...
    pip install torch torchvision pillow --index-url https://download.pytorch.org/whl/cpu
)

REM 4. 启动API服务器（使用绝对路径）
echo [4/6] 启动API服务器...
start "表情识别API服务器" cmd /k "cd /d D:\deployment && python api\api_server.py"

echo 等待服务器启动（5秒）...
timeout /t 5 /nobreak >nul

REM 5. 启动HTTP文件服务器
echo [5/6] 启动HTTP服务器...
cd /d "D:\deployment"
start "HTTP文件服务器" cmd /k "python -m http.server 8000 --bind 127.0.0.1"

echo 等待HTTP服务器启动（2秒）...
timeout /t 2 /nobreak >nul

REM 6. 测试连接
echo [6/6] 测试API连接...
curl http://localhost:5000/health 2>nul
if errorlevel 1 (
    echo ❌ API服务器启动失败，正在重试...
    timeout /t 3 /nobreak >nul
    curl http://localhost:5000/health 2>nul
)

echo.
echo ========================================
echo ✅ 系统启动完成！
echo.
echo 📍 访问地址：
echo     API服务器:    http://localhost:5000
echo     前端界面:     http://127.0.0.1:8000/examples/emotion_ui.html
echo.
echo 🔧 快速测试：
echo     1. 浏览器打开: http://127.0.0.1:8000/examples/emotion_ui.html
echo     2. 终端测试: curl http://localhost:5000/health
echo     3. 查看API: http://localhost:5000
echo ========================================
echo.

REM 自动打开浏览器
echo 正在打开浏览器...
start http://127.0.0.1:8000/examples/emotion_ui.html

pause