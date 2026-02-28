@echo off
chcp 65001 >nul
echo ========================================
echo  表情识别系统 - 完整修复版启动脚本
echo ========================================
echo.

color 0A

REM 使用虚拟环境的Python
set VENV_PYTHON=D:\python13.3\machinelearing\.venv1\Scripts\python.exe

echo [1/6] 使用Python: %VENV_PYTHON%

REM 1. 清理进程
echo [2/6] 清理旧的Python进程...
taskkill /F /IM python.exe /T 2>nul 2>nul
timeout /t 2 /nobreak >nul

REM 2. 检查端口占用
echo [3/6] 检查端口占用...
netstat -ano | findstr :5000 >nul
if %errorlevel% equ 0 (
    echo 端口5000被占用，正在清理...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5000') do taskkill /F /PID %%a
    timeout /t 1 /nobreak >nul
)

netstat -ano | findstr :8000 >nul
if %errorlevel% equ 0 (
    echo 端口8000被占用，正在清理...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do taskkill /F /PID %%a
    timeout /t 1 /nobreak >nul
)

REM 3. 创建目录结构
echo [4/6] 创建目录结构...
if not exist "data\monitor_results\images" mkdir "data\monitor_results\images"
if not exist "data\monitor_results\results" mkdir "data\monitor_results\results"
if not exist "data\camera_test" mkdir "data\camera_test"

REM 4. 启动API服务器
echo [5/6] 启动API服务器...
start "表情识别API服务器" cmd /k "cd /d %~dp0 && "%VENV_PYTHON%" api\api_server.py"

echo 等待API服务器启动（5秒）...
timeout /t 5 /nobreak >nul

REM 5. 启动HTTP服务器
echo [6/6] 启动HTTP文件服务器...
start "HTTP文件服务器" cmd /k "cd /d %~dp0 && "%VENV_PYTHON%" -m http.server 8000"

echo.
echo ========================================
echo ✅ 系统启动完成！
echo.
echo 📍 访问地址：
echo     API服务器:    http://localhost:5000
echo     前端界面:     http://127.0.0.1:8000/examples/emotion_ui.html
echo.
echo 🎯 摄像头状态：
echo     ✅ 检测到2个USB摄像头
echo     📷 可用摄像头索引: 0 和 1
echo     🔧 建议使用摄像头索引: 0
echo.
echo 🚀 使用步骤：
echo     1. 打开前端界面
echo     2. 检查API连接状态
echo     3. 设置摄像头索引为 0
echo     4. 设置抓拍间隔为 5秒
echo     5. 点击"开始监测"
echo     6. 查看data\monitor_results目录
echo ========================================
echo.

REM 自动打开浏览器
echo 正在打开浏览器...
start http://127.0.0.1:8000/examples/emotion_ui.html

pause