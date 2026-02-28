@echo off
chcp 65001 >nul
echo ========================================
echo  表情识别系统 - AI大模型版本启动脚本
echo ========================================
echo.

color 0A

REM 设置阿里云百炼API Key环境变量
set DASHSCOPE_API_KEY=111
echo [1/5] 设置阿里云百炼API Key环境变量

REM 使用虚拟环境的Python
set VENV_PYTHON=D:\python13.3\machinelearing\.venv1\Scripts\python.exe
echo [2/5] 使用Python: %VENV_PYTHON%

REM 清理进程
echo [3/5] 清理旧的Python进程...
taskkill /F /IM python.exe /T 2>nul 2>nul
timeout /t 2 /nobreak >nul

REM 检查端口占用
echo [4/5] 检查端口占用...
netstat -ano | findstr :5000 >nul
if %errorlevel% equ 0 (
    echo 端口5000被占用，正在清理...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5000') do taskkill /F /PID %%a
    timeout /t 1 /nobreak >nul
)

REM 创建目录
echo [5/5] 创建目录结构...
if not exist "data\monitor_results\images" mkdir "data\monitor_results\images"
if not exist "data\monitor_results\results" mkdir "data\monitor_results\results"
if not exist "data\comprehensive_results" mkdir "data\comprehensive_results"

REM 启动API服务器
echo 启动API服务器（AI大模型版本）...
start "表情识别API服务器" cmd /k "cd /d %~dp0 && "%VENV_PYTHON%" api\api_server.py"

echo 等待API服务器启动（5秒）...
timeout /t 5 /nobreak >nul

echo.
echo ========================================
echo ✅ 系统启动完成！
echo.
echo 📍 访问地址：
echo     API服务器:    http://localhost:5000
echo     前端界面:     http://127.0.0.1:8000/examples/emotion_ui.html
echo.
echo 🧠 AI大模型功能：
echo     ✅ 阿里云百炼API Key已配置
echo     🤖 可用模型: qwen3-max, qwen-plus等
echo     🔗 综合情绪分析支持大模型调用
echo.
echo 🚀 新功能：
echo     1. 单张图片"分析情绪"按钮
echo     2. "AI深度分析"按钮
echo     3. "综合情绪分析与建议"调用大模型
echo     4. 新增"AI大模型分析"标签页
echo ========================================
echo.

REM 自动打开浏览器
echo 正在打开浏览器...
start http://127.0.0.1:8000/examples/emotion_ui.html

pause