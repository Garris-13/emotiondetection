@echo off
chcp 65001 >nul
REM 表情识别与健康建议系统启动脚本

echo ======================================================================
echo           表情识别与健康建议系统 - 一键启动
echo ======================================================================
echo.

color 0A

REM 切换到脚本所在目录
cd /d "%~dp0"
echo [1/7] 当前目录: %CD%

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 请先安装Python 3.8+
    pause
    exit /b 1
)

echo [2/7] Python版本: 
python --version

REM 检查API文件
if not exist "api\api_server.py" (
    echo [错误] 找不到API服务器文件
    pause
    exit /b 1
)

REM 检查前端文件
echo.
echo [3/7] 检查前端页面...
if exist "examples\emotion_ui.html" (
    echo [OK] 前端页面已找到: D:\deployment\examples\emotion_ui.html
    set "FRONTEND_PATH=examples\emotion_ui.html"
) else (
    echo [错误] 找不到前端页面！
    echo 请确保 emotion_ui.html 在 examples 目录中
    dir examples\*.html
    pause
    exit /b 1
)

REM 检查模型文件
echo.
echo [4/7] 检查模型文件...
if exist "best_model.pth" (
    echo [OK] 模型文件已找到
) else (
    echo [错误] 找不到模型文件 best_model.pth
    echo 请从训练目录复制: copy ..\checkpoints_optimized\best_model.pth .
    pause
    exit /b 1
)

REM 检查依赖
echo.
echo [5/7] 检查依赖...
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo 安装Flask...
    pip install flask flask-cors
)

python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo 安装PyTorch...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

REM 启动API服务器
echo.
echo [6/7] 启动API服务器 (端口:7860)...
start "API服务器" /MIN cmd /c "cd /d %CD% && python api\api_server.py"

echo 等待服务器启动...
timeout /t 5 /nobreak >nul

REM 测试API
curl http://localhost:7860/health >nul 2>&1
if errorlevel 1 (
    echo [警告] API服务器启动可能较慢...
    timeout /t 3 /nobreak >nul
)

REM 启动HTTP服务器（用于前端页面）
echo.
echo [7/7] 启动HTTP服务器 (端口:8000)...
cd /d "D:\deployment"
start "HTTP服务器" /MIN cmd /c "python -m http.server 8000"

echo 等待HTTP服务器启动...
timeout /t 2 /nobreak >nul

echo.
echo ======================================================================
echo 🎉 系统启动完成！
echo.
echo 📍 访问信息:
echo     API服务器:     http://localhost:5000
echo     前端界面:      http://localhost:8000/examples/emotion_ui.html
echo     文件路径:      D:\deployment\examples\emotion_ui.html
echo.
echo 🔧 测试命令:
echo     curl http://localhost:5000/health
echo     curl http://localhost:5000/emotions
echo.
echo ⚠️  操作提示:
echo     • 按 [1] 打开前端界面
echo     • 按 [2] 测试API状态
echo     • 按 [3] 查看API文档
echo     • 按 [S] 停止所有服务
echo     • 按 [Q] 退出
echo ======================================================================
echo.

:menu
echo.
echo [1] 打开前端界面
echo [2] 测试API状态
echo [3] 查看API文档
echo [S] 停止服务器
echo [Q] 退出
echo.
set /p CHOICE="请选择操作: "

if "%CHOICE%"=="1" (
    echo 正在打开前端界面...
    start http://localhost:8000/examples/emotion_ui.html
    goto menu
)

if "%CHOICE%"=="2" (
    echo API状态测试:
    curl http://localhost:5000/health
    echo.
    goto menu
)

if "%CHOICE%"=="3" (
    echo 打开API文档...
    start http://localhost:5000
    goto menu
)

if /i "%CHOICE%"=="s" (
    goto stop
)

if /i "%CHOICE%"=="q" (
    goto stop
)

goto menu

:stop
echo.
echo 正在停止服务...
taskkill /F /IM python.exe /T 2>nul
echo 服务已停止
timeout /t 2
exit /b 0