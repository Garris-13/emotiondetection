@echo off
chcp 65001 >nul
echo ========================================
echo  表情识别系统 - Python 3.13 兼容版启动脚本 (极速启动)
echo ========================================
echo.
color 0A

REM ========================================================
REM  配置虚拟环境路径 (当前目录下的 .venv)
REM ========================================================
set "VENV_NAME=.venv"
set "VENV_DIR=%~dp0%VENV_NAME%"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

REM 检查虚拟环境是否存在，不存在则创建
if not exist "%VENV_PYTHON%" (
    echo [初始化] 未检测到虚拟环境，正在当前目录下创建...
    echo 目标路径: %VENV_DIR%

    REM 尝试使用系统 Python 创建虚拟环境
    python -m venv "%VENV_DIR%"

    if %errorlevel% neq 0 (
        echo.
        echo ❌ 创建虚拟环境失败！
        echo 请确保您的电脑已安装 Python 并且已添加到系统环境变量 PATH 中。
        pause
        exit /b
    )
    echo ✅ 虚拟环境创建成功！
) else (
    echo [初始化] 检测到现有虚拟环境，准备启动...
)

echo [1/7] 使用Python: %VENV_PYTHON%

REM 1. 清理进程
echo [2/7] 清理旧的Python进程...
taskkill /F /IM python.exe >nul 2>&1
timeout /t 1 /nobreak >nul

REM ========================================================
REM 5. 检查并安装项目依赖 (清华大学镜像源 + 进度可见版)
REM ========================================================
set "INSTALL_MARKER=%VENV_DIR%\.deps_installed"

if not exist "%INSTALL_MARKER%" (
    echo [5/7] 首次运行或环境缺失，正在安装项目依赖...
    echo 💡 提示: 已开启进度显示，并全程强制使用【清华大学镜像源】加速下载！

    echo 正在升级pip...
    "%VENV_PYTHON%" -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

    echo 正在安装兼容性依赖...
    "%VENV_PYTHON%" -m pip install flask==2.3.3 werkzeug==2.3.7 -i https://pypi.tuna.tsinghua.edu.cn/simple
    "%VENV_PYTHON%" -m pip install flask-cors>=4.0.0 --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple

    if exist "requirements.txt" (
        echo.
        echo 🚀 正在从 requirements.txt 安装核心依赖...
        echo ⚠️ 注意: 包含 PyTorch 等超大文件，正在通过清华源极速下载，请耐心等待进度条走完！
        "%VENV_PYTHON%" -m pip install -r requirements.txt --ignore-installed flask flask-cors werkzeug -i https://pypi.tuna.tsinghua.edu.cn/simple
    )

    REM 安装完成后创建标记文件，下次启动将跳过此阶段
    echo Done > "%INSTALL_MARKER%"
    echo ✅ 依赖安装完成！
) else (
    echo [5/7] ✅ 检测到依赖已安装，跳过安装步骤，极速启动中...
)

REM 5. 启动API服务器
echo [6/7] 启动API服务器...
start "表情识别API服务器" cmd /k "cd /d %~dp0 && "%VENV_PYTHON%" api\api_server.py"

echo 等待API服务器启动，约5秒...
timeout /t 5 /nobreak >nul

REM 6. 启动HTTP服务器
echo [7/7] 启动HTTP文件服务器...
start "HTTP文件服务器" cmd /k "cd /d %~dp0 && "%VENV_PYTHON%" -m http.server 8000"

echo.
echo ========================================
echo 🎉 系统启动完成！
echo 👉 请在浏览器中访问: http://localhost:8000/emotion_ui.html
echo ========================================
start http://127.0.0.1:8000/examples/emotion_ui.html
pause