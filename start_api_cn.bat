@echo off
setlocal
echo ========================================
echo Emotion Detection - Fast Startup Script
echo ========================================
echo.
color 0A

set "VENV_NAME=.venv"
set "ROOT_DIR=%~dp0"
set "VENV_DIR=%ROOT_DIR%%VENV_NAME%"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "INSTALL_MARKER=%VENV_DIR%\.deps_installed"

if not exist "%VENV_PYTHON%" (
    echo [Init] No virtual environment found. Creating: "%VENV_DIR%"
    python -m venv "%VENV_DIR%"
    if %errorlevel% neq 0 (
        echo [Error] Failed to create virtual environment.
        echo Please install Python and add it to PATH.
        pause
        exit /b 1
    )
    echo [Init] Virtual environment created.
) else (
    echo [Init] Existing virtual environment detected.
)

echo [1/7] Python: "%VENV_PYTHON%"

echo [2/7] Cleaning old python processes...
taskkill /F /IM python.exe >nul 2>&1
timeout /t 1 /nobreak >nul

if not exist "%INSTALL_MARKER%" (
    echo [5/7] Installing dependencies from Tsinghua mirror...
    "%VENV_PYTHON%" -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
    "%VENV_PYTHON%" -m pip install flask==2.3.3 werkzeug==2.3.7 -i https://pypi.tuna.tsinghua.edu.cn/simple
    "%VENV_PYTHON%" -m pip install "flask-cors>=4.0.0" --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple

    if exist "%ROOT_DIR%requirements.txt" (
        "%VENV_PYTHON%" -m pip install -r "%ROOT_DIR%requirements.txt" --ignore-installed flask flask-cors werkzeug -i https://pypi.tuna.tsinghua.edu.cn/simple
    )

    echo Done>"%INSTALL_MARKER%"
    echo [5/7] Dependency install completed.
) else (
    echo [5/7] Dependencies already installed. Skipping.
)

echo [6/7] Starting API server...
start "Emotion API Server" cmd /k "cd /d ""%ROOT_DIR%"" && ""%VENV_PYTHON%"" api\api_server.py"

echo Waiting for API startup...
timeout /t 5 /nobreak >nul

echo [7/7] Starting static file server...
start "Emotion HTTP Server" cmd /k "cd /d ""%ROOT_DIR%"" && ""%VENV_PYTHON%"" -m http.server 8000"

echo.
echo ========================================
echo Startup completed.
echo Open: http://127.0.0.1:8000/examples/emotion_ui.html
echo ========================================
start http://127.0.0.1:8000/examples/emotion_ui.html
pause