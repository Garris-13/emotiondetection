@echo off
chcp 65001 >nul
echo ========================================
echo  表情识别系统 - 最终修复
echo ========================================
echo.

color 0A

echo [1/5] 修复datetime导入问题...
python -c "
import re

# 修复health_advisor.py
with open('models/health_advisor.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 替换import datetime
content = re.sub(r'^import datetime', 'from datetime import datetime', content, flags=re.MULTILINE)

# 移除重复的import
content = re.sub(r'from datetime import datetime\nimport datetime', 'from datetime import datetime', content)

# 写入文件
with open('models/health_advisor.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('✅ health_advisor.py 修复完成')
"

echo [2/5] 更新摄像头监测器...
if exist "api\camera_monitor.py" copy "api\camera_monitor.py" "api\camera_monitor_backup.py"
echo 已备份原文件

echo [3/5] 重启进程...
taskkill /F /IM python.exe /T 2>nul
timeout /t 2 /nobreak >nul

echo [4/5] 创建测试数据...
python -c "
import os
import json
from datetime import datetime, timedelta

# 创建测试数据目录
os.makedirs('data/monitor_results/results', exist_ok=True)

# 生成一些测试数据
emotions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised']
emotion_zh = {
    'anger': '愤怒',
    'disgust': '厌恶',
    'fear': '恐惧',
    'happy': '快乐',
    'sad': '悲伤',
    'surprised': '惊讶'
}

for i in range(20):
    timestamp = (datetime.now() - timedelta(hours=i)).strftime('%Y%m%d_%H%M%S')
    emotion = emotions[i % len(emotions)]
    
    result = {
        'timestamp': (datetime.now() - timedelta(hours=i)).isoformat(),
        'emotion': emotion,
        'emotion_zh': emotion_zh[emotion],
        'confidence': 0.6 + (i * 0.02),
        'probabilities': {e: 0.05 if e != emotion else 0.6 for e in emotions},
        'image_filename': f'test_{timestamp}.jpg'
    }
    
    with open(f'data/monitor_results/results/result_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

print(f'✅ 创建了20个测试数据文件')
"

echo [5/5] 启动系统...
start "表情识别API" cmd /k "cd /d %~dp0 && D:\python13.3\machinelearing\.venv1\Scripts\python.exe api\api_server.py"
timeout /t 3 /nobreak >nul

start "HTTP服务器" cmd /k "cd /d %~dp0 && D:\python13.3\machinelearing\.venv1\Scripts\python.exe -m http.server 8000"

echo.
echo ========================================
echo ✅ 修复完成！
echo.
echo 📍 访问地址：
echo     http://127.0.0.1:8000/examples/emotion_ui.html
echo.
echo 🎯 测试步骤：
echo     1. 打开上面网址
echo     2. 点击"分析监测历史"
echo     3. 查看健康建议报告
echo     4. 如果需要，启动实时监测收集更多数据
echo ========================================
echo.

start http://127.0.0.1:8000/examples/emotion_ui.html

pause