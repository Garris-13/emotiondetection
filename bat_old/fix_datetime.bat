@echo off
chcp 65001 >nul
echo ========================================
echo  修复datetime导入问题
echo ========================================
echo.

color 0A

echo [1/4] 备份原文件...
copy "models\health_advisor.py" "models\health_advisor.py.backup"

echo [2/4] 替换datetime导入...
python -c "
import re

# 读取文件
with open('models/health_advisor.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 替换import datetime为from datetime import datetime
content = content.replace('import datetime', 'from datetime import datetime')

# 确保移除重复的datetime.now()
content = content.replace('datetime.datetime.now()', 'datetime.now()')
content = content.replace('datetime.now().now()', 'datetime.now()')

# 写入文件
with open('models/health_advisor.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('✅ datetime导入已修复')
"

echo [3/4] 测试修复...
python -c "
try:
    from models.health_advisor import HealthAdvisor, EmotionResult
    advisor = HealthAdvisor()
    print('✅ HealthAdvisor导入成功')
    
    # 测试datetime
    from datetime import datetime
    print(f'✅ datetime.now() 工作正常: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
except Exception as e:
    print(f'❌ 测试失败: {e}')
"

echo [4/4] 重启系统...
taskkill /F /IM python.exe /T 2>nul
timeout /t 2 /nobreak >nul

echo.
echo ========================================
echo ✅ 修复完成！
echo 请重新运行 start_correct.bat
echo ========================================
pause