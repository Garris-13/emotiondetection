#!/bin/bash

# 表情识别与健康建议系统启动脚本 (Linux/macOS)

echo "======================================================================"
echo "          表情识别与健康建议系统 - 一键启动脚本"
echo "======================================================================"
echo

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未检测到Python3，请先安装Python 3.8+"
    exit 1
fi

echo "[1/7] 检查Python版本..."
python3 --version

# 检查是否在deployment目录
if [ ! -f "api/api_server.py" ]; then
    echo
    echo "[错误] 请在deployment目录下运行此脚本"
    echo "当前目录: $(pwd)"
    exit 1
fi

# 检查模型文件
echo
echo "[2/7] 检查模型文件..."
if [ -f "best_model.pth" ]; then
    echo "[OK] 模型文件已找到"
else
    echo "[警告] 模型文件未找到，尝试从checkpoints复制..."
    if [ -f "../checkpoints_optimized/best_model.pth" ]; then
        cp "../checkpoints_optimized/best_model.pth" "best_model.pth"
        echo "[OK] 模型文件已复制"
    else
        echo "[错误] 无法找到模型文件"
        echo "请确保 best_model.pth 在以下位置之一:"
        echo "  - deployment/best_model.pth"
        echo "  - checkpoints_optimized/best_model.pth"
        exit 1
    fi
fi

# 检查前端页面
echo
echo "[3/7] 检查前端页面..."
if [ -f "examples/emotion_ui.html" ]; then
    echo "[OK] 前端页面已找到: examples/emotion_ui.html"
else
    echo "[警告] 前端页面未找到，尝试创建..."
    echo "正在创建基本的前端页面..."
    create_basic_frontend
fi

# 检查依赖
echo
echo "[4/7] 检查依赖包..."
if python3 -c "import torch, flask, PIL" 2>/dev/null; then
    echo "[OK] 依赖包已安装"
else
    echo "[警告] 部分依赖包未安装"
    read -p "是否现在安装依赖包? (y/n): " INSTALL
    if [ "$INSTALL" = "y" ] || [ "$INSTALL" = "Y" ]; then
        echo "正在安装依赖..."
        pip3 install -r requirements.txt
        if [ $? -ne 0 ]; then
            echo "[警告] 依赖安装失败，尝试安装核心包..."
            pip3 install flask flask-cors torch torchvision pillow requests numpy
        fi
    else
        echo "请手动安装依赖: pip3 install -r requirements.txt"
        exit 1
    fi
fi

# 检查GPU
echo
echo "[5/7] 检查计算设备..."
python3 -c "import torch; print('[OK] 使用GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '[OK] 使用CPU')"

# 启动Web服务器（后台）
echo
echo "[6/7] 启动Web服务器..."
start_web_server &

# 等待Web服务器启动
sleep 2

# 自动打开浏览器
echo
echo "[7/7] 自动打开浏览器..."
if command -v xdg-open &> /dev/null; then
    xdg-open "http://localhost:8000/examples/emotion_ui.html" &
elif command -v open &> /dev/null; then
    open "http://localhost:8000/examples/emotion_ui.html" &
else
    echo "[提示] 请手动在浏览器中访问: http://localhost:8000/examples/emotion_ui.html"
fi

# 启动API服务器
echo
echo "======================================================================"
echo "[状态] 系统正在启动..."
echo "======================================================================"
echo "API服务器: http://localhost:5000"
echo "Web界面: http://localhost:8000/examples/emotion_ui.html"
echo "按 Ctrl+C 停止所有服务"
echo "======================================================================"
echo

cd api
python3 api_server.py

# 清理函数
function create_basic_frontend() {
    cat > examples/emotion_ui.html << 'EOF'
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>表情识别与健康建议系统</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #333; }
        .upload-area { border: 2px dashed #ccc; padding: 50px; text-align: center; margin: 20px 0; }
        .result { display: flex; gap: 20px; margin-top: 30px; }
        .panel { flex: 1; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
        .emotion-bar { height: 20px; background: #f0f0f0; margin: 5px 0; border-radius: 10px; }
        .emotion-fill { height: 100%; border-radius: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>表情识别与健康建议系统</h1>
        <p>上传面部图像，系统将分析情绪并提供健康建议</p>
        
        <div class="upload-area">
            <h3>上传面部图像</h3>
            <input type="file" id="imageInput" accept="image/*">
            <br><br>
            <button onclick="analyzeImage()">开始分析</button>
        </div>
        
        <div id="preview" style="display:none;">
            <img id="imagePreview" width="200">
        </div>
        
        <div class="result" id="resultSection" style="display:none;">
            <div class="panel">
                <h3>情绪分析</h3>
                <div id="emotionResults"></div>
            </div>
            <div class="panel">
                <h3>健康建议</h3>
                <div id="adviceResults"></div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('imageInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('imagePreview').src = e.target.result;
                    document.getElementById('preview').style.display = 'block';
                }
                reader.readAsDataURL(file);
            }
        });

        async function analyzeImage() {
            const fileInput = document.getElementById('imageInput');
            if (!fileInput.files[0]) {
                alert('请先选择图像');
                return;
            }
            
            const formData = new FormData();
            formData.append('image', fileInput.files[0]);
            formData.append('user_context', JSON.stringify({
                age_group: 'adult',
                has_support_system: true
            }));

            try {
                const response = await fetch('http://localhost:5000/predict_with_advice', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const result = await response.json();
                    displayResults(result);
                } else {
                    throw new Error('请求失败');
                }
            } catch (error) {
                alert('分析失败: ' + error.message);
            }
        }

        function displayResults(result) {
            document.getElementById('resultSection').style.display = 'flex';

            // 显示情绪分析结果
            const prediction = result.prediction;
            let emotionHtml = `<h4>${prediction.emotion_zh} (${(prediction.confidence*100).toFixed(1)}%)</h4>`;
            emotionHtml += '<p>情绪概率分布:</p>';

            Object.entries(prediction.probabilities).forEach(([emotion, prob]) => {
                const width = prob * 100;
                emotionHtml += `
                    <div>${emotion}: ${(prob*100).toFixed(1)}%</div>
                    <div class="emotion-bar">
                        <div class="emotion-fill" style="width: ${width}%; background: #3498db;"></div>
                    </div>
                `;
            });

            document.getElementById('emotionResults').innerHTML = emotionHtml;

            // 显示健康建议
            const advice = result.health_advice_report;
            let adviceHtml = `<p>${advice.health_advice.description}</p>`;

            adviceHtml += '<h4>立即行动:</h4><ul>';
            advice.health_advice.immediate_actions.forEach(action => {
                adviceHtml += `<li>${action}</li>`;
            });
            adviceHtml += '</ul>';

            adviceHtml += '<h4>日常贴士:</h4><ul>';
            advice.health_advice.daily_tips.forEach(tip => {
                adviceHtml += `<li>${tip}</li>`;
            });
            adviceHtml += '</ul>';

            document.getElementById('adviceResults').innerHTML = adviceHtml;
        }
    </script>
</body>
</html>
EOF
    echo "[INFO] 已创建基本的前端页面"
}

function start_web_server() {
    cd examples
    python3 -m http.server 8000
}