# 快速开始指南

本指南将帮助您快速部署和运行表情识别API。

---

## 📋 前置要求

### 系统要求
- **操作系统**: Windows 10/11, Linux, macOS
- **Python**: 3.8 或更高版本
- **内存**: 至少 4GB RAM
- **GPU**: 可选，NVIDIA GPU with CUDA (推荐用于加速)

### 检查Python版本
```bash
python --version
```

---

## 🚀 快速开始（5分钟）

### 步骤 1: 进入部署目录
```bash
cd deployment1
```

### 步骤 2: 安装依赖
```bash
pip install -r requirements.txt
```

### 步骤 3: 启动API服务器
```bash
cd api
python api_server.py
```

### 步骤 4: 测试API
打开新的终端窗口：
```bash
# 检查健康状态
curl http://localhost:5000/health

# 或使用Python客户端测试
python api_client.py
```

### 步骤 5: 使用Web界面
在浏览器中打开：
```
deployment/examples/example_web.html
```

---

## 📦 详细安装步骤

### Windows 用户

#### 1. 克隆或进入项目目录
```powershell
cd E:\Competition\ExpressionAck\deployment
```

#### 2. 创建虚拟环境（推荐）
```powershell
python -m venv venv
.\venv\Scripts\activate
```

#### 3. 安装依赖
```powershell
pip install -r requirements.txt
```

#### 4. 验证安装
```powershell
python -c "import torch; print('PyTorch版本:', torch.__version__)"
python -c "import flask; print('Flask版本:', flask.__version__)"
```

### Linux/macOS 用户

#### 1. 进入项目目录
```bash
cd /path/to/ExpressionAck/deployment1
```

#### 2. 创建虚拟环境（推荐）
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. 安装依赖
```bash
pip install -r requirements.txt
```

#### 4. 验证安装
```bash
python -c "import torch; print('PyTorch版本:', torch.__version__)"
python -c "import flask; print('Flask版本:', flask.__version__)"
```

---

## 🎯 运行API服务器

### 方式1: 命令行运行（开发模式）

```bash
cd deployment1/api
python api_server.py
```

服务器将在 `http://localhost:5000` 启动。

控制台输出：
```
======================================================================
🚀 启动表情识别 API
======================================================================
使用设备: cuda
✅ 模型加载成功
✅ 预处理器初始化完成

======================================================================
✅ API 服务器启动中...
======================================================================
访问地址: http://localhost:5000
API 文档: http://localhost:5000/
======================================================================
 * Running on http://0.0.0.0:5000
```

### 方式2: 后台运行（生产模式）

#### Windows (PowerShell):
```powershell
Start-Process python -ArgumentList "api_server.py" -WindowStyle Hidden
```

#### Linux/macOS:
```bash
nohup python api_server.py > api.log 2>&1 &
```

---

## 🧪 测试API

### 方法1: 使用Python客户端

```bash
cd deployment1/api
python api_client.py
```

### 方法2: 使用cURL

```bash
# 1. 健康检查
curl http://localhost:5000/health

# 2. 获取支持的表情
curl http://localhost:5000/emotions

# 3. 预测图像（需要准备测试图像）
curl -X POST http://localhost:5000/predict \
  -F "image=@/path/to/your/image.jpg"
```

### 方法3: 使用Web界面

1. 用浏览器打开 `deployment/examples/example_web.html`
2. 点击"选择图像"按钮
3. 选择一张人脸图像
4. 查看识别结果

### 方法4: 使用Python代码

```python
from api.api_client import EmotionRecognitionClient

# 创建客户端
client = EmotionRecognitionClient('http://localhost:5000')

# 预测图像
result = client.predict_from_file('test_image.jpg')

# 打印结果
if result.get('success'):
    print(f"表情: {result['emotion']}")
    print(f"置信度: {result['confidence']:.2%}")
```

---

## 📝 使用示例

### 示例1: 单张图像识别

创建文件 `test_predict.py`:
```python
from api.api_client import EmotionRecognitionClient

client = EmotionRecognitionClient('http://localhost:5000')
result = client.predict_from_file('your_image.jpg')

print(f"预测表情: {result['emotion']}")
print(f"置信度: {result['confidence']:.2%}")
print("\n各类别概率:")
for emotion, prob in result['probabilities'].items():
    print(f"  {emotion}: {prob:.2%}")
```

运行：
```bash
python test_predict.py
```

### 示例2: 批量识别

```python
from api.api_client import EmotionRecognitionClient

client = EmotionRecognitionClient('http://localhost:5000')

# 批量预测多张图像
images = ['image1.jpg', 'image2.jpg', 'image3.jpg']
result = client.predict_batch(images)

if result.get('success'):
    for i, pred in enumerate(result['results'], 1):
        print(f"图像{i}: {pred['emotion']} ({pred['confidence']:.2%})")
```

### 示例3: 集成到Flask应用

```python
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)
API_URL = 'http://localhost:5000'

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    
    files = {'image': request.files['image']}
    response = requests.post(f'{API_URL}/predict', files=files)
    return response.json()

if __name__ == '__main__':
    app.run(port=8080)
```

---

## 🔧 配置选项

### 修改服务器端口

编辑 `api/api_server.py` 最后一行：
```python
app.run(host='0.0.0.0', port=5000, debug=False)
# 修改为其他端口，例如：
app.run(host='0.0.0.0', port=8080, debug=False)
```

### 修改模型路径

编辑 `api/api_server.py` 中的 `initialize_model()` 函数：
```python
model_path = os.path.join(os.path.dirname(__file__), '..', 'best_model.pth')
# 修改为你的模型路径
```

### 启用调试模式

```python
app.run(host='0.0.0.0', port=5000, debug=True)
```

---

## ⚠️ 常见问题

### 问题1: 找不到模块错误

**错误信息**:
```
ModuleNotFoundError: No module named 'torch'
```

**解决方案**:
```bash
pip install torch torchvision
```

### 问题2: 模型文件未找到

**错误信息**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'best_model.pth'
```

**解决方案**:
```bash
# 确保模型文件在正确位置
cd deployment1
ls best_model.pth  # Windows: dir best_model.pth

# 如果不存在，从checkpoints复制
cp ../checkpoints_optimized/best_model.pth .
```

### 问题3: 端口被占用

**错误信息**:
```
OSError: [Errno 98] Address already in use
```

**解决方案**:

Windows:
```powershell
# 查找占用端口的进程
netstat -ano | findstr :5000

# 终止进程 (替换PID)
taskkill /PID <PID> /F
```

Linux/macOS:
```bash
# 查找占用端口的进程
lsof -i :5000

# 终止进程
kill -9 <PID>
```

或者修改为其他端口。

### 问题4: CUDA out of memory

**解决方案**:
- 使用CPU模式（自动检测）
- 减小batch size
- 使用更小的模型

### 问题5: Flask CORS 错误

**错误信息**:
```
No 'Access-Control-Allow-Origin' header
```

**解决方案**:

确保已安装 flask-cors:
```bash
pip install flask-cors
```

---

## 🎨 使用Web界面

### 本地测试

1. 确保API服务器正在运行
2. 用浏览器打开 `deployment/examples/example_web.html`
3. 上传图像并查看结果

### 部署到Web服务器

1. 将 `example_web.html` 复制到你的Web服务器
2. 修改其中的API地址：
```javascript
const API_URL = 'http://your-server-ip:5000';
```

---

## 📊 性能监控

### 查看API日志

在运行API服务器的终端中查看实时日志。

### 监控资源使用

#### Windows:
```powershell
# CPU和内存
Get-Process python | Select-Object CPU, PM

# GPU (如果有NVIDIA GPU)
nvidia-smi
```

#### Linux/macOS:
```bash
# CPU和内存
top -p $(pgrep -f api_server)

# GPU
nvidia-smi
watch -n 1 nvidia-smi  # 实时监控
```

---

## 🛑 停止服务

### 前台运行
在运行服务器的终端按 `Ctrl+C`

### 后台运行

#### Windows:
```powershell
# 找到进程
Get-Process python

# 停止进程
Stop-Process -Name python
```

#### Linux/macOS:
```bash
# 找到进程ID
ps aux | grep api_server.py

# 停止进程
kill <PID>
```

---

## 📚 下一步

- 查看 [API文档](docs/API_DOCUMENTATION.md) 了解详细的API使用方法
- 查看 [模型信息](docs/MODEL_INFO.md) 了解模型性能和限制
- 查看 [示例代码](examples/) 学习更多使用方式

---

## 💬 获取帮助

如果遇到问题：
1. 检查本文档的常见问题部分
2. 查看终端输出的错误信息
3. 确保所有依赖都已正确安装
4. 联系项目维护者

---

## ✅ 验证清单

安装完成后，确认以下各项：

- [ ] Python版本 ≥ 3.8
- [ ] 所有依赖包已安装
- [ ] 模型文件存在 (`best_model.pth`)
- [ ] API服务器能正常启动
- [ ] 健康检查接口返回正常
- [ ] 能成功预测测试图像
- [ ] Web界面能正常访问（可选）

---

祝您使用愉快！🎉
