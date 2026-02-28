# 表情识别模型部署包

这是一个完整的表情识别模型部署包，包含训练好的模型、API服务器和客户端代码。

## 📁 目录结构

```
deployment/
├── models/                      # 模型相关文件
│   ├── emotion_model.py        # 模型定义
│   └── README.md               # 模型说明
├── api/                        # API相关文件
│   ├── api_server.py          # Flask API 服务器
│   ├── api_client.py          # Python 客户端
│   └── README.md              # API说明
├── docs/                       # 文档
│   ├── API_DOCUMENTATION.md   # API详细文档
│   ├── DEPLOYMENT_GUIDE.md    # 部署指南
│   └── MODEL_INFO.md          # 模型信息
├── examples/                   # 示例代码
│   ├── example_usage.py       # Python 使用示例
│   ├── example_curl.sh        # cURL 使用示例
│   └── example_javascript.html # JavaScript 使用示例
├── requirements.txt           # Python 依赖
└── README.md                  # 本文件
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd deployment1
pip install -r requirements.txt
```

### 2. 复制模型文件

将训练好的模型文件复制到项目根目录：

```bash
# 从项目根目录执行
cp checkpoints_optimized/best_model.pth deployment1/
```

### 3. 启动 API 服务器

```bash
cd deployment1/api
python api_server.py
```

服务器将在 `http://localhost:5000` 启动

### 4. 测试 API

使用 Python 客户端：

```bash
cd deployment1/api
python api_client.py
```

或使用 cURL：

```bash
curl http://localhost:5000/health
```

## 📊 模型信息

### 模型架构
- **基础架构**: ResNet50
- **输入尺寸**: 224x224x3
- **输出类别**: 6 种表情
- **参数量**: ~25M

### 性能指标
- **验证集准确率**: 72.38%
- **宏平均F1**: 0.7122
- **加权平均F1**: 0.7218

### 支持的表情
1. Anger (愤怒)
2. Disgust (厌恶)
3. Fear (恐惧)
4. Happy (快乐)
5. Sad (悲伤)
6. Surprised (惊讶)

### 训练数据
- **FER2013**: 23,744 张
- **CK+**: 927 张
- **JAFFE**: 183 张
- **总计**: 24,854 张

### 优化技术
- ✅ ResNet50 架构
- ✅ 加权损失函数
- ✅ Focal Loss
- ✅ Mixup 数据增强
- ✅ OneCycleLR 学习率策略
- ✅ 梯度累积

## 🔌 API 端点

### GET /
获取 API 信息

### GET /health
健康检查

### GET /emotions
获取支持的表情列表

### POST /predict
单张图像预测

### POST /predict_batch
批量图像预测

详细文档请查看：[API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md)

## 💻 使用示例

### Python

```python
from api.api_client import EmotionRecognitionClient

# 创建客户端
client = EmotionRecognitionClient('http://localhost:5000')

# 预测图像
result = client.predict_from_file('path/to/image.jpg')
print(f"表情: {result['emotion']}, 置信度: {result['confidence']:.2%}")
```

### cURL

```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@path/to/image.jpg"
```

### JavaScript

```javascript
const formData = new FormData();
formData.append('image', fileInput.files[0]);

fetch('http://localhost:5000/predict', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

## 📦 部署选项

### 1. 本地部署
直接运行 Flask 服务器（开发环境）

### 2. Docker 部署
使用 Docker 容器化部署

### 3. 云部署
- AWS Lambda
- Google Cloud Run
- Azure Functions
- Heroku

详细部署指南：[DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)

## 🔧 配置

### 修改服务器端口

在 `api/api_server.py` 中修改：

```python
app.run(host='0.0.0.0', port=5000)  # 修改端口号
```

### 修改模型路径

在 `api/api_server.py` 中修改：

```python
model_path = 'path/to/your/model.pth'
```

### 启用 CUDA

模型会自动检测并使用 GPU（如果可用）

## 📈 性能优化

### 1. 批量预测
使用批量预测端点可以提高吞吐量

### 2. 模型量化
使用 PyTorch 量化可以减小模型大小和提高推理速度

### 3. ONNX 导出
导出为 ONNX 格式可以在多种平台上高效运行

### 4. TensorRT 加速
在 NVIDIA GPU 上使用 TensorRT 可以显著加速推理

## 🐛 故障排查

### 问题：端口被占用
解决方案：修改端口号或关闭占用端口的程序

### 问题：CUDA out of memory
解决方案：减小 batch size 或使用 CPU

### 问题：模型加载失败
解决方案：检查模型文件路径和权限

## 📝 许可证

本项目仅供学习和研究使用。

## 👥 贡献者

- 模型训练和优化
- API 开发和部署

## 📧 联系方式

如有问题或建议，请联系项目维护者。

## 🔄 更新日志

### v1.0.0 (2025-10-23)
- ✅ 初始版本发布
- ✅ 支持 6 种表情识别
- ✅ 提供 REST API
- ✅ 验证集准确率 72.38%

## 🎯 未来计划

- [ ] 增加更多表情类别
- [ ] 支持视频流实时识别
- [ ] 提供 Web UI 界面
- [ ] 支持多人脸识别
- [ ] 模型压缩和加速
- [ ] 移动端部署支持
