# 🧠 EmoCare - 智能情绪识别与健康管理系统

> **基于深度学习与大模型的心理健康数字助手** > **西安交通大学 大学生创新创业训练项目**


## 📖 项目简介 (Introduction)

**EmoCare** 是一套结合了**计算机视觉 (CV)** 与 **生成式人工智能 (GenAI)** 的情绪健康管理系统。

传统的心理评估往往依赖滞后的问卷调查，而本系统通过摄像头实时捕捉面部微表情，利用 **ResNet** 深度神经网络精准识别 6 种基本情绪（快乐、悲伤、愤怒、恐惧、厌恶、惊讶）。系统不仅能“看懂”你的表情，还能通过内置的**专家规则库**与**阿里云百炼大模型**，为您提供即时、温暖的心理调节建议。

### ✨ 核心功能
* **📷 多模态识别**：支持 **实时摄像头监测** 与 **静态图片上传** 分析。
* **📊 六维情绪量化**：精准计算并展示情绪的概率分布（雷达图），捕捉复杂混合情绪。
* **💡 智能健康顾问**：
    * **规则引擎**：基于心理学专家的“急救包”建议（如愤怒时的深呼吸引导）。
    * **LLM 深度分析**：集成阿里云大模型，生成富有共情力的综合心理分析报告。
* **⚡ 极速部署**：提供一键启动脚本，自动配置虚拟环境与依赖。

---

## 🛠️ 技术架构 (Tech Stack)

* **前端交互**：HTML5, JavaScript (Fetch API), Chart.js (数据可视化)
* **后端服务**：Python Flask (RESTful API)
* **核心算法**：
    * **模型**：ResNet50 / ResNet18 (PyTorch)
    * **预处理**：OpenCV (Haar Cascade Face Detection)
* **大模型集成**：Alibaba DashScope (通义千问 / OpenAI Compatible SDK)

---

## 📂 目录结构 (Directory Structure)

```text
EmoCare/
├── api/                    # 后端 API 代码
│   └── api_server.py       # Flask 服务器入口
├── models/                 # 模型定义与处理逻辑
│   ├── emotion_model.py    # PyTorch ResNet 模型结构
│   └── health_advisor.py   # 建议生成与 LLM 接口逻辑
├── examples/               # 前端界面示例
│   └── emotion_ui.html     # 用户交互界面
├── data/                   # 数据存储（日志/临时文件）
├── best_model.pth          # 训练好的模型权重文件
├── advice_rules.json       # 心理健康建议规则库
├── requirements.txt        # 项目依赖列表
├── start_api.bat           # ✅ Windows 一键启动脚本
└── README.md               # 项目说明文档



🚀 快速开始 (Quick Start)
1. 环境准备
操作系统：Windows 10/11

Python 版本：建议 Python 3.10 - 3.13

摄像头：用于实时监测功能（可选）

2. 启动项目
本项目实现了**“零配置启动”**。您无需手动安装依赖，只需运行脚本即可。

双击项目根目录下的 start_api.bat。

脚本会自动检测/创建虚拟环境 (.venv) 并安装所需库。

启动成功后，脚本会自动打开浏览器。

3. 访问界面
如果浏览器没有自动打开，请手动访问： 👉 http://localhost:8000/examples/emotion_ui.html

注意：请务必保留 URL 中的 /examples/ 路径，否则会导致页面资源加载失败。

⚙️ 配置说明 (Configuration)
启用 AI 大模型分析功能
系统默认使用本地规则库生成建议。如果您希望启用基于 阿里云通义千问 的深度分析功能，请按以下步骤配置 API Key：

打开 start_api.bat 文件（右键 -> 编辑）。

在文件顶部的 setlocal 下方添加一行：
set DASHSCOPE_API_KEY=sk-您的阿里云API密钥

保存并重启脚本。

⚠️ 常见问题 (FAQ)
Q: 打开网页显示 404 Not Found？ A: 请确认您的访问地址是 http://localhost:8000/examples/emotion_ui.html。如果不包含 examples，服务器将无法找到前端文件。

Q: 启动脚本闪退？ A: 请尝试右键点击 start_api.bat 选择“以管理员身份运行”。如果问题依旧，请在文件夹地址栏输入 cmd，然后手动运行脚本查看具体报错信息。

Q: 摄像头无法开启？ A: 请确保浏览器已获得摄像头权限（通常在地址栏左侧有个锁图标或摄像机图标，点击允许即可）。