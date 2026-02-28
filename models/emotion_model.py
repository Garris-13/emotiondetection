"""
表情识别模型定义
用于部署和推理
"""

import torch
import torch.nn as nn
import torchvision.models as models


class EmotionRecognitionModel(nn.Module):
    """表情识别模型"""

    # 将默认的 model_name 修改为 'resnet18'，以匹配您的新模型
    def __init__(self, num_classes=7, model_name='resnet18', pretrained=False):
        super(EmotionRecognitionModel, self).__init__()

        self.model_name = model_name

        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)

        elif model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)

        else:
            raise ValueError(f"不支持的模型: {model_name}")

    def forward(self, x):
        return self.backbone(x)


def load_model(model_path, model_name='resnet18', num_classes=7, device='cpu'):
    """加载训练好的模型"""
    model = EmotionRecognitionModel(num_classes=num_classes, model_name=model_name, pretrained=False)

    try:
        # 1. 加载字典
        state_dict = torch.load(model_path, map_location=device)

        # 2. 【核心修复】自动修复键名不匹配的问题
        new_state_dict = {}
        for k, v in state_dict.items():
            # 如果保存的键名没有 backbone. 前缀，就帮它加上
            if not k.startswith('backbone.'):
                new_state_dict[f'backbone.{k}'] = v
            else:
                new_state_dict[k] = v

        # 3. 加载修复后的字典
        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"❌ 加载模型权重失败: {e}")
        raise e