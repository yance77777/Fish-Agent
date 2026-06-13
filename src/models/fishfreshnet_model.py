"""
FishFreshNetV1 模型定义
完全匹配权重文件结构
"""
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class ChannelAttention(nn.Module):
    """通道注意力模块（使用Conv2d，无bias）"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
    
    def forward(self, x):
        # 全局平均池化
        y = x.mean(dim=[2, 3], keepdim=True)
        # 通道注意力
        y = torch.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
    
    def forward(self, x):
        # 沿通道维度计算均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接
        y = torch.cat([avg_out, max_out], dim=1)
        # 空间注意力
        y = torch.sigmoid(self.conv(y))
        return x * y


class FishFreshNetV1(nn.Module):
    """
    FishFreshNetV1 新鲜度分类网络
    结构完全匹配权重文件
    """
    
    def __init__(self, num_classes=3, pretrained=False):
        super(FishFreshNetV1, self).__init__()
        
        # 加载 EfficientNet-B0 的特征提取层
        if pretrained:
            backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            backbone = efficientnet_b0(weights=None)
        
        # 直接使用 features 和 avgpool，不包装在 backbone 下
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        
        # EfficientNet-B0 的最后一层特征维度是 1280
        feature_dim = 1280
        
        # 注意力模块（使用 ModuleDict 以匹配 'att.ca' 和 'att.sa' 键名）
        self.att = nn.ModuleDict({
            'ca': ChannelAttention(feature_dim),
            'sa': SpatialAttention()
        })
        
        # 分类头（使用 Sequential 以匹配 'classifier.1.weight' 键名）
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dim, num_classes)
        )
        
        # 用于 Grad-CAM
        self.feature_maps = None
    
    def forward(self, x):
        # 提取特征
        features = self.features(x)
        
        # 应用注意力（CBAM）
        features = self.att['ca'](features)
        features = self.att['sa'](features)
        
        # 保存注意力处理后的特征图用于 Grad-CAM。
        self.feature_maps = features
        
        # 全局平均池化
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        
        # 分类
        output = self.classifier(features)
        
        return output
    
    def get_feature_maps(self):
        """获取特征图（用于 Grad-CAM）"""
        return self.feature_maps


def load_model(model_path: str, device: str = 'cpu'):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型权重文件路径
        device: 设备 ('cpu' 或 'cuda')
    
    Returns:
        加载好的模型
    """
    # 创建模型实例
    model = FishFreshNetV1(num_classes=3, pretrained=False)
    
    # 加载权重（安全模式优先）
    try:
        # 优先使用安全模式加载（PyTorch 2.0+ 推荐）
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except Exception:
        # 如果权重文件包含非张量对象，回退到兼容模式并记录警告
        import logging
        logging.warning(f"模型文件 {model_path} 需要 weights_only=False 加载，请确保文件来源可信")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 检查 checkpoint 类型并加载
    if isinstance(checkpoint, dict):
        # 可能是包含额外信息的 checkpoint
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 尝试直接作为 state_dict 加载
            model.load_state_dict(checkpoint)
    elif isinstance(checkpoint, nn.Module):
        # 如果保存的是完整模型对象
        model = checkpoint
    else:
        # 尝试作为 state_dict 加载
        model.load_state_dict(checkpoint)
    
    model.eval()
    model.to(device)
    
    return model


if __name__ == "__main__":
    import os
    
    # 测试模型加载
    model_path = "src/storage/fishfreshnet_v1.pth"
    if os.path.exists(model_path):
        print(f"✅ 找到模型文件: {model_path}")
        print("🔄 正在加载模型...")
        
        try:
            model = load_model(model_path)
            print("✅ 模型加载成功！")
            
            # 测试推理
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = model(dummy_input)
            
            print(f"✅ 推理测试成功！输出形状: {output.shape}")
            
            # 测试分类
            probs = torch.softmax(output, dim=1)
            classes = ["高度新鲜", "新鲜", "不新鲜"]
            pred_class = classes[torch.argmax(probs).item()]
            print(f"✅ 预测类别: {pred_class}")
            print(f"✅ 置信度分布: {probs[0].tolist()}")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"❌ 模型文件不存在: {model_path}")
