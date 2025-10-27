# -*- coding: utf-8 -*-

"""
本文件定义了LPM框架的核心模型组件。

1. SceneEncoderHashGrid:
   - 架构A：基于场景的隐式表示。
   - 输入3D坐标，输出该点的(密度, 特征)。

2. ImageEncoder:
   - 2D图像编码器 E_ϕ。
   - 基于预训练的EfficientNet，用于从2D图像中提取高维特征图。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# 尝试导入 tinycudann
try:
    import tinycudann as tcnn
except ImportError:
    tcnn = None # 允许在没有tcnn的环境下，仍能使用下面的ImageEncoder

# ##############################################################################
# 1. 3D场景编码器 (SceneEncoderHashGrid)
# ##############################################################################

class SceneEncoderHashGrid(nn.Module):
    """
    一个3D场景编码器，将3D坐标映射到解耦的(密度, 特征)对。
    """
    def __init__(
        self,
        bounding_box: tuple = (-1.0, 1.0),
        feature_dim: int = 32,
        hashgrid_log2_hashmap_size: int = 19,
        hashgrid_n_levels: int = 16,
        hashgrid_n_features_per_level: int = 2,
        hashgrid_base_resolution: int = 16,
        hashgrid_per_level_scale: float = 2.0,
        decoder_n_hidden_layers: int = 1,
        decoder_n_neurons: int = 64
    ):
        super().__init__()
        
        if tcnn is None:
            raise ImportError("tiny-cuda-nn未能成功导入，无法构建 SceneEncoderHashGrid 模型。")

        self.register_buffer('bounding_box', torch.tensor([bounding_box[0], bounding_box[1]]))
        
        self.grid_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid", "n_levels": hashgrid_n_levels, "n_features_per_level": hashgrid_n_features_per_level,
                "log2_hashmap_size": hashgrid_log2_hashmap_size, "base_resolution": hashgrid_base_resolution,
                "per_level_scale": hashgrid_per_level_scale,
            },
        )
        
        self.decoder_mlp = tcnn.Network(
            n_input_dims=self.grid_encoder.n_output_dims,
            n_output_dims=decoder_n_neurons,
            network_config={
                "otype": "FullyFusedMLP", "activation": "ReLU", "output_activation": "None",
                "n_neurons": decoder_n_neurons, "n_hidden_layers": decoder_n_hidden_layers,
            },
        )

        self.density_head = nn.Linear(decoder_n_neurons, 1)
        self.feature_head = nn.Linear(decoder_n_neurons, feature_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_x = (x - self.bounding_box[0]) / (self.bounding_box[1] - self.bounding_box[0])
        grid_features = self.grid_encoder(normalized_x)
        hidden_features = self.decoder_mlp(grid_features)
        
        # --- 错误修正 ---
        # 将tiny-cuda-nn输出的半精度(float16)张量，转换为全精度(float32)
        # 以匹配后续 nn.Linear 层的期望数据类型。
        hidden_features_float = hidden_features.float()
        
        density_output = self.density_head(hidden_features_float)
        f = self.feature_head(hidden_features_float)
        # -----------------
        
        rho = F.softplus(density_output)
        
        return rho, f

# ##############################################################################
# 2. 2D图像编码器 (ImageEncoder)
# ##############################################################################

class ImageEncoder(nn.Module):
    """
    一个2D图像编码器 E_ϕ，用于从2D图像中提取高维特征图。
    本实现基于在ImageNet上预训练的EfficientNet-B3。
    """
    def __init__(self, pretrained: bool = True, output_dim: int = 32):
        super().__init__()
        
        if pretrained:
            weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
            efficientnet = models.efficientnet_b3(weights=weights)
        else:
            efficientnet = models.efficientnet_b3(weights=None)
            
        self.feature_extractor = efficientnet.features
        
        # 适配器 (将 1 通道灰度图变为 3 通道)
        self.input_adapter = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.input_adapter.weight.data = torch.tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]]])
        self.input_adapter.weight.requires_grad = False

        # --- [!!! 关键修改 2 !!!] ---
        # 添加一个 1x1 卷积层，将主干的输出通道 (例如 1536) 映射到
        # 我们需要的输出维度 (例如 32)，以匹配 Projector。
        
        # 自动获取 EfficientNet-B3 主干的输出通道数 (即 1536)
        in_channels = efficientnet.classifier[1].in_features
        
        self.channel_mapper = nn.Conv2d(in_channels, output_dim, kernel_size=1)
        # --- [修改结束] ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 1, 256, 256) -> (B, 3, 256, 256)
        x_3channel = self.input_adapter(x)
        
        # (B, 3, 256, 256) -> (B, 1536, 8, 8)
        feature_map = self.feature_extractor(x_3channel)
        
        # --- [!!! 关键修改 3 !!!] ---
        # (B, 1536, 8, 8) -> (B, 32, 8, 8)
        mapped_feature_map = self.channel_mapper(feature_map)
        # --- [修改结束] ---
        
        return mapped_feature_map

# --- 模型测试入口 ---
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"--- 模型测试 ---")
    print(f"使用设备: {device}")
    
    # --- 测试 2D ImageEncoder ---
    print("\n--- 2D ImageEncoder (EfficientNet-B3) 测试 ---")
    
    image_encoder = ImageEncoder(pretrained=True).to(device)
    image_encoder.eval() 

    batch_size = 4
    image_height = 256
    image_width = 256
    dummy_input_image = torch.randn(batch_size, 1, image_height, image_width).to(device)

    print(f"输入图像形状: {dummy_input_image.shape}")

    with torch.no_grad():
        output_feature_map = image_encoder(dummy_input_image)

    print(f"输出特征图形状: {output_feature_map.shape}")
    print("ImageEncoder 测试成功！")
    
    # --- 测试 3D SceneEncoderHashGrid (如果可用) ---
    print("\n--- 3D SceneEncoderHashGrid 测试 ---")
    if torch.cuda.is_available() and tcnn is not None:
        model_A = SceneEncoderHashGrid(feature_dim=32).to(device)
        input_A = torch.rand(1024, 3, device=device) * 2 - 1
        rho, f = model_A(input_A)
        print(f"输入形状 (3D坐标): {input_A.shape}")
        print(f"输出形状 (密度, 特征): {rho.shape}, {f.shape}")
        print("SceneEncoderHashGrid 测试成功！")
    else:
        print("跳过测试：需要CUDA和tiny-cuda-nn。")