# models/EfficentNet.py (或 models/models.py)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import tinycudann as tcnn # 假设 tcnn 已导入

# ##############################################################################
# 1. 3D场景编码器 (SceneEncoderHashGrid)
# ##############################################################################

class SceneEncoderHashGrid(nn.Module):
    """
    一个3D场景编码器，将3D坐标映射到解耦的(密度, 特征)对。
    [已修正] 包含了 float() 转换以修复 Half/Float 错误。
    """
    def __init__(
        self,
        bounding_box: tuple = (-1.0, 1.0),
        feature_dim: int = 32, # [!!! 注意 !!!] train.py 中会传入 1536
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
        # [!!! 注意 !!!] 输出维度将由传入的 feature_dim (1536) 决定
        self.feature_head = nn.Linear(decoder_n_neurons, feature_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_x = (x - self.bounding_box[0]) / (self.bounding_box[1] - self.bounding_box[0])
        grid_features = self.grid_encoder(normalized_x)
        hidden_features = self.decoder_mlp(grid_features)
        
        # --- 修正 Half/Float 错误 ---
        hidden_features_float = hidden_features.float()
        # ---------------------------
        
        density_output = self.density_head(hidden_features_float)
        f = self.feature_head(hidden_features_float)
        
        rho = F.softplus(density_output)
        
        return rho, f

# ##############################################################################
# 2. 2D图像编码器 (ImageEncoder)
# ##############################################################################

class ImageEncoder(nn.Module):
    """
    一个2D图像编码器 E_ϕ，用于从2D图像中提取高维特征图。
    [已修改] 移除了 channel_mapper，直接输出 EfficientNet 的特征。
    """
    # [!!! 修改 1 !!!] 移除 output_dim 参数
    def __init__(self, pretrained: bool = True): 
        super().__init__()
        
        if pretrained:
            weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
            efficientnet = models.efficientnet_b3(weights=weights)
        else:
            efficientnet = models.efficientnet_b3(weights=None)
            
        # 使用 EfficientNet 的特征提取器部分
        self.feature_extractor = efficientnet.features
        
        # 适配器 (将 1 通道灰度图变为 3 通道)
        self.input_adapter = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.input_adapter.weight.data = torch.tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]]])
        self.input_adapter.weight.requires_grad = False

        # --- [!!! 修改 2 !!!] ---
        # 移除 self.channel_mapper
        # --- [修改结束] ---

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 1, 256, 256) -> (B, 3, 256, 256)
        x_3channel = self.input_adapter(x)
        
        # (B, 3, 256, 256) -> (B, 1536, 8, 8)
        feature_map = self.feature_extractor(x_3channel)
        
        # --- [!!! 修改 3 !!!] ---
        # 直接返回主干网络的输出
        return feature_map
        # --- [修改结束] ---

# ... (测试代码部分可以保留或移除) ...