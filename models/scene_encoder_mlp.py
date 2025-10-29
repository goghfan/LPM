# models/scene_encoder_mlp.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoder(nn.Module):
    """
    NeRF 风格的位置编码器。
    """
    def __init__(self, input_dims, num_freqs, include_input=True):
        super().__init__()
        self.input_dims = input_dims
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.log_sampling = True # 通常设置为 True

        # 频率带 [1, 2, 4, ..., 2^(L-1)] * pi
        if self.log_sampling:
            freq_bands_ = 2.**torch.linspace(0., num_freqs - 1, num_freqs) * torch.pi
        else:
            freq_bands_ = torch.linspace(1., 2.**(num_freqs - 1), num_freqs) * torch.pi
        self.register_buffer('freq_bands', freq_bands_)
        # 计算输出维度
        self.output_dims = 0
        if self.include_input:
            self.output_dims += self.input_dims
        self.output_dims += self.input_dims * self.num_freqs * 2 # sin/cos

    def forward(self, x):
        """
        对输入坐标进行编码。
        Args:
            x: 输入张量，形状 [..., input_dims]
        Returns:
            编码后的张量，形状 [..., output_dims]
        """
        outputs = []
        if self.include_input:
            outputs.append(x)
        
        # x shape [..., C] -> [..., C, 1] -> [..., C, N_freqs]
        scaled_inputs = x.unsqueeze(-1) * self.freq_bands
        # [..., C, N_freqs] -> [..., C, N_freqs*2]
        encoded = torch.cat([torch.sin(scaled_inputs), torch.cos(scaled_inputs)], dim=-1)
        # [..., C, N_freqs*2] -> [..., C*N_freqs*2]
        encoded = encoded.flatten(start_dim=-2)
        outputs.append(encoded)
        
        return torch.cat(outputs, dim=-1)


class SceneEncoderMLP(nn.Module):
    """
    纯 PyTorch 实现的 3D 场景编码器 (位置编码 + MLP)。
    将 3D 坐标映射到 (密度, 特征) 对。
    """
    def __init__(
        self,
        bounding_box: tuple = (-1.0, 1.0),
        feature_dim: int = 1536,
        pos_encode_freqs: int = 10,  # 位置编码的频率数量 (NeRF 常用 10)
        num_layers: int = 8,        # MLP 的层数
        hidden_dim: int = 256       # MLP 的隐藏层维度
    ):
        super().__init__()
        
        # 註册边界框，用于可能的输入检查或归一化（尽管 MLP 对输入范围不如 HashGrid 敏感）
        self.register_buffer('bounding_box', torch.tensor([bounding_box[0], bounding_box[1]]))
        
        # 1. 位置编码器
        self.pos_encoder = PositionalEncoder(input_dims=3, num_freqs=pos_encode_freqs, include_input=True)
        mlp_input_dim = self.pos_encoder.output_dims
        
        # 2. MLP 主体
        layers = []
        layers.append(nn.Linear(mlp_input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers - 2): # 中间层
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            # 可以考虑加入 Skip Connection (类似 NeRF) 来提升效果，但会更复杂
            
        self.mlp_body = nn.Sequential(*layers)
        
        # 3. 输出头
        self.density_head = nn.Linear(hidden_dim, 1)
        self.feature_head = nn.Linear(hidden_dim, feature_dim)

        print(f"SceneEncoderMLP initialized:")
        print(f"  Positional Encoding Freqs: {pos_encode_freqs}")
        print(f"  Pos Encoded Dim: {mlp_input_dim}")
        print(f"  MLP Depth: {num_layers}, Hidden Dim: {hidden_dim}")
        print(f"  Output Feature Dim: {feature_dim}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        模型的前向传播。

        Args:
            x (torch.Tensor): 输入的三维坐标张量，形状为 [N, 3]。
                              建议输入值在 [-1, 1] 范围内。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 一个包含 (rho, f) 的元组。
                - rho: 密度张量，形状为 [N, 1]。
                - f: 特征张量，形状为 [N, feature_dim]。
        """
        # (可选) 检查输入范围或进行归一化到 [-1, 1]
        x_normalized = (x - self.bounding_box[0]) / (self.bounding_box[1] - self.bounding_box[0]) * 2.0 - 1.0

        # 1. 进行位置编码
        encoded_x = self.pos_encoder(x) # [N, encoded_dim]
        
        # 2. 通过 MLP 主体
        hidden_features = self.mlp_body(encoded_x) # [N, hidden_dim]
        
        # 3. 通过输出头
        density_output = self.density_head(hidden_features) # [N, 1]
        f = self.feature_head(hidden_features)              # [N, feature_dim]
        
        # 4. 密度激活
        # 使用 ReLU 也可以，Softplus 更平滑且严格为正
        rho = F.relu(density_output) # 或者 F.softplus(density_output)
        
        return rho, f

# --- 模型测试入口 (可选) ---
if __name__ == "__main__":
    print("--- SceneEncoderMLP 模型测试 ---")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 模型参数
    FEATURE_DIM = 1536
    BBOX = (-1.0, 1.0)
    
    # 创建模型实例
    model = SceneEncoderMLP(bounding_box=BBOX, feature_dim=FEATURE_DIM, pos_encode_freqs=10).to(device)
    
    # 创建一批随机的3D坐标作为输入 ([-1, 1] 范围)
    num_points = 1024
    input_coords = (torch.rand(num_points, 3, device=device) * 2.0 - 1.0) * BBOX[1] # 确保在 bbox 内
    
    print(f"\n模型已创建在: {device}")
    print(f"输入坐标形状: {input_coords.shape}")
    
    # 执行前向传播
    rho_output, f_output = model(input_coords)
    
    # 打印输出类型和形状以验证
    print("\n--- 模型输出验证 ---")
    print(f"密度 (rho) 输出类型: {rho_output.dtype}, 形状: {rho_output.shape}")
    print(f"特征 (f) 输出类型: {f_output.dtype}, 形状: {f_output.shape}")
    
    # 检查形状是否符合预期
    assert rho_output.shape == (num_points, 1), "密度输出形状错误！"
    assert f_output.shape == (num_points, FEATURE_DIM), "特征输出形状错误！"
    # 检查数据类型是否为 float32
    assert rho_output.dtype == torch.float32, "密度输出类型错误！"
    assert f_output.dtype == torch.float32, "特征输出类型错误！"
    
    print("\n模型结构:")
    print(model)
    
    print("\n测试成功！SceneEncoderMLP 可以正确地进行前向传播。")