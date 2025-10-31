# models/scene_encoder_mlp.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoder(nn.Module):
    """
    NeRF 风格的位置编码器。
    对输入坐标进行傅里叶特征映射，增加高频信息。
    """
    def __init__(self, input_dims: int, num_freqs: int, include_input: bool = True):
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
        # 将 freq_bands 注册为 buffer，它不会被训练，但会在 GPU 上
        self.register_buffer('freq_bands', freq_bands_)

        # 计算输出维度
        self.output_dims = 0
        if self.include_input:
            self.output_dims += self.input_dims
        self.output_dims += self.input_dims * self.num_freqs * 2 # sin/cos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        # [..., C, N_freqs] -> [..., C, N_freqs*2] (拼接 sin 和 cos)
        encoded = torch.cat([torch.sin(scaled_inputs), torch.cos(scaled_inputs)], dim=-1)
        # [..., C, N_freqs*2] -> [..., C*N_freqs*2] (展平 C 和 N_freqs*2 维度)
        encoded = encoded.flatten(start_dim=-2)
        outputs.append(encoded)
        
        return torch.cat(outputs, dim=-1)


class SceneEncoderMLP(nn.Module):
    """
    纯 PyTorch 实现的 3D 场景编码器 (位置编码 + MLP)。
    将 3D 坐标映射到 (密度, 特征) 对，并包含 NeRF 风格的跳跃连接。
    """
    def __init__(
        self,
        bounding_box: tuple = (-1.0, 1.0),
        feature_dim: int = 1536,
        pos_encode_freqs: int = 10,  # 位置编码的频率数量 (NeRF 常用 10)
        num_layers: int = 8,        # MLP 主体的线性层总数 (NeRF 常用 8)
        hidden_dim: int = 256,      # MLP 的隐藏层维度 (NeRF 常用 256)
        skip_connection_layer: int = 4 # 跳跃连接发生在哪一层之后 (NeRF 常用 4)
    ):
        super().__init__()
        
        # 註册边界框，用于可能的输入检查或归一化
        self.register_buffer('bounding_box', torch.tensor([bounding_box[0], bounding_box[1]], dtype=torch.float32))
        
        # 1. 位置编码器
        # 输入维度为 3 (x,y,z)，频率数量由 pos_encode_freqs 决定，并包含原始输入
        self.pos_encoder = PositionalEncoder(input_dims=3, num_freqs=pos_encode_freqs, include_input=True)
        mlp_input_dim = self.pos_encoder.output_dims # e.g., 3 + 3*10*2 = 63

        # 确保 num_layers 至少为 skip_connection_layer，否则跳跃连接没有意义
        if num_layers < skip_connection_layer:
            raise ValueError(f"num_layers ({num_layers}) must be >= skip_connection_layer ({skip_connection_layer})")

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skip_connection_layer = skip_connection_layer
        
        # 2. MLP 主体 (使用 ModuleList 存储层，便于手动控制跳跃连接)
        self.layers = nn.ModuleList()
        current_input_dim = mlp_input_dim

        for i in range(num_layers):
            if i == skip_connection_layer:
                # 跳跃连接发生后，输入维度会增加
                current_input_dim = hidden_dim + mlp_input_dim
            
            # 第一个线性层从位置编码维度开始，其余的从 hidden_dim 或 (hidden_dim + skip_input_dim) 开始
            layer_input_dim = mlp_input_dim if i == 0 else current_input_dim
            
            self.layers.append(nn.Linear(layer_input_dim, hidden_dim))
            current_input_dim = hidden_dim # 下一层的输入维度恢复为隐藏维度 (除非有新的跳跃连接)

        # 3. 输出头
        # 密度头接收 MLP 主体的最终输出 (hidden_dim)
        self.density_head = nn.Linear(hidden_dim, 1)
        # 特征头接收 MLP 主体的最终输出 (hidden_dim)，输出维度为 feature_dim
        self.feature_head = nn.Linear(hidden_dim, feature_dim)

        print(f"SceneEncoderMLP initialized:")
        print(f"  Positional Encoding Freqs: {pos_encode_freqs}")
        print(f"  Pos Encoded Dim: {mlp_input_dim}")
        print(f"  MLP Depth (Linear layers): {num_layers}, Hidden Dim: {hidden_dim}")
        print(f"  Skip Connection after layer: {skip_connection_layer}")
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
        # 注意: 这里假设输入 x 已经是 NeRF 期望的归一化范围 (通常是 [-1, 1] 或 [0, 1])
        # 如果不是，可以进行归一化。原始 NeRF 通常直接使用输入的归一化坐标。
        # x_normalized = (x - self.bounding_box[0]) / (self.bounding_box[1] - self.bounding_box[0]) * 2.0 - 1.0
        # For now, let's assume x is already appropriately scaled.

        # 1. 进行位置编码
        encoded_x = self.pos_encoder(x) # [N, encoded_dim]
        
        # 2. 通过 MLP 主体，手动处理跳跃连接
        h = encoded_x
        for i, layer in enumerate(self.layers):
            if i == self.skip_connection_layer:
                # 执行跳跃连接：将原始的编码输入 h_pos 拼接到当前特征
                h = torch.cat([h, encoded_x], dim=-1)
            
            h = F.relu(layer(h)) # 每个线性层后接 ReLU 激活

        # 3. 通过输出头
        density_output = self.density_head(h) # [N, 1]
        f = self.feature_head(h)              # [N, feature_dim]
        
        # 4. 密度激活
        # 使用 ReLU 确保密度非负。原始 NeRF 也使用 ReLU。
        rho = F.relu(density_output) # 或者 F.softplus(density_output) 更平滑
        
        return rho, f

# --- 模型测试入口 (可选) ---
if __name__ == "__main__":
    print("--- SceneEncoderMLP 模型测试 ---")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 模型参数 (与原始 NeRF 常用参数保持一致)
    FEATURE_DIM = 1536 # 示例，可以根据你的 ImageEncoder 实际输出调整
    BBOX = (-1.0, 1.0)
    POS_ENCODE_FREQS = 10 # 3D 位置编码的频率数
    MLP_NUM_LAYERS = 8    # MLP 主体的线性层总数
    MLP_HIDDEN_DIM = 256  # MLP 隐藏层维度
    SKIP_LAYER = 4        # 跳跃连接发生在哪一层之后

    # 创建模型实例
    model = SceneEncoderMLP(
        bounding_box=BBOX,
        feature_dim=FEATURE_DIM,
        pos_encode_freqs=POS_ENCODE_FREQS,
        num_layers=MLP_NUM_LAYERS,
        hidden_dim=MLP_HIDDEN_DIM,
        skip_connection_layer=SKIP_LAYER
    ).to(device)
    
    # 创建一批随机的3D坐标作为输入 ([-1, 1] 范围)
    num_points = 1024
    # 确保输入在 bounding_box 范围内，这里假设是 [-1, 1]
    input_coords = (torch.rand(num_points, 3, device=device) * 2.0 - 1.0)
    
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
    
    print("\n模型结构 (精简表示，因为 ModuleList 不会像 Sequential 那样详细展开):")
    print(model)
    
    print("\n测试成功！SceneEncoderMLP 可以正确地进行前向传播，并包含跳跃连接。")
