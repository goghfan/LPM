# -*- coding: utf-8 -*-

"""
本文件定义了LPM框架的核心模块之一：可微特征投影仪 P。

它的作用是根据给定的相机位姿，将一个3D神经场景 (由SceneEncoderHashGrid表示)
渲染成一张2D的高维特征图。
"""

import torch
import torch.nn as nn

# 从我们项目中的其他模块导入必要的组件
from models.EfficentNet import SceneEncoderHashGrid
from utils import get_rays
import tinycudann as tcnn
class DifferentiableProjector(nn.Module):
    """
    可微特征投影仪 P。
    
    本模块遵循NeRF中的体积渲染方程，但渲染的不是颜色，而是高维特征。
    它本身没有可训练的参数，但其计算过程完全可微。
    """
    def __init__(
        self,
        height: int,
        width: int,
        n_samples: int,
        focal_length: float,
        near: float,
        far: float
    ):
        """
        初始化投影仪。

        Args:
            height (int): 输出特征图的高度。
            width (int): 输出特征图的宽度。
            n_samples (int): 沿每条射线采样的点数。
            focal_length (float): 虚拟相机的焦距。
            near (float): 射线采样的近裁剪平面。
            far (float): 射线采样的远裁剪平面。
        """
        super().__init__()
        self.height = height
        self.width = width
        self.n_samples = n_samples
        self.focal_length = focal_length
        self.near = near
        self.far = far

    def forward(self, scene_encoder: SceneEncoderHashGrid, pose: torch.Tensor) -> torch.Tensor:
        """
        执行一次前向传播，即一次完整的特征投影/渲染。

        Args:
            scene_encoder (SceneEncoderHashGrid): 经过训练的3D场景编码器 F_θ。
            pose (torch.Tensor): 一个4x4的"相机到世界"变换矩阵，代表相机位姿 π。
                                     可以是单个位姿 [4, 4] 或一个批次的位姿 [B, 4, 4]。

        Returns:
            torch.Tensor: 渲染出的2D特征图，形状为 [B, C, H, W]，
                          其中C是场景编码器输出的特征维度。
        """
        # 如果输入是单个位姿，为其增加一个批次维度
        if pose.dim() == 2:
            pose = pose.unsqueeze(0)
        
        batch_size = pose.shape[0]
        device = pose.device

        # 1. 生成射线
        #    注意：我们假设所有位姿共享相同的内参(H, W, focal)
        rays_o_all, rays_d_all = get_rays(self.height, self.width, self.focal_length, pose[0])
        # 将射线展平以便批处理
        rays_o = rays_o_all.view(-1, 3) # [H*W, 3]
        rays_d = rays_d_all.view(-1, 3) # [H*W, 3]

        # 2. 沿射线采样3D点
        z_vals = torch.linspace(self.near, self.far, self.n_samples, device=device).view(1, -1) # [1, N_samples]
        pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1) # [H*W, N_samples, 3]
        
        # 3. 查询场景编码器，获取每个采样点的 (ρ, f)
        #    为了节省显存，可以分块(chunk)查询
        #    这里为简化，一次性查询
        rho, f = scene_encoder(pts.view(-1, 3)) # rho:[H*W*N, 1], f:[H*W*N, C]
        
        # 将输出重塑为 [H*W, N_samples, C+1] 的形式
        rho = rho.view(self.height * self.width, self.n_samples, 1)
        f = f.view(self.height * self.width, self.n_samples, -1)
        
        feature_dim = f.shape[-1]
        
        # 4. Alpha合成: 计算每个采样点的贡献权重 w_k
        delta_t = (self.far - self.near) / self.n_samples
        alpha = 1. - torch.exp(-rho * delta_t) # [H*W, N_samples, 1]

        # 计算透射率 T_k
        # T_k = exp(-Σ(ρ_j*Δt_j)) for j < k
        # 为了数值稳定性，我们使用累积乘积
        # 1. - alpha 是每个小段的“透明度”
        # cumprod 在最后一个维度上计算累积乘积
        transmittance = torch.cumprod(1. - alpha + 1e-10, dim=1) # [H*W, N_samples, 1]
        
        # 将T向后移动一位，T_1应该是1.0
        transmittance = torch.roll(transmittance, 1, dims=1)
        transmittance[:, 0, :] = 1.0

        weights = transmittance * alpha # [H*W, N_samples, 1]

        # 5. 特征加权求和
        #    使用计算出的权重，对每个采样点的特征 f_k 进行加权求和
        projected_features = torch.sum(weights * f, dim=1) # [H*W, C]

        # 6. 将展平的特征图重塑为最终的2D特征图形状
        # [H*W, C] -> [H, W, C] -> [1, C, H, W] (假设batch_size=1)
        projected_features = projected_features.view(self.height, self.width, feature_dim)
        projected_features = projected_features.permute(2, 0, 1).unsqueeze(0) # C, H, W -> B, C, H, W

        return projected_features

# --- 模块测试入口 ---
if __name__ == '__main__':
    print("--- DifferentiableProjector 模块测试 ---")
    
    # 检查是否有可用的CUDA设备和tiny-cuda-nn
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"CUDA 和 tiny-cuda-nn 可用，开始测试。设备: {device}")
        
        # 1. 初始化所有需要的组件
        FEATURE_DIM = 32
        HEIGHT, WIDTH = 64, 64
        N_SAMPLES = 128
        
        # 初始化3D场景编码器
        scene_encoder = SceneEncoderHashGrid(feature_dim=FEATURE_DIM).to(device)
        
        # 初始化可微投影仪
        projector = DifferentiableProjector(
            height=HEIGHT,
            width=WIDTH,
            n_samples=N_SAMPLES,
            focal_length=300.0,
            near=1.5,
            far=3.5,
        ).to(device)

        # 2. 创建一个虚拟的相机位姿
        #    这是一个从Z轴正方向2.5个单位处，看向原点的标准相机位姿
        pose_matrix = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 2.5],
            [0, 0, 0, 1]
        ], dtype=torch.float32, device=device)
        
        print(f"\n投影仪已初始化，输出尺寸: ({HEIGHT}, {WIDTH})")
        print(f"场景编码器已初始化，特征维度: {FEATURE_DIM}")
        print(f"输入位姿形状: {pose_matrix.shape}")
        
        # 3. 执行前向传播
        #    在测试时，我们不需要计算梯度
        with torch.no_grad():
            output_feature_map = projector(scene_encoder, pose_matrix)
            
        # 4. 验证输出形状
        print("\n--- 模块输出验证 ---")
        print(f"输出特征图形状: {output_feature_map.shape}")
        
        expected_shape = (1, FEATURE_DIM, HEIGHT, WIDTH)
        assert output_feature_map.shape == expected_shape, f"输出形状错误！应为 {expected_shape}"
        
        print("\n测试成功！DifferentiableProjector 可以正确地进行前向传播并输出正确形状的特征图。")
        
    else:
        print("\n测试跳过：需要CUDA和 tiny-cuda-nn 才能运行此测试。")
# ```

# ### **如何使用**

# 现在，我们LPM框架的所有核心组件都已经代码化了。在您的主训练脚本`train.py`中，您可以像搭乐高一样将它们组合起来：

# ```python
# # 伪代码：train.py

# from models import SceneEncoderHashGrid, ImageEncoder
# from projector import DifferentiableProjector
# from utils import get_rays
# from losses import TripletLoss

# # 1. 初始化所有模块
# scene_encoder = SceneEncoderHashGrid(...).to(device)
# image_encoder = ImageEncoder(...).to(device)
# projector = DifferentiableProjector(...).to(device)
# triplet_loss_fn = TripletLoss(...)

# # 2. 启动训练循环
# for i, (I_gt, pi_gt) in enumerate(dataloader):
#     # ...
#     # 2D路径
#     F_target = image_encoder(I_gt)

#     # 3D路径
#     F_proj_pos = projector(scene_encoder, pi_gt)
#     F_proj_neg = projector(scene_encoder, pi_wrong)

#     # 计算损失
#     loss = triplet_loss_fn(F_target, F_proj_pos, F_proj_neg)
    
#     # 反向传播...
#     # ...