# -*- coding: utf-8 -*-

"""
本文件是LPM框架的在线配准脚本。

它负责执行术中在线配准阶段，其主要流程包括：
1. 加载预训练好的 3D 场景编码器 (F_θ) 和 2D 图像编码器 (E_ϕ)。
2. 冻结两个编码器的权重。
3. 加载目标 X 光图像 (I_live)。
4. 使用 E_ϕ 提取目标特征图 (F_target)。
5. 初始化一个可优化的位姿 (π)。
6. 启动快速迭代优化循环：
   a. 使用可微特征投影仪 P 从 F_θ 渲染当前投影特征 (F_proj)。
   b. 计算 F_proj 和 F_target 之间的 L2 损失。
   c. 反向传播，仅更新位姿 π。
7. 输出最终优化后的位姿 π*。
8. (可选) 使用最终位姿和 F_θ 中的 ρ 生成可视化DRR。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import numpy as np
import os
import imageio # 用于加载和保存图像
from tqdm import tqdm
import argparse # 用于从命令行接收参数

# 从我们项目中的其他模块导入必要的组件
from models import SceneEncoderHashGrid, ImageEncoder
from projector import DifferentiableProjector
from utils import get_rays # 确保utils.py在可导入的路径下
# 假设有一个预处理函数可以加载和处理单张X光图像
# from dataloader import preprocess_single_xray # 需要您根据实际情况实现或调整
# 假设有一个函数可以将6D向量转换为4x4矩阵 (如果使用6D表示位姿)
# from pose_utils import vec_to_matrix, matrix_to_vec # 需要您根据实际情况实现

# 临时的图像预处理函数 (需要根据您训练时使用的预处理进行调整)
def preprocess_single_xray(image_path: str, target_height: int, target_width: int, device: torch.device) -> torch.Tensor:
    """
    加载单张X光图像并进行预处理。
    这是一个示例实现，您需要确保它与训练 E_ϕ 时使用的预处理一致。
    """
    img = imageio.imread(image_path).astype(np.float32)
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0) # [H, W] -> [1, 1, H, W]
    
    # 示例预处理：调整尺寸，归一化到[0,1] (这可能不足够，需要匹配训练时的预处理)
    # 注意：这里的预处理应尽可能复现训练数据加载器中的逻辑
    # 比如是否需要对数变换 log(I0) - log(I)？
    img_tensor = nn.functional.interpolate(img_tensor, size=(target_height, target_width), mode='bilinear', align_corners=False)
    min_val = torch.min(img_tensor)
    max_val = torch.max(img_tensor)
    if max_val > min_val:
        img_tensor = (img_tensor - min_val) / (max_val - min_val)
        
    return img_tensor.to(device)

# --- 定义6D位姿参数化 (一个简单的实现) ---
# 在实际应用中，使用李代数 (se(3)) 或四元数+平移通常更稳定
class PoseParameterization(nn.Module):
    """
    将一个6D向量参数化为一个4x4 SE(3)矩阵。
    这是一个简化的实现，使用欧拉角 ZYX。
    """
    def __init__(self, initial_pose_matrix: torch.Tensor):
        super().__init__()
        # 从初始矩阵估计旋转和平移 (简化逻辑)
        initial_rotation = initial_pose_matrix[:3, :3]
        initial_translation = initial_pose_matrix[:3, 3]
        # TODO: 将旋转矩阵转换为欧拉角或其他6D表示，这里用randn简化
        self.pose_vec = nn.Parameter(torch.randn(6, device=initial_pose_matrix.device)) 
        
    def forward(self) -> torch.Tensor:
        """从6D向量构造4x4矩阵"""
        # TODO: 实现从6D向量 (例如: rx, ry, rz, tx, ty, tz) 到4x4矩阵的精确转换
        # 这里返回一个简化的单位矩阵+参数平移，仅用于结构演示
        mat = torch.eye(4, device=self.pose_vec.device)
        mat[:3, 3] = self.pose_vec[3:] # 假设后三个是平移
        # 旋转部分需要更复杂的实现 (例如 Rodrigues' formula)
        return mat

# ##############################################################################
# 1. 主配准函数
# ##############################################################################

def register(config):
    """
    执行在线配准流程的主函数。
    """
    # --- A. 初始化与模型加载 ---
    print("--- 1. 初始化与加载模型 ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载预训练的3D场景编码器
    scene_encoder = SceneEncoderHashGrid(feature_dim=config['feature_dim']).to(device)
    scene_encoder.load_state_dict(torch.load(config['scene_encoder_path'], map_location=device))
    scene_encoder.eval() # 设置为评估模式
    scene_encoder.requires_grad_(False) # 冻结权重
    print(f"3D场景编码器加载自: {config['scene_encoder_path']}")

    # 加载预训练的2D图像编码器
    image_encoder = ImageEncoder(pretrained=False).to(device) # pretrained=False 因为我们要加载自己的权重
    image_encoder.load_state_dict(torch.load(config['image_encoder_path'], map_location=device))
    image_encoder.eval() # 设置为评估模式
    image_encoder.requires_grad_(False) # 冻结权重
    print(f"2D图像编码器加载自: {config['image_encoder_path']}")

    # 初始化可微特征投影仪 (参数需与训练时一致)
    projector = DifferentiableProjector(
        height=config['image_height'], width=config['image_width'], n_samples=config['n_samples_per_ray'],
        focal_length=config['focal_length'], near=config['near_plane'], far=config['far_plane'],
    ).to(device)
    
    # --- B. 加载目标图像并提取特征 ---
    print("\n--- 2. 加载目标X光并提取特征 ---")
    
    # 加载并预处理实时X光图像
    I_live = preprocess_single_xray(
        config['live_xray_path'], 
        config['image_height'], 
        config['image_width'], 
        device
    )
    print(f"实时X光图像加载自: {config['live_xray_path']}")

    # 提取目标特征图
    with torch.no_grad():
        F_target = image_encoder(I_live)
    print(f"目标特征图已提取，形状: {F_target.shape}")

    # --- C. 初始化位姿与优化器 ---
    print("\n--- 3. 初始化位姿与优化器 ---")

    # 加载或设置初始位姿猜测 (4x4矩阵)
    # 这里我们假设一个初始位姿矩阵存储在numpy文件中，或者直接创建一个
    if config['initial_pose_path'] and os.path.exists(config['initial_pose_path']):
        initial_pose_np = np.load(config['initial_pose_path'])
        initial_pose = torch.from_numpy(initial_pose_np).float().to(device)
        print(f"从文件加载初始位姿: {config['initial_pose_path']}")
    else:
        # 如果没有提供初始位姿，创建一个默认位姿 (例如单位矩阵 + Z轴偏移)
        initial_pose = torch.eye(4, device=device)
        initial_pose[2, 3] = 2.5 # 示例偏移，应根据场景调整
        print("未提供初始位姿文件，使用默认位姿。")
        
    # 将4x4矩阵参数化为可优化的形式 (例如6D向量)
    # 注意：这里的PoseParameterization是一个简化示例
    pose_param = PoseParameterization(initial_pose).to(device)
    
    # 定义仅优化位姿参数的优化器
    optimizer = optim.Adam(pose_param.parameters(), lr=config['lr_pose'])
    
    # 定义配准损失 (L2距离)
    registration_loss_fn = nn.MSELoss()

    # --- D. 迭代优化位姿 ---
    print("\n--- 4. 开始迭代优化位姿 ---")
    
    # 使用tqdm显示优化进度
    progress_bar = tqdm(range(config['num_iterations']), desc="配准优化")
    for iter_num in progress_bar:
        # 1. 从可优化参数构造当前的4x4位姿矩阵
        current_pose_matrix = pose_param() 
        
        # 2. 使用当前位姿，通过投影仪渲染投影特征
        #    注意 projector 可能不支持批处理，确保输入是 [4, 4]
        F_proj_current = projector(scene_encoder, current_pose_matrix.squeeze(0)) # 假设 projector 处理单个 pose
        
        # 3. 计算隐空间中的L2损失
        loss = registration_loss_fn(F_proj_current, F_target)
        
        # 4. 反向传播，仅计算位姿参数的梯度
        optimizer.zero_grad()
        loss.backward()
        
        # 5. 更新位姿参数
        optimizer.step()
        
        # 在进度条上显示当前损失
        progress_bar.set_postfix(loss=f"{loss.item():.6f}")

    # --- E. 获取最终位姿 ---
    print("\n--- 5. 优化完成 ---")
    
    # 从优化后的参数构造最终的4x4位姿矩阵
    final_pose_matrix = pose_param().detach() # detach()确保不再追踪梯度
    print("最终优化后的位姿矩阵:")
    print(final_pose_matrix.cpu().numpy())
    
    # 保存最终位姿
    if config['output_pose_path']:
        np.save(config['output_pose_path'], final_pose_matrix.cpu().numpy())
        print(f"最终位姿已保存至: {config['output_pose_path']}")

    # --- F. (可选) 生成可视化DRR ---
    if config['generate_visualization']:
        print("\n--- 6. 生成最终可视化DRR ---")
        try:
            # 需要一个物理DRR渲染器实例 (这里假设它在config中定义或可以构建)
            # from diffdrr.drr import DRR # 确保导入
            # physical_drr_renderer = DRR(...) # 需要正确的参数初始化
            
            # --- 注意：这部分需要您根据物理渲染器的具体API进行调整 ---
            # 假设物理渲染器接受RigidTransform对象
            # from diffpose.calibration import RigidTransform # 确保导入
            # final_pose_rt = RigidTransform(rotation=final_pose_matrix[:3, :3], translation=final_pose_matrix[:3, 3])
            
            # with torch.no_grad():
            #     I_final_drr = physical_drr_renderer(pose=final_pose_rt) # 调用渲染
                
            # 保存或显示最终DRR
            # final_drr_np = I_final_drr.squeeze().cpu().numpy()
            # final_drr_np = (np.clip(final_drr_np, 0, 1) * 255).astype(np.uint8)
            # imageio.imwrite(config['output_drr_path'], final_drr_np)
            # print(f"最终可视化DRR已保存至: {config['output_drr_path']}")
            print("可视化DRR生成部分已注释掉，请根据您的物理渲染器API取消注释并调整。")

        except Exception as e:
            print(f"生成可视化DRR时出错: {e}")
            print("请确保物理渲染器已正确配置并可以访问必要的CT数据。")
            
    return final_pose_matrix

# ##############################################################################
# 4. 主程序入口与参数解析
# ##############################################################################
if __name__ == '__main__':
    # 使用 argparse 来从命令行接收参数，使脚本更灵活
    parser = argparse.ArgumentParser(description="LPM 在线配准脚本")
    
    # --- 必须提供的路径 ---
    parser.add_argument('--scene_encoder_path', type=str, required=True, help='预训练的3D场景编码器模型 (.pth) 路径')
    parser.add_argument('--image_encoder_path', type=str, required=True, help='预训练的2D图像编码器模型 (.pth) 路径')
    parser.add_argument('--live_xray_path', type=str, required=True, help='需要配准的实时X光图像文件路径')
    
    # --- 可选的输入/输出路径 ---
    parser.add_argument('--initial_pose_path', type=str, default=None, help='初始位姿猜测的numpy文件 (.npy) 路径 (4x4矩阵)')
    parser.add_argument('--output_pose_path', type=str, default='./final_pose.npy', help='保存最终优化位姿的路径 (.npy)')
    
    # --- 模型与渲染参数 (应与训练时匹配) ---
    parser.add_argument('--feature_dim', type=int, default=32, help='场景编码器输出的特征维度')
    parser.add_argument('--image_height', type=int, default=256, help='投影仪和图像处理的目标高度')
    parser.add_argument('--image_width', type=int, default=256, help='投影仪和图像处理的目标宽度')
    parser.add_argument('--n_samples_per_ray', type=int, default=128, help='每条射线上的采样点数')
    parser.add_argument('--focal_length', type=float, default=300.0, help='虚拟相机的焦距')
    parser.add_argument('--near_plane', type=float, default=1.5, help='渲染近裁剪平面')
    parser.add_argument('--far_plane', type=float, default=3.5, help='渲染远裁剪平面')
    
    # --- 优化参数 ---
    parser.add_argument('--num_iterations', type=int, default=200, help='位姿优化迭代次数')
    parser.add_argument('--lr_pose', type=float, default=1e-3, help='位姿优化学习率')
    
    # --- 可视化选项 ---
    parser.add_argument('--generate_visualization', action='store_true', help='是否生成最终的可视化DRR图像')
    parser.add_argument('--output_drr_path', type=str, default='./final_drr.png', help='保存最终可视化DRR的路径 (.png)')
    
    args = parser.parse_args()
    
    # 将解析后的参数转换为配置字典
    config = vars(args)
    
    # 启动配准流程
    final_pose = register(config)
