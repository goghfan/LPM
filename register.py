# -*- coding: utf-8 -*-

"""
[!! 最终修复版 - 2025/10/30 !!]
本文件是LPM框架的在线配准脚本。

[修复历史]
1. 修正了 preprocess_single_xray (移除 Log)。
2. 修正了 F.3. 可视化位姿转换 (R.transpose)。
3. 修正了 F.5. 可视化后处理 (移除 Log)。
4. [!! 关键 !!] 修正了 F.1 & F.2，使用 'load' 函数加载渲染器，
   以确保与 live_xray 完全一致的渲染参数 (解决旋转问题)。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import numpy as np
import os
import imageio # 用于加载和保存图像
from tqdm import tqdm
# import argparse # [!! 已移除 !!] 
import torch.nn.functional as F

# --- 项目内部模块导入 ---
from models.EfficentNet import SceneEncoderHashGrid, ImageEncoder
from models.scene_encoder_mlp import SceneEncoderMLP # 作为备选项
from projector import DifferentiableProjector
from utils import get_rays
from data.dataloader import LPMDataset # 用于加载归一化参数和CT体积
from diffpose.deepfluoro import DeepFluoroDataset # [!! 已使用 !!] 用于加载 x0, y0, focal_len
from diffpose.calibration import RigidTransform # 用于DRR可视化
from diffdrr.drr import DRR # 用于DRR可视化
from diffdrr.utils import se3_log_map, se3_exp_map # 关键：用于位姿参数化

# [!! 新增导入 !!] 从 test_drr.py 借鉴，用于加载一致的渲染器
try:
    from get_deepfluoro_4000 import load
except ImportError:
    print("="*50)
    print("!! 错误: 无法从 'get_deepfluoro_4000.py' 导入 'load' 函数。")
    print("!! 请确保此脚本与 'get_deepfluoro_4000.py' 在同一个文件夹中。")
    print("="*50)
    exit()

# ##############################################################################
# 1. 辅助模块 (预处理, 位姿参数化, 归一化)
# ##############################################################################

def preprocess_single_xray(image_path: str, target_height: int, target_width: int, device: torch.device) -> torch.Tensor:
    """
    [!! 已修复 !!]
    加载单张X光图像并进行预处理。
    我们只需要： 1. 加载 2. 调整大小 3. 归一化 [0, 255] -> [0, 1]
    """
    try:
        with imageio.v2.imopen(image_path, 'r') as f:
            img = f.read(index=0).astype(np.float32) # img 范围是 [0, 255]
    except Exception as e:
        print(f"使用 imageio 加载失败: {e}。")
        return None

    # 处理灰度图 (H, W) -> (1, 1, H, W)
    if img.ndim == 2:
        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
    # 处理RGB图 (H, W, 3) -> (1, 1, H, W)
    elif img.ndim == 3:
        img_tensor = torch.from_numpy(img).mean(dim=2).unsqueeze(0).unsqueeze(0)
    else:
        print(f"不支持的图像维度: {img.shape}")
        return None

    img_tensor = img_tensor.to(device)

    # 1. 调整尺寸
    img_tensor = F.interpolate(img_tensor, size=(target_height, target_width), mode='bilinear', align_corners=False)

    # 2. [!! 已移除 !!] Log 变换

    # 3. [!! 已修改 !!] Min-Max 归一化到 [0, 1]
    img_tensor = img_tensor / 255.0
    img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
    
    return img_tensor

class OptimizablePoseSE3(nn.Module):
    # ... [此类保持不变] ...
    def __init__(self, initial_pose_matrix: torch.Tensor):
        super().__init__()
        if initial_pose_matrix.shape != (4, 4):
            raise ValueError(f"初始位姿必须是 (4, 4) 矩阵, 但收到 {initial_pose_matrix.shape}")
        
        temp_matrix_for_logmap = initial_pose_matrix.clone()
        temp_matrix_for_logmap[:3, 3] = 0.0 
        temp_matrix_for_logmap[:3, :3] = initial_pose_matrix[:3, :3] 
        temp_matrix_for_logmap[3, :3] = initial_pose_matrix[3, :3] 
        temp_matrix_for_logmap[3, 3] = 1.0
        
        initial_se3_log = se3_log_map(temp_matrix_for_logmap.unsqueeze(0)) 
        self.se3_log = nn.Parameter(initial_se3_log.squeeze(0))
        
    def forward(self) -> torch.Tensor:
        matrix_row_vec = se3_exp_map(self.se3_log.unsqueeze(0))
        return matrix_row_vec.squeeze(0)

def normalize_pose_batch(pi_phys_batch: torch.Tensor, C_phys_tensor: torch.Tensor, s: float) -> torch.Tensor:
    # ... [此函数保持不变] ...
    R_batch = pi_phys_batch[:, :3, :3] 
    t_phys_batch = pi_phys_batch[:, 3, :3] 
    t_norm_batch = (t_phys_batch - C_phys_tensor.unsqueeze(0)) * s
    pi_norm_batch = torch.eye(4, device=pi_phys_batch.device, dtype=pi_phys_batch.dtype).unsqueeze(0).repeat(pi_phys_batch.shape[0], 1, 1)
    pi_norm_batch[:, :3, :3] = R_batch
    pi_norm_batch[:, 3, :3] = t_norm_batch 
    return pi_norm_batch

# ##############################################################################
# 2. 主配准函数
# ##############################################################################

def register(config):
    """
    [!! 最终修复版 !!]
    使用正确的预处理和可视化参数执行完整优化。
    """
    # --- A. 初始化与加载模型 ---
    print("--- 1. 初始化与加载模型 ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 加载归一化参数和可视化所需数据 ---
    print("正在从数据集中加载归一化参数...")
    try:
        # [!! 注意 !!] 'load' 函数会自己加载 volume, 
        # 但我们仍然需要 LPMDataset 来获取 C_phys 和 s
        temp_dataset = LPMDataset(config['h5_dataset_path'], config['specimen_id'], split='val')
        C_phys_tensor = torch.from_numpy(temp_dataset.C_phys_xyz).float().to(device)
        normalization_scale = temp_dataset.s
        temp_dataset.close() # C 和 s 已拿到，关闭文件

        # [!! 已修改 !!] 始终从 DeepFluoroDataset 加载 x0, y0, 和 真实的focal_len
        print(f"...正在从原始数据集 '{config['original_h5_path']}' 加载 x0, y0, 和 focal_len...")
        specimen_loader = DeepFluoroDataset(config['specimen_id'], filename=config['original_h5_path'])
        x0 = specimen_loader.x0
        y0 = specimen_loader.y0
        true_focal_length = specimen_loader.focal_len 
        print(f"    > 成功加载: x0={x0}, y0={y0}, true_focal_length={true_focal_length}")
            
        print(f"  > 成功加载 (来自 {config['h5_dataset_path']}): s={normalization_scale:.6f}, C_phys={C_phys_tensor.cpu().numpy()}")

    except Exception as e:
        print(f"!!! 错误: 无法从HDF5文件加载数据。")
        print(f"  请确保 h5_dataset_path 和 original_h5_path 正确。错误: {e}")
        return

    # --- 加载预训练的3D场景编码器 ---
    print("加载3D场景编码器 (F_θ)...")
    scene_encoder = SceneEncoderMLP(feature_dim=config['feature_dim']).to(device)
    try:
        scene_encoder.load_state_dict(torch.load(config['scene_encoder_path'], map_location=device))
        scene_encoder.eval()
        scene_encoder.requires_grad_(False)
        print(f"  > 3D场景编码器加载自: {config['scene_encoder_path']}")
    except Exception as e:
        print(f"!!! 错误: 加载场景编码器失败: {e}")
        return

    # --- 加载预训练的2D图像编码器 ---
    print("加载2D图像编码器 (E_ϕ)...")
    try:
        image_encoder = ImageEncoder(pretrained=False).to(device)
        image_encoder.load_state_dict(torch.load(config['image_encoder_path'], map_location=device))
        image_encoder.eval()
        image_encoder.requires_grad_(False)
        print(f"  > 2D图像编码器加载自: {config['image_encoder_path']}")
    except Exception as e:
        print(f"!!! 错误: 加载图像编码器失败: {e}")
        return

    # --- 初始化可微特征投影仪 ---
    # [注] 这是用于特征优化的，与F节的可视化渲染器无关
    print("初始化可微投影仪 (P)...")
    
    focal_length_norm = true_focal_length * normalization_scale
    near_plane_norm = config['near_plane'] * normalization_scale
    far_plane_norm = config['far_plane'] * normalization_scale
    print(f"  > [!] 投影仪使用 *真实* 焦距: {true_focal_length:.2f} (归一化后: {focal_length_norm:.4f})")
    print(f"  > [!] 投影仪使用 *物理* 近/远平面: {config['near_plane']:.2f} / {config['far_plane']:.2f}")

    projector = DifferentiableProjector(
        height=config['feature_height'], 
        width=config['feature_width'],
        n_samples=config['n_samples_per_ray'],
        focal_length = focal_length_norm,
        near = near_plane_norm,
        far = far_plane_norm,
    ).to(device)
    
    # --- B. 加载目标图像并提取特征 ---
    print("\n--- 2. 加载目标X光并提取特征 (使用修正后的预处理) ---")

    I_live = preprocess_single_xray(
        config['live_xray_path'],
        config['image_height'], 
        config['image_width'],
        device
    )
    if I_live is None:
        print(f"!!! 错误: 无法加载或处理实时X光图像: {config['live_xray_path']}")
        return

    print(f"  > 实时X光图像加载自: {config['live_xray_path']}")

    with torch.no_grad():
        F_target = image_encoder(I_live) # [1, C, H_feat, W_feat]
    print(f"  > 目标特征图已提取，形状: {F_target.shape}")


    # --- C. 初始化位姿与优化器 ---
    print("\n--- 3. 初始化位姿与优化器 ---")

    # [注] 这是 Ground Truth 位姿，优化器应该几乎不动
    initial_pose_np = np.array([
        [-4.2012874e-02,  9.3289483e-01, -3.5769010e-01,  0.0000000e+00], 
        [ 2.6626430e-02, -3.5683364e-01, -9.3378842e-01,  0.0000000e+00], 
        [-9.9876219e-01, -4.8755147e-02, -9.8480554e-03,  0.0000000e+00], 
        [ 2.4465094e+02,  3.1753845e+02,  6.4038788e+01,  1.0000000e+00]  
    ], dtype=np.float32)
    
    initial_pose = torch.from_numpy(initial_pose_np).float().to(device)
    print(f"  > [!] 使用初始位姿 (行向量约定):")
    print(initial_pose_np)
    
    pose_param = OptimizablePoseSE3(initial_pose).to(device)

    optimizer = optim.Adam(pose_param.parameters(), lr=config['lr_pose'])
    registration_loss_fn = nn.MSELoss()
    print(f"  > 优化器: Adam, 学习率: {config['lr_pose']}")

    # --- D. 迭代优化位姿 ---
    print("\n--- 4. 开始迭代优化位姿 ---")

    progress_bar = tqdm(range(config['num_iterations']), desc="配准优化")
    for iter_num in progress_bar:
        optimizer.zero_grad()
        current_pose_phys = pose_param() 
        current_pose_norm = normalize_pose_batch(
            current_pose_phys.unsqueeze(0), 
            C_phys_tensor,
            normalization_scale
        )
        F_proj_current = projector(scene_encoder, current_pose_norm.squeeze(0)) 
        loss = registration_loss_fn(F_proj_current, F_target)
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(loss=f"{loss.item():.6f}")
    
    print(f"优化完成。最终损失: {loss.item():.8f}")


    # --- E. 获取最终位姿 ---
    print("\n--- 5. 优化完成 ---")
    final_pose_matrix = pose_param().detach() # (4, 4) 行向量约定矩阵
    print("最终优化后的物理位姿矩阵 (行向量约定):")
    print(final_pose_matrix.cpu().numpy())

    if config['output_pose_path']:
        np.save(config['output_pose_path'], final_pose_matrix.cpu().numpy())
        print(f"  > 最终位姿已保存至: {config['output_pose_path']}")

    # --- F. (可选) 生成可视化DRR ---
    if config['generate_visualization']:
        print("\n--- 6. 生成最终可视化DRR (使用一致的参数和后处理) ---")
        try:
            # 1. & 2. [!! 已修正 - 解决旋转问题 !!] 
            #    不再手动初始化 DRR。
            #    使用与 test_drr.py 相同的 'load' 函数来获取预配置的渲染器。
            #    这能保证渲染参数 (sdr, delx, x0, y0, ...) 绝对一致。
            
            print("    > 正在使用 'load' 函数加载预配置的DRR渲染器 (同 test_drr.py)...")
            
            # 'load' 函数会返回它自己的 CT 体积 (specimen) 和渲染器
            # 我们使用它返回的 drr_renderer
            specimen_vis, isocenter_pose_vis, drr_renderer = load(
                id_number=config['specimen_id'],
                height=config['image_height'], # 确保高度匹配
                device=device,
                h5_datapath=config['original_h5_path'] # 使用原始数据路径
            )
            
            # 确保渲染器在正确的设备上 (load 可能在 cpu 上创建)
            drr_renderer = drr_renderer.to(device)
            
            # print(f"    > 成功加载渲染器: H={drr_renderer.height}, W={drr_renderer.width}, delx={drr_renderer.delx:.6f}, sdr={drr_renderer.sdr:.2f}")

            # 3. 准备位姿 [!! 关键修正 - 基于 test_drr.py !!]
            print("    > [!!] 修正位姿转换: R = R_matrix[:3, :3].transpose(-1, -2), t = t_matrix[3, :3]")
            final_R_for_vis = final_pose_matrix[:3, :3].transpose(-1, -2) # R_col = R_row.T
            final_t_for_vis = final_pose_matrix[3, :3]                # t_row
            final_pose_rt = RigidTransform(R=final_R_for_vis, t=final_t_for_vis)


            # 4. 渲染DRR (生成 原始 衰减图)
            with torch.no_grad():
                I_final_drr_raw_tensor = drr_renderer(None, None, None, pose=final_pose_rt) 

            # 5. [!! 关键 - 已修正 !!] 
            #    应用与目标图像 (live_xray) 相同的后处理 (仅 Min-Max 归一化)
            print("    > 正在应用 Min-Max 归一化 (以匹配目标图像)...")
            
            # --- [!! 已移除 !!] Log 变换 ---

            # --- Min-Max 归一化到 [0, 1] (直接在 raw tensor 上操作) ---
            I_final_drr_tensor = I_final_drr_raw_tensor.clone() # [!!] 使用 raw tensor
            
            min_val = torch.min(I_final_drr_tensor)
            max_val = torch.max(I_final_drr_tensor)
            if (max_val - min_val) > 1e-6:
                I_final_drr_tensor = (I_final_drr_tensor - min_val) / (max_val - min_val)
            else:
                I_final_drr_tensor = torch.zeros_like(I_final_drr_tensor)
            
            # 6. 保存图像 (归一化到 0-255)
            final_drr_np = I_final_drr_tensor.squeeze().cpu().numpy()
            
            if np.isnan(final_drr_np).any():
                print("!!! 警告: 最终DRR包含NaN，将替换为0。")
                final_drr_np = np.nan_to_num(final_drr_np, nan=0.0)

            final_drr_np = (final_drr_np * 255).astype(np.uint8)
            
            imageio.imwrite(config['output_drr_path'], final_drr_np)
            print(f"  > 最终可视化DRR已保存至: {config['output_drr_path']}")
            print(f"  > [!] 此DRR现在在几何形状和视觉外观上都应匹配目标图像。")

        except Exception as e:
            print(f"!!! 警告: 生成可视化DRR时出错: {e}")
            import traceback
            traceback.print_exc() # 打印详细错误

    return final_pose_matrix

# ##############################################################################
# 3. 主程序入口与参数解析
# ##############################################################################
if __name__ == '__main__':
    
    print("===================================================")
    print("== [!! 最终修复版 (2025/10/30) !!] 正在运行 ==")
    print("===================================================")

    config = {
        # --- 必需的路径 ---
        'scene_encoder_path': "./checkpoints/checkpoints-1030/best_scene_encoder.pth",
        'image_encoder_path': "./checkpoints/checkpoints-1030/best_image_encoder.pth",
        'live_xray_path': "live_xray_for_registration.png", # <--- 必须由 generation_drr.py 生成
        'h5_dataset_path': "F:/desktop/2D-3D/LPM/LPM/drr_pose_dataset_with_landmarks.h5",
        'original_h5_path': "F:\\desktop\\2D-3D\\LPM\\LPM\\data\\ipcai_2020_full_res_data\\ipcai_2020_full_res_data.h5",
        
        # --- 必需的ID ---
        'specimen_id': 1,
        
        # --- 可选的输入/输出路径 ---
        'initial_pose_path': None, # 已被硬编码的 initial_pose_np 替代
        'output_pose_path': "./final_registered_pose.npy",
        
        # --- 模型与渲染参数 ---
        'feature_dim': 1536,
        'image_height': 256,
        'image_width': 256,
        'feature_height': 8,
        'feature_width': 8,
        'n_samples_per_ray': 128,
        'near_plane': 10.0,
        'far_plane': 400.0,
        
        # --- 优化参数 ---
        'num_iterations': 200, # 您可以增加这个值以获得更精确的结果
        'lr_pose': 1e-3,       
        
        # --- 可视化选项 ---
        'generate_visualization': True,
        'output_drr_path': "./final_registered_drr.png"
    }

    # 启动配准流程
    final_pose = register(config)