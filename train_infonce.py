# -*- coding: utf-8 -*-

"""
本文件是LPM框架的核心训练脚本。
[已更新]
1. 使用 InfoNCE Loss 替代 Triplet Margin Loss。
2. 在计算相似度前进行 L2 特征归一化。
3. 添加了 temperature 超参数。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F # 用于 L2 归一化和余弦相似度
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np

# --- TensorBoard 和可视化导入 ---
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import io
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid

# --- 项目内部模块导入 ---
from data.dataloader import LPMDataset
# [注意] 确保从包含 float() 修复的文件导入
from models.EfficentNet import SceneEncoderHashGrid, ImageEncoder
from models.scene_encoder_mlp import SceneEncoderMLP
from projector import DifferentiableProjector

# --- 外部库导入 (用于可视化) ---
from diffdrr.drr import DRR
from diffpose.calibration import RigidTransform, perspective_projection 
from diffdrr.utils import convert as convert_rotation
# ##############################################################################
# 1. 辅助模块
# ##############################################################################

# [!!! 修改 1 !!!] 移除 TripletMarginLoss 类

# [!!! 修改 2 !!!] 定义 InfoNCE Loss 函数
def info_nce_loss(anchor, positive, negative, temperature=0.1, eps=1e-6):
    """
    计算简化的 InfoNCE 损失 (针对一个正样本和一个负样本)。
    
    Args:
        anchor (torch.Tensor): 锚点特征 (B, C, H, W) 或 (B, D)。
        positive (torch.Tensor): 正样本特征 (B, C, H, W) 或 (B, D)。
        negative (torch.Tensor): 负样本特征 (B, C, H, W) 或 (B, D)。
        temperature (float): 温度超参数。
        eps (float): 防止除以零的小常数。

    Returns:
        torch.Tensor: InfoNCE 损失值 (标量)。
    """
    # 1. L2 归一化特征 (沿通道/特征维度)
    #    假设输入已经是 (B, D) 或需要先 flatten
    if anchor.dim() > 2: # 如果是特征图 (B, C, H, W)
        anchor = anchor.flatten(start_dim=1)     # (B, C*H*W)
        positive = positive.flatten(start_dim=1) # (B, C*H*W)
        negative = negative.flatten(start_dim=1) # (B, C*H*W)

    anchor_norm = F.normalize(anchor, p=2, dim=1)
    positive_norm = F.normalize(positive, p=2, dim=1)
    negative_norm = F.normalize(negative, p=2, dim=1)

    # 2. 计算余弦相似度
    #    sim(a, b) = sum(a * b) / (sqrt(sum(a^2)) * sqrt(sum(b^2)))
    #    由于已经 L2 归一化，分母为 1，直接点积即可
    sim_pos = torch.sum(anchor_norm * positive_norm, dim=1) # (B,)
    sim_neg = torch.sum(anchor_norm * negative_norm, dim=1) # (B,)

    # 3. 计算 Softmax 分母 (exp(pos) + exp(neg))
    logits = torch.stack([sim_pos, sim_neg], dim=1) / temperature # (B, 2)
    
    # 4. 计算 LogSoftmax
    #    我们希望正样本的概率最大化，即 LogSoftmax 在第 0 个位置的值最大化
    #    交叉熵损失等价于 -log(softmax(positive))
    #    目标标签是 0 (代表 positive)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=anchor.device)
    loss = F.cross_entropy(logits, labels)
    
    return loss

# [!!! 關鍵修改 !!!] get_random_pose_offset 現在包含旋轉
def get_random_pose_offset(batch_size: int, device: torch.device, rot_std=0.1, trans_std=50.0) -> torch.Tensor:
    """
    生成一个包含随机 *旋转* 和 *平移* 的物理空间偏移矩阵 (Batch, 4, 4)。
    """
    offset_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # 1. 生成随机旋转 (轴角表示)
    #   向量方向是旋转轴，向量长度是旋转角度 (弧度)
    random_rotation_vector = torch.randn(batch_size, 3, device=device) * rot_std
    
    # 2. 将轴角向量转换为 3x3 旋转矩阵
    #   使用 diffdrr.utils.convert (它内部调用 pytorch3d)
    try:
        rotation_matrix = convert_rotation(random_rotation_vector, "axis_angle", "matrix")
    except ImportError:
        print("警告: 无法从 diffdrr.utils 导入 convert。跳过旋转扰动。")
        print("请确保 pytorch3d 已正确安装。")
        rotation_matrix = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1) # Fallback

    # 3. 应用旋转矩阵到偏移矩阵
    offset_matrix[:, :3, :3] = rotation_matrix
    
    # 4. 生成并应用随机平移 (保持不变)
    random_translations = torch.randn(batch_size, 3, device=device) * trans_std
    offset_matrix[:, :3, 3] = random_translations
    
    return offset_matrix

def normalize_pose_batch(pi_phys_batch: torch.Tensor, C_phys_tensor: torch.Tensor, s: float) -> torch.Tensor:
    # 保持不变
    R_batch = pi_phys_batch[:, :3, :3]
    t_phys_batch = pi_phys_batch[:, :3, 3]
    t_norm_batch = (t_phys_batch - C_phys_tensor.unsqueeze(0)) * s
    pi_norm_batch = torch.eye(4, device=pi_phys_batch.device, dtype=pi_phys_batch.dtype).unsqueeze(0).repeat(pi_phys_batch.shape[0], 1, 1)
    pi_norm_batch[:, :3, :3] = R_batch
    pi_norm_batch[:, :3, 3] = t_norm_batch
    return pi_norm_batch

# --- 投影函数 (用于可视化) ---
def convert_diffdrr_to_deepfluoro(dataset: LPMDataset, pose: RigidTransform):
    # 保持不变
    return (
        dataset.lps2volume.inverse()
        .compose(pose.inverse())
        .compose(dataset.translate)
        .compose(dataset.flip_xz)
    )

def project_landmarks_for_vis(dataset: LPMDataset, pose: RigidTransform) -> np.ndarray:
    # 保持不变
    device = pose.get_matrix().device
    extrinsic = convert_diffdrr_to_deepfluoro(dataset, pose.cpu()).to(device)
    intrinsic_dev = dataset.intrinsic.to(device)
    fiducials_3d_dev = dataset.fiducials_3d_phys.to(device)
    fiducials_3d_dev_batched = fiducials_3d_dev.unsqueeze(0) 
    projected_coords = perspective_projection(extrinsic, intrinsic_dev, fiducials_3d_dev_batched)
    return projected_coords.squeeze(0).cpu().numpy()

def plot_landmarks_to_tensor(image_tensor: torch.Tensor, true_landmarks: np.ndarray) -> torch.Tensor:
    # 保持不变
    image_np = image_tensor.squeeze().cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_np, cmap='gray')
    ax.scatter(true_landmarks[:, 0], true_landmarks[:, 1], c='lime', marker='x', s=100, label='True Landmarks')
    ax.legend()
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf)
    img_tensor = TF.to_tensor(img)
    plt.close(fig)
    return img_tensor
# --- 投影函数结束 ---

# ##############################################################################
# 2. 主训练函数
# ##############################################################################

def train(config):
    # --- A. 初始化 ---
    print("--- 1. 初始化所有组件 ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    writer = SummaryWriter(log_dir=config['log_dir'])

    # 1. 初始化数据集
    print("正在加载训练数据集...")
    train_dataset = LPMDataset(config['h5_dataset_path'], config['specimen_id'], split='train')
    print("\n正在加载验证数据集...")
    val_dataset = LPMDataset(config['h5_dataset_path'], config['specimen_id'], split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, 
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, 
                            num_workers=0, pin_memory=True)
    print("\n数据集加载完毕。")

    # 2. 获取归一化参数
    normalization_scale = train_dataset.s
    C_phys_tensor = torch.from_numpy(train_dataset.C_phys_xyz).float().to(device)
    print(f"训练脚本已获取归一化参数: s={normalization_scale:.6f}, C_phys={C_phys_tensor.cpu().numpy()}")
    
    can_visualize_landmarks = val_dataset.can_project_landmarks
    if not can_visualize_landmarks:
        print("❌ 警告: Dataloader 报告无法投影地标点 (缺少元数据)。将跳过地标点可视化。")
    else:
        print("✅ 成功加载地标点和投影元数据，将激活可视化。")

    # 3. 初始化模型
    print("初始化模型: SceneEncoderHashGrid, ImageEncoder...")
    # 使用 1536 特征维度
    # scene_encoder = SceneEncoderHashGrid(
    #     bounding_box=(-1.0, 1.0), 
    #     feature_dim=config['feature_dim'] 
    # ).to(device)
    scene_encoder = SceneEncoderMLP(
        bounding_box=(-1.0, 1.0), 
        feature_dim=config['feature_dim'] 
    ).to(device)
    # ImageEncoder 不需要 output_dim
    image_encoder = ImageEncoder(pretrained=True).to(device)
    print("\n" + "="*50)
    print("--- 2D 图像编码器 (ImageEncoder) 架构 ---")
    print(image_encoder)
    print("\n" + "="*50)
    print("--- 3D 场景编码器 (SceneEncoderHashGrid) 架构 ---")
    print(scene_encoder)
    print("="*50 + "\n")
    # 4. 初始化可微特征投影仪
    focal_length_norm = config['focal_length'] * normalization_scale
    near_plane_norm = config['near_plane'] * normalization_scale
    far_plane_norm = config['far_plane'] * normalization_scale
    
    print(f"初始化投影仪: 渲染尺寸={config['feature_height']}x{config['feature_width']}, 归一化焦距={focal_length_norm:.4f}")
    # 仍然渲染 8x8 特征图
    projector = DifferentiableProjector(
        height=config['feature_height'], 
        width=config['feature_width'],  
        n_samples=config['n_samples_per_ray'],
        focal_length = focal_length_norm,
        near = near_plane_norm,
        far = far_plane_norm,
    ).to(device)

    # 5. 初始化物理DRR渲染器
    print("初始化物理DRR渲染器 (用于可视化)...")
    try:
        drr_renderer = DRR(
            train_dataset.volume_data,
            train_dataset.spacing_zyx,
            sdr=config['focal_length'] / 2, 
            height=config['image_height'],           
            delx=0.194 * (1536 / config['image_height']),
            width=config['image_width']             
        ).to(device)
    except Exception as e:
        print(f"!!! 警告: 初始化物理 DRR 渲染器失败: {e}")
        drr_renderer = None

    # 6. 初始化损失函数和优化器
    # [!!! 修改 3 !!!] 移除 TripletLoss 初始化
    # 我们将在循环中直接调用 info_nce_loss 函数
    print(f"使用 InfoNCE Loss, temperature={config['temperature']}")
    
    # 使用降低后的学习率
    optimizer = optim.Adam([
        {'params': scene_encoder.parameters(), 'lr': config['lr_scene_encoder']},
        {'params': image_encoder.parameters(), 'lr': config['lr_image_encoder']},
    ])

    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    best_val_loss = float('inf')

    # --- B. 训练循环 ---
    print("\n--- 2. 开始训练循环 ---")
    for epoch in range(config['num_epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['num_epochs']} ---")
        
        # --- 训练阶段 ---
        scene_encoder.train()
        image_encoder.train()
        train_loss_total = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"训练 (Epoch {epoch+1})")
        for i, (I_gt, pi_gt_phys) in enumerate(progress_bar):
            I_gt, pi_gt_phys = I_gt.to(device), pi_gt_phys.to(device)
            
            with torch.no_grad():
                offset = get_random_pose_offset(pi_gt_phys.shape[0], device)
                pi_wrong_phys = torch.bmm(offset, pi_gt_phys)
                pi_gt_norm = normalize_pose_batch(pi_gt_phys, C_phys_tensor, normalization_scale)
                pi_wrong_norm = normalize_pose_batch(pi_wrong_phys, C_phys_tensor, normalization_scale)

            # F_target 应该是 [B, 1536, 8, 8]
            F_target = image_encoder(I_gt)
            # F_proj 应该是 [B, 1536, 8, 8]
            F_proj_pos = torch.cat([projector(scene_encoder, p) for p in pi_gt_norm], dim=0)
            F_proj_neg = torch.cat([projector(scene_encoder, p) for p in pi_wrong_norm], dim=0)
            
            # [!!! 修改 4 !!!] 计算 InfoNCE Loss
            loss = info_nce_loss(F_target, F_proj_pos, F_proj_neg, 
                                 temperature=config['temperature'])
            
            optimizer.zero_grad()
            loss.backward()
            
            # [添加] 梯度裁剪 (推荐)
            torch.nn.utils.clip_grad_norm_(scene_encoder.parameters(), max_norm=1.0) 
            torch.nn.utils.clip_grad_norm_(image_encoder.parameters(), max_norm=1.0) 
            
            optimizer.step()
            
            train_loss_total += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss_total / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        print(f"Epoch {epoch+1} 结束, 平均训练损失: {avg_train_loss:.4f}")

        # --- 验证与可视化阶段 ---
        scene_encoder.eval()
        image_encoder.eval()
        val_loss_total = 0.0
        
        with torch.no_grad():
            progress_bar_val = tqdm(val_loader, desc=f"验证 (Epoch {epoch+1})")
            for i, (I_gt_val, pi_gt_val_phys) in enumerate(progress_bar_val):
                I_gt_val, pi_gt_val_phys = I_gt_val.to(device), pi_gt_val_phys.to(device)
                
                offset_val = get_random_pose_offset(pi_gt_val_phys.shape[0], device)
                pi_wrong_val_phys = torch.bmm(offset_val, pi_gt_val_phys)
                pi_gt_val_norm = normalize_pose_batch(pi_gt_val_phys, C_phys_tensor, normalization_scale)
                pi_wrong_val_norm = normalize_pose_batch(pi_wrong_val_phys, C_phys_tensor, normalization_scale)

                F_target_val = image_encoder(I_gt_val)
                F_proj_pos_val = torch.cat([projector(scene_encoder, p) for p in pi_gt_val_norm], dim=0)
                F_proj_neg_val = torch.cat([projector(scene_encoder, p) for p in pi_wrong_val_norm], dim=0)
                
                # [!!! 修改 5 !!!] 计算 InfoNCE Loss
                loss = info_nce_loss(F_target_val, F_proj_pos_val, F_proj_neg_val,
                                     temperature=config['temperature'])
                                     
                val_loss_total += loss.item()
                progress_bar_val.set_postfix(loss=f"{loss.item():.4f}")
                
                # --- 可视化 (代码无改动) ---
                if i == 0:
                    idx_vis = np.random.randint(0, I_gt_val.shape[0])
                    pose_phys_vis = pi_gt_val_phys[idx_vis]
                    fixed_image = I_gt_val[idx_vis] 
                    pose_rt = RigidTransform(R=pose_phys_vis[:3, :3], t=pose_phys_vis[:3, 3])
                    
                    # DRR 对比
                    if drr_renderer is not None:
                        try:
                            drr_generated = drr_renderer(None, None, None, pose=pose_rt)
                            error_map = torch.abs(fixed_image - drr_generated)
                            img_grid = make_grid([fixed_image.unsqueeze(0), drr_generated, error_map], normalize=True, nrow=3)
                            writer.add_image(f'Validation/Comparison', img_grid, epoch)
                        except Exception as e:
                            print(f"!!! 警告: 验证DRR可视化失败: {e}")
                            drr_renderer = None 
                    
                    # 地标点
                    if can_visualize_landmarks:
                        try:
                            true_landmarks_2d = project_landmarks_for_vis(val_dataset, pose_rt)
                            landmark_vis_tensor = plot_landmarks_to_tensor(fixed_image, true_landmarks_2d)
                            writer.add_image(f'Validation/Landmarks', landmark_vis_tensor, epoch)
                        except Exception as e:
                            print(f"!!! 警告: 地标点投影失败: {e}")
                            can_visualize_landmarks = False 
        
        avg_val_loss = val_loss_total / len(val_loader)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        print(f"Epoch {epoch+1} 结束, 平均验证损失: {avg_val_loss:.4f}")

        # --- 模型保存 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"✅ 发现新的最佳模型 (验证损失: {best_val_loss:.4f})，正在保存检查点...")
            torch.save(scene_encoder.state_dict(), os.path.join(config['checkpoint_dir'], 'best_scene_encoder.pth'))
            torch.save(image_encoder.state_dict(), os.path.join(config['checkpoint_dir'], 'best_image_encoder.pth'))

    # --- C. 训练结束 ---
    writer.close()
    train_dataset.close()
    val_dataset.close()
    print("\n--- 训练完成 ---")

# ##############################################################################
# 3. 主程序入口
# ##############################################################################
if __name__ == '__main__':
    # 配置参数
    config = {
        # 数据相关
        "h5_dataset_path": r"F:\desktop\2D-3D\LPM\LPM\drr_pose_dataset_with_landmarks.h5", 
        "specimen_id": 1,
        "image_height": 256,    # 2D 输入图像的 H
        "image_width": 256,     # 2D 输入图像的 W
        
        # 特征维度恢复为 1536
        "feature_dim": 1536,     # 特征通道 C (匹配 EfficientNet)
        "feature_height": 8,     # 特征图 H (匹配 EfficientNet)
        "feature_width": 8,      # 特征图 W (匹配 EfficientNet)
        
        # 训练过程相关
        "batch_size": 32,      
        "num_epochs": 100,     
        # 降低学习率
        "lr_scene_encoder": 5e-4,  # 降低 Scene Encoder 学习率
        "lr_image_encoder": 1e-4,  # 降低 Image Encoder 学习率
        "checkpoint_dir": "./checkpoints", 
        "log_dir": "./logs",           

        # 模型与渲染相关
        "n_samples_per_ray": 128,    
        
        # 物理单位 (mm)
        "focal_length": 300.0, 
        "near_plane": 1.5,     
        "far_plane": 3.5,      
        
        # [!!! 修改 6 !!!] InfoNCE Loss 相关
        # "triplet_margin": 1.0, # 不再需要 margin
        "temperature": 0.1,    # InfoNCE 温度参数 (常用值)
    }

    # 启动训练
    train(config)