# -*- coding: utf-8 -*-

"""
本文件包含LPM项目中可复用的辅助函数。
"""

import torch

def get_rays(height: int, width: int, focal_length: float, c2w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    根据给定的相机参数，为图像中的每个像素生成射线。

    Args:
        height (int): 图像高度。
        width (int): 图像宽度。
        focal_length (float): 相机焦距。
        c2w (torch.Tensor): "相机到世界" (camera-to-world) 的4x4变换矩阵。

    Returns:
        tuple[torch.Tensor, torch.Tensor]: 一个包含 (rays_o, rays_d) 的元组。
            - rays_o: 射线起点在世界坐标系中的坐标，形状为 [height, width, 3]。
            - rays_d: 射线方向在世界坐标系中的单位向量，形状为 [height, width, 3]。
    """
    device = c2w.device
    
    # 1. 创建像素网格坐标
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32, device=device),
        torch.arange(height, dtype=torch.float32, device=device),
        indexing='xy'
    )

    # 2. 将像素坐标转换为相机局部坐标系下的方向向量
    #    我们假设相机朝向-Z方向，Y轴向上，X轴向右。
    dirs = torch.stack([
        (i - width * 0.5) / focal_length,
        -(j - height * 0.5) / focal_length, # Y轴向下是图像坐标，向上是相机坐标，因此取负
        -torch.ones_like(i)
    ], dim=-1) # Shape: [height, width, 3]

    # 3. 使用旋转矩阵将方向向量从相机坐标系转换到世界坐标系
    #    c2w[:3, :3] 是4x4矩阵中的3x3旋转部分
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
    
    # 4. 射线起点就是相机在世界坐标系中的位置
    #    c2w[:3, 3] 是4x4矩阵中的3x1平移部分
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    
    return rays_o, rays_d