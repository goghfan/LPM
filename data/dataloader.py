# data/dataloader.py

import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
# 导入 RigidTransform 以便加载矩阵
from diffpose.calibration import RigidTransform

class LPMDataset(Dataset):
    """
    用于LPM框架的数据加载器。
    
    [已修正] 核心功能：
    1. 精确读取 HDF5 结构 (vol/pixels, vol/spacing, vol/origin)。
    2. 使用 'origin' 和 'spacing' 精确计算归一化参数 (C_phys, s)。
    3. [已修正] 使用正确的构造函数 (R, t) 加载投影元数据。
    4. 加载 3D 物理地标点 (fiducials_3d_phys)。
    5. 设置 can_project_landmarks = True。
    6. 返回 (图像, 物理位姿)。
    """
    def __init__(self, h5_dataset_path: str, specimen_id: int, split: str = 'train', train_split_ratio: float = 0.8, random_seed: int = 42):
        super().__init__()
        
        self.h5_dataset_path = h5_dataset_path
        self.specimen_id = specimen_id
        self.split = split
        
        try:
            self.hf = h5py.File(self.h5_dataset_path, 'r')
        except Exception as e:
            raise IOError(f"无法打开HDF5文件: {self.h5_dataset_path}. 错误: {e}")
            
        specimen_group_key = f'specimen_{self.specimen_id}'
        if specimen_group_key not in self.hf:
            raise ValueError(f"在HDF5文件中未找到样本组: {specimen_group_key}")
            
        self.specimen_group = self.hf[specimen_group_key]
        self.projections_group = self.specimen_group['projections']

        # --- 1. 计算归一化参数 (C_phys 和 s) ---
        print(f"LPMDataset: 正在为 specimen_{specimen_id} 计算归一化参数...")
        try:
            self.volume_data = self.specimen_group['vol/pixels'][:]
            self.spacing_zyx = self.specimen_group['vol/spacing'][:].flatten() 
            self.origin_xyz = self.specimen_group['vol/origin'][:].flatten()
        except KeyError as e:
            raise KeyError(f"无法从 {specimen_group_key} 加载 'vol/pixels', 'vol/spacing' 或 'vol/origin'。错误: {e}")

        shape_zyx = self.volume_data.shape
        phys_dims_zyx = shape_zyx * self.spacing_zyx
        phys_dims_xyz = np.array([phys_dims_zyx[2], phys_dims_zyx[1], phys_dims_zyx[0]])
        self.C_phys_xyz = self.origin_xyz + (phys_dims_xyz / 2.0)
        self.C_phys_xyz = self.C_phys_xyz.astype(np.float32)
        max_phys_dim = np.max(phys_dims_xyz)
        self.s = 2.0 / max_phys_dim

        print(f"  CT 体积形状 (z,y,x): {shape_zyx}")
        print(f"  CT 间距 (z,y,x): {self.spacing_zyx}")
        print(f"  CT 原点 (x,y,z): {self.origin_xyz}")
        print(f"  归一化中心 C_phys (x,y,z): {self.C_phys_xyz}")
        print(f"  归一化缩放因子 s: {self.s}")
        # --- 归一化参数计算完毕 ---

        # --- [已激活] 2. 加载投影元数据和 3D 地标点 ---
        print("  正在加载投影元数据和 3D 地标点...")
        try:
            # 加载相机内参
            self.intrinsic = torch.from_numpy(self.specimen_group['intrinsic'][:]).float()
            
            # --- [!!! 关键修正 !!!] ---
            # 加载坐标变换矩阵
            # 1. 加载 4x4 矩阵
            matrix_lps2volume = torch.from_numpy(self.specimen_group['lps2volume_matrix'][:]).float()
            # 2. 提取 R (3x3) 和 t (3,) 并传入构造函数
            self.lps2volume = RigidTransform(R=matrix_lps2volume[:3, :3], t=matrix_lps2volume[:3, 3])

            matrix_translate = torch.from_numpy(self.specimen_group['translate_matrix'][:]).float()
            self.translate = RigidTransform(R=matrix_translate[:3, :3], t=matrix_translate[:3, 3])

            matrix_flip_xz = torch.from_numpy(self.specimen_group['flip_xz_matrix'][:]).float()
            self.flip_xz = RigidTransform(R=matrix_flip_xz[:3, :3], t=matrix_flip_xz[:3, 3])
            # --- [修正结束] ---
            
            # 加载 3D 地标点
            vol_landmarks_group = self.specimen_group['vol-landmarks']
            self.landmark_names = list(vol_landmarks_group.keys())
            fiducials_list = [vol_landmarks_group[name][:] for name in self.landmark_names]
            # 形状: (N, 3, 1) -> (N, 3)
            self.fiducials_3d_phys = torch.tensor(np.array(fiducials_list), dtype=torch.float32).squeeze(axis=2) 
            
            print(f"  ✅ 成功加载 {len(self.landmark_names)} 个 3D 地标点和投影矩阵。")
            self.can_project_landmarks = True

        except KeyError as e:
            print(f"  ❌ 警告: 未能加载投影元数据 (例如 {e})。")
            print("     地标点可视化功能将被禁用。")
            self.can_project_landmarks = False
            self.intrinsic = None
            self.lps2volume = None
            self.translate = None
            self.flip_xz = None
            self.fiducials_3d_phys = None


        # --- 3. 创建训练/验证集划分 ---
        self.poses_dset = self.projections_group['poses']
        self.drrs_dset = self.projections_group['drrs']
        
        num_total_images = len(self.poses_dset)
        if num_total_images == 0:
            raise ValueError(f"在 {specimen_group_key}/projections 中未找到图像。")
            
        indices = np.arange(num_total_images)
        rng = np.random.default_rng(random_seed)
        rng.shuffle(indices)
        split_point = int(num_total_images * train_split_ratio)
        
        if self.split == 'train':
            self.indices = indices[:split_point]
            print(f"  加载 'train' 划分, 包含 {len(self.indices)} / {num_total_images} 图像。")
        elif self.split == 'val':
            self.indices = indices[split_point:]
            print(f"  加载 'val' 划分, 包含 {len(self.indices)} / {num_total_images} 图像。")
        else:
            raise ValueError(f"split 参数必须是 'train' 或 'val', 但收到了 '{self.split}'")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        
        # 1. 加载数据
        I_gt_phys = self.drrs_dset[data_idx].copy()
        pose_matrix_phys = self.poses_dset[data_idx].copy()
        
        # 2. 归一化DRR图像 (像素值)
        I_gt_tensor = torch.from_numpy(I_gt_phys).float().unsqueeze(0) # [1, H, W]
        min_val = I_gt_tensor.min()
        max_val = I_gt_tensor.max()
        if max_val - min_val > 1e-6:
            I_gt_tensor = (I_gt_tensor - min_val) / (max_val - min_val)
        else:
            I_gt_tensor = torch.zeros_like(I_gt_tensor)
            
        # 3. 转换位姿为Tensor (返回物理位姿)
        pi_gt_phys_tensor = torch.from_numpy(pose_matrix_phys)
        
        return I_gt_tensor, pi_gt_phys_tensor
        
    def close(self):
        """关闭HDF5文件句柄。"""
        if hasattr(self, 'hf') and self.hf:
            self.hf.close()

    def __del__(self):
        self.close()