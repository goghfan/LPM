import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# -----------------------------------------------------------------
# 1. 假设 'diffdrr' 和 'diffpose' 库已在您的环境中安装
# -----------------------------------------------------------------
try:
    from diffpose.calibration import RigidTransform
except ImportError:
    print("错误：无法导入 'diffpose' 库。")
    print("请确保您已在Python环境中正确安装了 'diffpose'。")
    exit()

# -----------------------------------------------------------------
# 2. 假设 'get_deepfluoro_4000.py' 在同一目录下，以便导入 'load' 函数
# -----------------------------------------------------------------
try:
    # 导入 'load' 函数，该函数负责加载CT数据和DRR渲染器
    from get_deepfluoro_4000 import load
except ImportError:
    print("错误：无法从 'get_deepfluoro_4000.py' 导入 'load' 函数。")
    print("请确保此脚本与 'get_deepfluoro_4000.py' 在同一个文件夹中。")
    exit()

# -----------------------------------------------------------------
# 3. 定义文件路径 (请根据您的设置修改)
# -----------------------------------------------------------------
# 这是您在 'get_deepfluoro_4000.py' 中生成的H5文件
GENERATED_H5_PATH = r"F:\desktop\2D-3D\LPM\LPM\drr_pose_dataset_with_landmarks.h5"

# 这是 'load' 函数需要的原始H5数据文件
# (路径来自 'get_deepfluoro_4000.py' 脚本)
ORIGINAL_DATA_PATH = r"F:\desktop\2D-3D\LPM\LPM\data\ipcai_2020_full_res_data\ipcai_2020_full_res_data.h5"

# -----------------------------------------------------------------
# 4. 验证脚本主逻辑
# -----------------------------------------------------------------
def verify_pose():
    print(f"正在打开生成的H5文件: {GENERATED_H5_PATH}")
    
    if not os.path.exists(GENERATED_H5_PATH):
        print(f"错误：找不到文件: {GENERATED_H5_PATH}")
        return

    if not os.path.exists(ORIGINAL_DATA_PATH):
        print(f"错误：找不到原始数据文件: {ORIGINAL_DATA_PATH}")
        print("请确保 'ORIGINAL_DATA_PATH' 变量设置正确。")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    with h5py.File(GENERATED_H5_PATH, 'r') as hf:
        # 随机选择一个样本
        specimen_keys = [key for key in hf.keys() if key.startswith('specimen_')]
        if not specimen_keys:
            print("错误：H5文件中没有找到 'specimen_' 组。")
            return
            
        selected_key = random.choice(specimen_keys)
        specimen_group = hf[selected_key]
        specimen_id = int(selected_key.split('_')[-1])
        print(f"已随机选择样本: {selected_key} (ID: {specimen_id})")

        # 随机选择一个索引
        num_images = len(specimen_group['projections/drrs'])
        random_index = random.randint(0, num_images - 1)
        print(f"已随机选择索引: {random_index}")

        # -----------------
        # A. 从H5文件读取预生成的DRR和位姿
        # -----------------
        print("A. 正在从H5文件读取数据...")
        
        # 读取预生成的DRR
        pre_generated_drr = specimen_group['projections/drrs'][random_index]
        
        # 读取对应的位姿矩阵 (Numpy 数组)
        pose_matrix_numpy = specimen_group['projections/poses'][random_index]
        
        print(f"  - 预生成DRR的形状: {pre_generated_drr.shape}")
        print(f"  - 位姿矩阵 (Numpy):\n{pose_matrix_numpy}")

        # 保存预生成的DRR图像
        plt.imsave('pre_generated_drr.png', pre_generated_drr, cmap='gray')
        print("  - 已保存 'pre_generated_drr.png'")

        # -----------------
        # B. 使用该位姿重新渲染DRR
        # -----------------
        print("B. 正在加载原始CT数据并重新渲染...")
        
        # 1. 加载DRR渲染器 (使用 'get_deepfluoro_4000.py' 中的 'load' 函数)
        #    注意：'height' 必须与生成DRR时的高度匹配
        height = pre_generated_drr.shape[0] 
        print(f"  - 使用高度 {height} 加载 'load' 函数...")
        specimen, isocenter_pose, drr_renderer = load(
            id_number=specimen_id,
            height=height,
            device=device,
            h5_datapath=ORIGINAL_DATA_PATH
        )

        # 2. 将Numpy位姿矩阵转换为DiffDRR可用的 'RigidTransform' 对象
        #    我们直接使用从H5读取的矩阵，不进行任何坐标转换
        pose_tensor = torch.tensor(pose_matrix_numpy, dtype=torch.float32, device=device).unsqueeze(0)
        
        # RigidTransform 需要分别传入旋转矩阵 R 和平移向量 t
        # 我们从 4x4 的位姿张量中提取这些分量
        # 根据 Pytorch3D 的约定，存储的矩阵是 [R.T | t]^T 的形式
        # 所以 R 是左上 3x3 块的转置, t 是第4行的前3个元素
        pose_object = RigidTransform(
            R=pose_tensor[:, :3, :3].transpose(-1, -2),
            t=pose_tensor[:, 3, :3],
        )

        # 3. 使用该位姿对象渲染新的DRR
        print("  - 正在使用加载的位姿进行渲染...")
        with torch.no_grad():
            newly_rendered_drr_tensor = drr_renderer(
                None, None, None, pose=pose_object
            )
        
        # 4. 将新DRR转为Numpy并保存
        newly_rendered_drr = newly_rendered_drr_tensor.squeeze().cpu().numpy()
        plt.imsave('newly_rendered_drr.png', newly_rendered_drr, cmap='gray')
        print("  - 已保存 'newly_rendered_drr.png'")

    print("\n===================================================")
    print("验证完成！")
    print("请对比 'pre_generated_drr.png' 和 'newly_rendered_drr.png'。")
    print("如果两张图完全一致，则证明H5中的位姿矩阵就是DiffDRR可直接使用的位姿。")
    print("===================================================")


if __name__ == "__main__":
    verify_pose()