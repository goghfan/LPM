import torch
import numpy as np
import imageio
from pathlib import Path
import os

# 导入你项目中用于加载数据的 LPMDataset
#
from data.dataloader import LPMDataset 

# --- 1. 参数配置 ---
# !!! 请修改这些路径以匹配你的设置 !!!

# 指向你的HDF5数据集文件
H5_DATASET_PATH = r"F:\desktop\2D-3D\LPM\LPM\drr_pose_dataset_with_landmarks.h5"
# 你想从中提取数据的样本ID
SPECIMEN_ID = 1
# 你想使用 'train' 还是 'val' 划分
DATA_SPLIT = 'val'
# 你想从该划分中提取第几张图像 (e.g., 0 = 第一张)
IMAGE_INDEX_TO_EXTRACT = 0

# --- 输出文件路径 ---
OUTPUT_IMAGE_PATH = "./live_xray_for_registration.png"
OUTPUT_POSE_PATH = "./ground_truth_pose_for_registration.npy"

# --- 2. 主执行函数 ---
def extract_single_item(h5_path, specimen_id, split, index):
    """
    从HDF5数据集中加载LPMDataset，并提取指定的单个数据项。
    """
    print(f"--- 开始提取数据 ---")
    print(f"数据集: {h5_path}")
    print(f"样本ID: {specimen_id}, 划分: {split}, 索引: {index}")
    
    if not os.path.exists(h5_path):
        print(f"!!! 错误: 找不到HDF5文件: {h5_path}")
        return

    try:
        # 1. 初始化 LPMDataset
        #
        print("正在初始化LPMDataset...")
        dataset = LPMDataset(
            h5_dataset_path=h5_path,
            specimen_id=specimen_id,
            split=split
        )
    except Exception as e:
        print(f"!!! 错误: 初始化LPMDataset失败: {e}")
        return

    if index >= len(dataset):
        print(f"!!! 错误: 索引 {index} 超出范围。'{split}' 划分只有 {len(dataset)} 个样本。")
        dataset.close()
        return

    try:
        # 2. 从数据集中获取数据项
        #
        print(f"正在提取索引为 {index} 的数据...")
        # I_gt_tensor: [1, H, W], pi_gt_phys_tensor: [4, 4]
        I_gt_tensor, pi_gt_phys_tensor = dataset[index]
        
        # 3. 处理并保存图像
        # Dataloader 返回的图像是 [0, 1] 范围的张量
        img_data_numpy = I_gt_tensor.squeeze().cpu().numpy()
        
        # 转换为 0-255 的 uint8 图像
        img_data_uint8 = (img_data_numpy * 255).astype(np.uint8)
        
        imageio.imwrite(OUTPUT_IMAGE_PATH, img_data_uint8)
        print(f"✅ 成功保存X-ray图像至: {OUTPUT_IMAGE_PATH}")

        # 4. 处理并保存位姿
        pose_data_numpy = pi_gt_phys_tensor.cpu().numpy()
        
        np.save(OUTPUT_POSE_PATH, pose_data_numpy)
        print(f"✅ 成功保存4x4物理位姿至: {OUTPUT_POSE_PATH}")
        print("\n--- 提取完成 ---")
        print("最终位姿矩阵内容:")
        print(pose_data_numpy)

    except Exception as e:
        print(f"!!! 错误: 提取或保存数据时失败: {e}")
    finally:
        dataset.close()
        print("HDF5 文件已关闭。")

# --- 3. 主程序入口 ---
if __name__ == "__main__":
    extract_single_item(
        h5_path=H5_DATASET_PATH,
        specimen_id=SPECIMEN_ID,
        split=DATA_SPLIT,
        index=IMAGE_INDEX_TO_EXTRACT
    )

    