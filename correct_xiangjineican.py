import h5py
import torch
import numpy as np
from tqdm import tqdm

# 确保您可以导入 DeepFluoroDataset
# (它只需要 load_deepfluoro_dataset 函数)
from diffpose.deepfluoro import DeepFluoroDataset

# ===================================================================
# 1. 辅助函数：仅加载元数据
# ===================================================================
def load_metadata_only(id_number, h5_datapath):
    """
    轻量级加载器：只加载 DeepFluoroDataset 并获取元数据。
    """
    print(f"  - 正在从 '{h5_datapath}' 加载 specimen_{id_number} 的原始元数据...")
    try:
        # 我们只初始化类以获取内存中的矩阵
        # DeepFluoroDataset 在 __init__ 时会加载所有需要的东西
        specimen = DeepFluoroDataset(id_number, filename=h5_datapath)
        return specimen
    except Exception as e:
        print(f"    - 错误: 无法加载原始 specimen {id_number}。错误: {e}")
        return None

# ===================================================================
# 2. 核心函数：修补 HDF5 文件
# ===================================================================
def patch_hdf5_with_metadata(ct_ids, original_h5_path, target_h5_path):
    """
    打开现有的 HDF5 文件并添加缺失的元数据 (intrinsic 等)。
    """
    print(f"===================================================")
    print(f"开始修补文件: '{target_h5_path}'")
    print(f"元数据来源: '{original_h5_path}'")
    print(f"===================================================\n")

    try:
        # [关键] 以 'a' (append) 模式打开您的 *目标* 文件
        with h5py.File(target_h5_path, 'a') as hf_target:
            
            for id_number in tqdm(ct_ids, desc="修补进度 (Processing CTs)"):
                print(f"\n正在处理 specimen_{id_number}...")
                
                group_name = f'specimen_{id_number}'
                if group_name not in hf_target:
                    print(f"  - 错误: 在 '{target_h5_path}' 中未找到组 '{group_name}'。跳过。")
                    continue
                
                main_group = hf_target[group_name]

                # --- 检查是否已修补 ---
                if 'intrinsic' in main_group:
                    print(f"  - 'intrinsic' 矩阵已存在。假定已修补，跳过。")
                    continue
                
                # --- 加载原始元数据 ---
                specimen = load_metadata_only(id_number, original_h5_path)
                if specimen is None:
                    continue
                    
                print(f"  - 正在写入缺失的元数据到 '{group_name}'...")

                # --- [!!! 关键：写入数据 !!!] ---
                try:
                    if hasattr(specimen, 'intrinsic'):
                        main_group.create_dataset('intrinsic', data=specimen.intrinsic.cpu().numpy())
                        print(f"    - 已写入 'intrinsic' 矩阵。")
                    
                    if hasattr(specimen, 'lps2volume'):
                        main_group.create_dataset('lps2volume_matrix', data=specimen.lps2volume.get_matrix().squeeze().cpu().numpy())
                        print(f"    - 已写入 'lps2volume_matrix'。")
                        
                    if hasattr(specimen, 'translate'):
                        main_group.create_dataset('translate_matrix', data=specimen.translate.get_matrix().squeeze().cpu().numpy())
                        print(f"    - 已写入 'translate_matrix'。")
                        
                    if hasattr(specimen, 'flip_xz'):
                        main_group.create_dataset('flip_xz_matrix', data=specimen.flip_xz.get_matrix().squeeze().cpu().numpy())
                        print(f"    - 已写入 'flip_xz_matrix'。")
                        
                    print(f"  - specimen_{id_number} 修补成功！")

                except Exception as e:
                    print(f"  - [!!!] 写入 HDF5 时发生错误: {e}")

    except Exception as e:
        print(f"\n[!!!] 无法打开目标文件 '{target_h5_path}'。错误: {e}")

    print("\n===================================================")
    print("HDF5 文件修补完成！")
    print("===================================================")

# ===================================================================
# 3. 主程序入口
# ===================================================================
if __name__ == "__main__":
    CT_IDS_TO_PROCESS = range(1, 7)
    
    # [重要] 这是您原始的、包含 'proj-params' 的数据文件
    ORIGINAL_DATA_PATH = r"F:\desktop\2D-3D\LPM\LPM\data\ipcai_2020_full_res_data\ipcai_2020_full_res_data.h5"
    
    # [重要] 这是您 *现有的*、需要被 *修补* 的文件
    TARGET_H5_FILE = r"F:\desktop\2D-3D\LPM\LPM\drr_pose_dataset_with_landmarks.h5"
    
    patch_hdf5_with_metadata(
        ct_ids=CT_IDS_TO_PROCESS,
        original_h5_path=ORIGINAL_DATA_PATH,
        target_h5_path=TARGET_H5_FILE
    )