import torch
from pathlib import Path
from tqdm import tqdm
import h5py
import numpy as np

# 假设这些是您项目中的模块
from diffdrr.drr import DRR
from diffpose.deepfluoro import DeepFluoroDataset, get_random_offset
from diffpose.calibration import RigidTransform, perspective_projection

# ===================================================================
# 0. 辅助函数 (已修正)
# ===================================================================
def convert_diffdrr_to_deepfluoro(specimen, pose: RigidTransform):
    """将DiffDRR的相机坐标系转换为DeepFluoro的约定。"""
    return (
        specimen.lps2volume.inverse()
        .compose(pose.inverse())
        .compose(specimen.translate)
        .compose(specimen.flip_xz)
    )

def project_landmarks(specimen, pose):
    """
    计算给定3D位姿下，所有3D地标点的2D投影坐标。
    """
    extrinsic = convert_diffdrr_to_deepfluoro(specimen, pose.cpu())
    projected_coords = perspective_projection(extrinsic, specimen.intrinsic, specimen.fiducials)
    
    landmarks_2d = {}
    
    # 【修正】直接从加载的HDF5对象中获取地标点名称列表
    landmark_names = list(specimen.specimen['vol-landmarks'].keys())
    
    for i, name in enumerate(landmark_names):
        coord = projected_coords[0, i, :].numpy().reshape(2, 1)
        landmarks_2d[name] = coord
        
    return landmarks_2d

# ===================================================================
# 1. load 函数 (无改动)
# ===================================================================
def load(id_number, height, device, h5_datapath):
    """加载数据集、DRR生成器等组件。"""
    specimen = DeepFluoroDataset(id_number, filename=h5_datapath)
    isocenter_pose = specimen.isocenter_pose.to(device)
    subsample = (1536 - 100) / height
    delx = 0.194 * subsample
    drr = DRR(
        specimen.volume, specimen.spacing, specimen.focal_len / 2,
        height, delx, x0=specimen.x0, y0=specimen.y0, reverse_x_axis=True,
    ).to(device)
    return specimen, isocenter_pose, drr

# ===================================================================
# 2. 核心函数：创建HDF5数据集 (已修正)
# ===================================================================
def create_drr_dataset_with_source(ct_ids, num_images_per_ct, height, h5_datapath, output_h5_filename):
    """
    为指定的CT样本生成DRR/位姿/2D地标点，并连同原始3D数据一起保存到HDF5数据集中。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"===================================================")
    print(f"Using device: {device}")
    print(f"Output dataset will be saved to: '{output_h5_filename}'")
    print(f"===================================================\n")

    with h5py.File(output_h5_filename, 'w') as hf_out, h5py.File(h5_datapath, 'r') as hf_in:
        print(f"Successfully created/opened HDF5 files.")

        for id_number in tqdm(ct_ids, desc="Overall Progress (Processing CTs)"):
            print(f"\nProcessing CT with ID: {id_number}...")
            
            try:
                specimen, isocenter_pose, drr = load(id_number, height, device, h5_datapath)
            except Exception as e:
                print(f"\nError: Failed to load data for ID {id_number}. Skipping. Error: {e}")
                continue

            main_group = hf_out.create_group(f'specimen_{id_number}')
            print(f"Created group '/specimen_{id_number}' in output HDF5 file.")

            source_specimen_name = specimen.specimen.name.lstrip('/')
            print(f"Copying source 3D data from group '{source_specimen_name}'...")
            for group_to_copy in ['vol', 'vol-landmarks', 'vol-seg']:
                if group_to_copy in hf_in[source_specimen_name]:
                    source_path = f"{source_specimen_name}/{group_to_copy}"
                    hf_in.copy(source_path, main_group)
                    print(f"  - Copied group '{group_to_copy}'.")

            projections_group = main_group.create_group('projections')
            
            drr_images = []
            poses_matrices = []

            # 【修正】从 specimen.specimen HDF5 对象中获取地标点名称
            landmark_names = list(specimen.specimen['vol-landmarks'].keys())
            projected_landmarks_data = {name: [] for name in landmark_names}

            for _ in tqdm(range(num_images_per_ct), desc=f"Generating data for ID {id_number}"):
                with torch.no_grad():
                    offset = get_random_offset(batch_size=1, device=device)
                    pose = isocenter_pose.compose(offset)
                    
                    img_tensor = drr(None, None, None, pose=pose)
                    pose_matrix = pose.get_matrix().squeeze(0).cpu().numpy()
                    img_numpy = img_tensor.squeeze().cpu().numpy()
                    drr_images.append(img_numpy)
                    poses_matrices.append(pose_matrix)
                    
                    landmarks_2d = project_landmarks(specimen, pose)
                    for name, coord in landmarks_2d.items():
                        projected_landmarks_data[name].append(coord)

            projections_group.create_dataset('drrs', data=np.array(drr_images, dtype=np.float32))
            projections_group.create_dataset('poses', data=np.array(poses_matrices, dtype=np.float32))

            landmarks_group = projections_group.create_group('projected_landmarks')
            for name, coords_list in projected_landmarks_data.items():
                landmarks_group.create_dataset(name, data=np.array(coords_list, dtype=np.float32))
            
            print(f"\nSuccessfully saved {len(drr_images)} DRRs, poses, and projected landmarks for ID {id_number}.")

    print("\n===================================================")
    print("Dataset generation complete!")
    print(f"All data saved in '{output_h5_filename}'")
    print("===================================================")

# ===================================================================
# 3. 主程序入口
# ===================================================================
if __name__ == "__main__":
    CT_IDS_TO_PROCESS = range(1, 7)
    NUM_IMAGES_PER_CT = 4000
    IMAGE_HEIGHT = 256
    
    ORIGINAL_DATA_PATH = r"F:\desktop\2D-3D\LPM\LPM\data\ipcai_2020_full_res_data\ipcai_2020_full_res_data.h5"
    OUTPUT_H5_FILE = r"F:\desktop\2D-3D\LPM\LPM\data\ipcai_2020_full_res_data\drr_pose_dataset_with_landmarks123123.h5"
    
    create_drr_dataset_with_source(
        ct_ids=CT_IDS_TO_PROCESS,
        num_images_per_ct=NUM_IMAGES_PER_CT,
        height=IMAGE_HEIGHT,
        h5_datapath=ORIGINAL_DATA_PATH,
        output_h5_filename=OUTPUT_H5_FILE
    )