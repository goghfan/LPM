import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import logm

# ==============================================================================
# 步驟 2: 實現 NeRF 風格的位置編碼器 (PoseEncoder)
# ==============================================================================

class PoseEncoder(nn.Module):
    """
    對一個 (B, 6) 的位姿向量進行 NeRF 風格的位置編碼。
    """
    def __init__(self, num_freqs: int = 10, include_input: bool = True):
        """
        初始化位置編碼器。
        :param num_freqs: L, 要使用的頻率數量。NeRF 論文中對坐標使用 10。
        :param include_input: 是否在最後拼接原始的6維輸入向量。
        """
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        # 創建頻率帶 [1, 2, 4, 8, ..., 2^(L-1)]
        self.freq_bands = 2.**torch.linspace(0., num_freqs - 1, num_freqs)
        
        # 計算最終輸出的維度
        # 每個維度 -> num_freqs * 2 (sin/cos)
        self.output_dim = 6 * self.num_freqs * 2
        if self.include_input:
            self.output_dim += 6
        print(f"位置編碼器已初始化。輸入維度: 6, 輸出維度: {self.output_dim}")

    def forward(self, pose_vec: torch.Tensor) -> torch.Tensor:
        """
        對輸入的位姿向量進行編碼。
        輸入: pose_vec, shape=(B, 6)
        輸出: encoded_vec, shape=(B, self.output_dim)
        """
        # (B, 6) -> (B, 6, 1) -> (B, 6, num_freqs)
        # 讓每個位姿向量元素與所有頻率相乘
        scaled_inputs = pose_vec.unsqueeze(-1) * self.freq_bands
        
        # 計算 sin 和 cos
        # (B, 6, num_freqs) -> (B, 6, num_freqs * 2)
        encoded = torch.cat([torch.sin(scaled_inputs), torch.cos(scaled_inputs)], dim=-1)
        
        # (B, 6, num_freqs * 2) -> (B, 6 * num_freqs * 2)
        # 展平成最終的特徵向量
        encoded = encoded.flatten(start_dim=1)
        
        if self.include_input:
            # 拼接原始輸入
            encoded = torch.cat([pose_vec, encoded], dim=-1)
            
        return encoded

# ==============================================================================
# 步驟 1: 實現 4x4 矩陣到 6維 se(3) 向量的轉換
# ==============================================================================

def matrix_to_se3_log_map(matrix: np.ndarray) -> np.ndarray:
    """
    將一個 4x4 的 SE(3) 齊次變換矩陣轉換為 6 維的 se(3) 李代數向量。
    :param matrix: 4x4 的 NumPy 數組
    :return: 6 維的 NumPy 向量 [wx, wy, wz, tx, ty, tz]
    """
    if matrix.shape != (4, 4):
        raise ValueError("輸入必須是一個 4x4 的矩陣")

    # 使用 scipy 的矩陣對數函數計算李代數矩陣
    # logm(T) 的結果是一個 4x4 的 se(3) 矩陣
    se3_matrix = logm(matrix)
    
    # 從 se(3) 矩陣中提取 6 維向量
    # se(3) 矩陣的形式為:
    # [[ 0, -wz,  wy, tx],
    #  [ wz,   0, -wx, ty],
    #  [-wy,  wx,   0, tz],
    #  [  0,   0,   0,  0]]
    
    # 提取平移部分
    translation = se3_matrix[:3, 3]
    
    # 提取旋轉部分 (注意反對稱矩陣的符號)
    wz = se3_matrix[1, 0]
    wy = se3_matrix[0, 2]
    wx = se3_matrix[2, 1]
    
    rotation = np.array([wx, wy, wz])
    
    # 拼接成 6 維向量 [rotation, translation]
    return np.concatenate([rotation, translation])


# ==============================================================================
# 主程序：演示整個流程
# ==============================================================================

if __name__ == "__main__":
    # --- 創建一個範例的 4x4 位姿矩陣 ---
    # 假設有一個繞 Z 軸旋轉 15 度，並沿 X,Y,Z 平移的位姿
    from scipy.spatial.transform import Rotation
    
    # 1. 創建旋轉部分
    rotation_matrix = Rotation.from_euler('z', 15, degrees=True).as_matrix()
    
    # 2. 創建平移部分
    translation_vector = np.array([10.0, -5.0, 20.0])
    
    # 3. 組合成 4x4 齊次變換矩陣
    pose_matrix_4x4 = np.eye(4)
    pose_matrix_4x4[:3, :3] = rotation_matrix
    pose_matrix_4x4[:3, 3] = translation_vector
    
    print("===================== 輸入 =====================")
    print("原始的 4x4 位姿矩陣:\n", pose_matrix_4x4)
    print("\n")
    
    # --- 步驟 1: 將 4x4 矩陣轉換為 6維 se(3) 向量 ---
    print("========== 步驟 1: 4x4 -> 6D 向量 ==========")
    pose_vector_6d = matrix_to_se3_log_map(pose_matrix_4x4)
    print("轉換後的 6 維 se(3) 向量 [rot, trans]:\n", pose_vector_6d)
    print("\n")

    # --- 步驟 2: 對 6維 向量進行 NeRF 風格的位置編碼 ---
    print("======== 步驟 2: 6D -> 高維特徵向量 ========")
    
    # 初始化編碼器
    encoder = PoseEncoder(num_freqs=10, include_input=True)
    
    # 將 numpy 向量轉換為 torch 張量，並增加一個批次維度 (Batch size = 1)
    pose_tensor = torch.from_numpy(pose_vector_6d).unsqueeze(0).float()
    
    # 進行編碼
    encoded_pose_feature = encoder(pose_tensor)
    
    print(f"\n輸入到編碼器的張量形狀: {pose_tensor.shape}")
    print(f"經過位置編碼後輸出的特徵向量形狀: {encoded_pose_feature.shape}")
    print("編碼後的高維特徵向量 (只顯示前10個元素):\n", encoded_pose_feature)