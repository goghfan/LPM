import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import imageio

# --- 1. 环境与参数设置 (Hyperparameters) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# 图像与相机参数
H, W = 128, 128      # DRR图像的分辨率
FOCAL = 180.0        # 相机焦距，决定了视场角（FOV），数值越大FOV越小，物体看起来越大

# 渲染参数
N_SAMPLES = 64       # 每条射线上的采样点数量。越多，渲染越精细，但计算量越大
NEAR_THRESH = 1.5    # 射线开始采样的最近距离
FAR_THRESH = 3.5     # 射线停止采样的最远距离
                     # 这两者定义了一个三维空间中的“视锥体切片”，只有在这个范围内的物体才会被渲染
I0 = 1.0             # X射线的初始强度，物理意义上的 I₀

# 训练参数
ITERS = 2000         # 总训练迭代次数
LR = 5e-4            # 学习率 (Learning Rate)
BATCH_SIZE = 4096    # 每次迭代中随机采样的射线数量。越大，梯度越稳定，但显存占用越高


# 输出目录
os.makedirs("output_minimal_demo", exist_ok=True)


# --- 2. 解析体（Ground Truth）定义 ---
# 我们在 3D 空间中定义一个球体
sphere_radius = 0.5
sphere_center = torch.tensor([0.0, 0.0, 2.5], device=DEVICE)
sphere_density = 15.0  # 球体内部的线性衰减系数值

def analytical_sphere(x: torch.Tensor) -> torch.Tensor:
    """
    对于给定的 3D 点 x，判断其是否在球体内，并返回其密度。
    Input:
      x: shape [N, 3], 3D 坐标点
    Output:
      rho: shape [N, 1], 每个点的密度值
    """
    # 计算所有输入点 x 到球心的距离
    dists = torch.linalg.norm(x - sphere_center, dim=-1, keepdim=True)
    # 创建一个和输入形状相同的全零密度张量
    rho = torch.zeros_like(dists)
    # 对于所有距离小于半径的点，将其密度设为预定义的值
    rho[dists < sphere_radius] = sphere_density
    return rho


# --- 3. 核心模块：MLP、位置编码、射线工具 ---

class PositionalEncoder(nn.Module):
    """Sinusoidal Positional Encoder for 3D coordinates."""
    def __init__(self, num_freqs: int):
        super().__init__()
        # CORRECTED: Register 'freqs' as a buffer. It will now be moved to the correct
        # device automatically when you call .to(DEVICE) on the module.
        # 注册一个名为'freqs'的buffer。Buffer是模型的一部分，会随模型移动到GPU，但不是可训练参数
        self.register_buffer(
            'freqs',
            2.**torch.arange(num_freqs) * torch.pi
        )
    # NeRF 中的一个关键技术。直接将 (x, y, z) 坐标输入MLP，网络很难学到物体的精细边缘和纹理。位置编码通过一系列不同频率的sin 和 cos 函数，将一个3维坐标向量映射成一个更高维的特征向量。这极大地增强了MLP表达高频细节的能力。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The input 'x' is already on the correct device, and now so is 'self.freqs'.
        # No need to move tensors around here.
        scaled_x = x.unsqueeze(-1) * self.freqs # [N, 3, F]
        encoded_x = torch.cat([torch.sin(scaled_x), torch.cos(scaled_x)], dim=-1) # [N, 3, 2F]
        return encoded_x.flatten(start_dim=1) # [N, 6F]


class DiffRadMLP(nn.Module):
    """
    f_theta(x) -> (rho, f)
    MLP to represent the implicit volume.
    """
    def __init__(self, pos_embed_dim: int, feat_dim: int = 16, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pos_embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.density_head = nn.Linear(hidden_dim, 1) # Output: pre-activation scalar 's'
        self.feature_head = nn.Linear(hidden_dim, feat_dim) # Output: feature vector 'f'

    def forward(self, x_embedded: torch.Tensor):
        hidden = self.net(x_embedded)
        s = self.density_head(hidden)
        # 保证 rho >= 0，使用 softplus，如你文档中所述
        rho = F.softplus(s)
        f = self.feature_head(hidden)
        return rho, f

def get_rays(H: int, W: int, focal: float, c2w: torch.Tensor):
    """
    生成相机坐标系下的射线
    c2w: camera to world transformation matrix [4, 4]
    """
    # 1. 创建像素网格坐标 (i, j)
    # 2. 根据焦距，将像素坐标转换为相机坐标系下的方向向量 dirs
    # 3. 使用 c2w (Camera-to-World) 矩阵，将相机坐标系下的射线原点和方向，变换到世界坐标系

    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=DEVICE),
        torch.arange(H, dtype=torch.float32, device=DEVICE),
        indexing='xy'
    )
    # 像素坐标到相机坐标
    dirs = torch.stack([
        (i - W * .5) / focal,
        -(j - H * .5) / focal,
        -torch.ones_like(i)
    ], dim=-1)
    # 相机坐标系到世界坐标系
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


# --- 4. 可微渲染器 (Differentiable Renderer) ---

def render_rays(
    model: nn.Module,
    pos_encoder: PositionalEncoder,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    n_samples: int,
    use_analytical_gt: bool = False
):
    """
    对一批射线进行渲染
    """
    # 1. 沿射线采样点 (Ray Marching)
    #    在[near, far]范围内，均匀取N_SAMPLES个点

    t_vals = torch.linspace(0., 1., n_samples, device=DEVICE)
    z_vals = near * (1. - t_vals) + far * t_vals
    # delta_t, 步长
    delta_t = z_vals[1:] - z_vals[:-1]
    delta_t = torch.cat([delta_t, delta_t[-1].unsqueeze(0)], dim=-1) # [n_samples]  # 计算每两个采样点之间的距离

    # [batch_size, n_samples, 3]
    pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1)# 计算采样点的3D坐标

    # 2. 查询密度值
    if use_analytical_gt:
        rho = analytical_sphere(pts)
    else:
        pts_embedded = pos_encoder(pts.view(-1, 3))
        rho, _ = model(pts_embedded)
        rho = rho.view(pts.shape[0], pts.shape[1], 1)

    # 3. 计算 Beer-Lambert 积分
    # 3. 数值积分 (Numerical Integration) - 核心物理模型
    #    这就是你文档中的离散化公式： Σ ρ_k * Δt_k
    integral = (rho * delta_t.unsqueeze(0).unsqueeze(-1)).sum(dim=1) # [batch_size, 1]

    # 4. 应用比尔-朗伯定律 (Beer-Lambert Law)
    #    这就是你文档中的公式： I_DRR = I_0 * exp(-Σ ρ_k * Δt_k)
    pixel_intensity = I0 * torch.exp(-integral)

    return pixel_intensity


# --- 5. 主流程 ---

def main():
    # 初始化模型
    pos_encoder = PositionalEncoder(num_freqs=10).to(DEVICE)
    model = DiffRadMLP(pos_embed_dim=10 * 2 * 3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 定义相机位姿 (固定，用于生成目标和训练)
    # 这里我们让相机位于原点，看向 Z 轴正方向
    # c2w = torch.eye(4, device=DEVICE)
    # NEW - Rotate camera to look along +Z
    c2w = torch.tensor([
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32, device=DEVICE)
    # 生成所有射线的原点和方向
    rays_o_all, rays_d_all = get_rays(H, W, FOCAL, c2w)
    rays_o_all = rays_o_all.view(-1, 3)
    rays_d_all = rays_d_all.view(-1, 3)

    # a) 生成基准真相 (Ground Truth) DRR 图像
    print("Generating ground truth DRR...")
    with torch.no_grad():
        gt_chunks = []
        for i in range(0, H * W, BATCH_SIZE):
            chunk_o = rays_o_all[i:i + BATCH_SIZE]
            chunk_d = rays_d_all[i:i + BATCH_SIZE]
            pixel_chunk = render_rays(
                model=None, pos_encoder=None,
                rays_o=chunk_o, rays_d=chunk_d,
                near=NEAR_THRESH, far=FAR_THRESH, n_samples=N_SAMPLES,
                use_analytical_gt=True
            )
            gt_chunks.append(pixel_chunk)
        
        target_drr = torch.cat(gt_chunks, dim=0).view(H, W)
        
    # 保存 Ground Truth 图像
    gt_img_np = target_drr.cpu().numpy()
    gt_img_np = (gt_img_np * 255).astype(np.uint8)
    imageio.imwrite("output_minimal_demo/00_ground_truth_drr.png", gt_img_np)
    print("Ground truth DRR saved.")

    # b) 训练 MLP
    print("Starting training...")
    model.train()
    all_pixels_gt = target_drr.view(-1, 1)

    for i in range(ITERS):
        # 随机采样一批射线
        indices = torch.randint(0, H * W, (BATCH_SIZE,))
        batch_rays_o = rays_o_all[indices]
        batch_rays_d = rays_d_all[indices]
        batch_pixels_gt = all_pixels_gt[indices]

        # 渲染
        rendered_pixels = render_rays(
            model=model, pos_encoder=pos_encoder,
            rays_o=batch_rays_o, rays_d=batch_rays_d,
            near=NEAR_THRESH, far=FAR_THRESH, n_samples=N_SAMPLES
        )

        # 计算损失 (L1 Loss, 如你文档所述)
        loss = F.l1_loss(rendered_pixels, batch_pixels_gt)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Iter: {i}, Loss: {loss.item():.6f}")

        # 每 200 步，渲染并保存一张完整图像以供可视化
        if i % 200 == 0 or i == ITERS - 1:
            model.eval()
            with torch.no_grad():
                rendered_chunks = []
                for j in range(0, H * W, BATCH_SIZE):
                    chunk_o = rays_o_all[j:j + BATCH_SIZE]
                    chunk_d = rays_d_all[j:j + BATCH_SIZE]
                    pixel_chunk = render_rays(
                        model=model, pos_encoder=pos_encoder,
                        rays_o=chunk_o, rays_d=chunk_d,
                        near=NEAR_THRESH, far=FAR_THRESH, n_samples=N_SAMPLES
                    )
                    rendered_chunks.append(pixel_chunk)
                
                rendered_drr = torch.cat(rendered_chunks, dim=0).view(H, W)
                
                img_np = rendered_drr.cpu().numpy()
                img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
                imageio.imwrite(f"output_minimal_demo/iter_{i:04d}.png", img_np)
            model.train()
    
    # 创建一个 GIF 动图展示训练过程
    images = []
    # Use a sorted list of filenames to ensure correct order
    fnames = sorted([f for f in os.listdir("output_minimal_demo") if f.endswith(".png") and "iter" in f])
    for filename in fnames:
        images.append(imageio.imread(os.path.join("output_minimal_demo", filename)))
    imageio.mimsave('output_minimal_demo/training_progress.gif', images, fps=5)

    print("Training finished. Check the 'output_minimal_demo' directory.")


if __name__ == '__main__':
    main()