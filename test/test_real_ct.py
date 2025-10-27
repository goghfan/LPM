import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import imageio
import SimpleITK as sitk

# --- 1. 环境与参数设置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

H, W = 256, 256
FOCAL = 350.0

N_SAMPLES = 128
I0 = 1.0

ITERS = 5000
LR = 1e-4
BATCH_SIZE = 4096

os.makedirs("output_ct_demo", exist_ok=True)

# --- 2. CT数据加载与预处理 ---
CT_FILE_PATH = "F:\\lung_vesscel\\imagesTr\\PA000053_0000.nii.gz" # <-- 把你的路径放在这里

def load_and_preprocess_ct(path: str):
    print(f"Loading CT from {path}...")
    sitk_img = sitk.ReadImage(path, sitk.sitkFloat32)
    ct_np = sitk.GetArrayFromImage(sitk_img)
    
    hu_min, hu_max = -1000., 1000.
    ct_np = np.clip(ct_np, hu_min, hu_max)

    density_min, density_max = 0.0, 1.0
    ct_np = (ct_np - hu_min) / (hu_max - hu_min) * (density_max - density_min) + density_min
    
    ct_tensor = torch.from_numpy(ct_np).float().unsqueeze(0).unsqueeze(0)
    print(f"CT loaded. Shape: {ct_tensor.shape}, Density range: [{ct_tensor.min():.2f}, {ct_tensor.max():.2f}]")
    
    return ct_tensor.to(DEVICE)

# --- 3. CT体数据采样器 ---
def sample_from_ct_grid(pts: torch.Tensor, ct_volume: torch.Tensor):
    sample_coords = pts.view(1, 1, 1, -1, 3)
    sampled_rho = F.grid_sample(
        ct_volume, 
        sample_coords, 
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )
    return sampled_rho.view(-1, 1)

# --- 核心模块 (PositionalEncoder, DiffRadMLP, get_rays) ---
class PositionalEncoder(nn.Module):
    def __init__(self, num_freqs: int):
        super().__init__()
        self.register_buffer('freqs', 2.**torch.arange(num_freqs) * torch.pi)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_x = x.unsqueeze(-1) * self.freqs
        encoded_x = torch.cat([torch.sin(scaled_x), torch.cos(scaled_x)], dim=-1)
        return encoded_x.flatten(start_dim=1)

class DiffRadMLP(nn.Module):
    def __init__(self, pos_embed_dim: int, feat_dim: int = 16, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pos_embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.density_head = nn.Linear(hidden_dim, 1)
        self.feature_head = nn.Linear(hidden_dim, feat_dim)
    def forward(self, x_embedded: torch.Tensor):
        hidden = self.net(x_embedded)
        s = self.density_head(hidden)
        rho = F.softplus(s)
        f = self.feature_head(hidden)
        return rho, f

def get_rays(H: int, W: int, focal: float, c2w: torch.Tensor):
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32, device=DEVICE), torch.arange(H, dtype=torch.float32, device=DEVICE), indexing='xy')
    dirs = torch.stack([(i - W * .5) / focal, -(j - H * .5) / focal, -torch.ones_like(i)], dim=-1)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d

# --- 4. 渲染器 ---
def render_rays(
    rays_o: torch.Tensor, rays_d: torch.Tensor,
    near: float, far: float, n_samples: int,
    use_ct_gt: bool = False,
    ct_volume: torch.Tensor = None,
    model: nn.Module = None,
    pos_encoder: PositionalEncoder = None,
):
    t_vals = torch.linspace(0., 1., n_samples, device=DEVICE)
    z_vals = near + (far - near) * t_vals
    delta_t = z_vals[1:] - z_vals[:-1]
    delta_t = torch.cat([delta_t, delta_t[-1].unsqueeze(0)], dim=-1)
    
    pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1)

    if use_ct_gt:
        if ct_volume is None: raise ValueError("ct_volume must be provided for ground truth rendering.")
        rho = sample_from_ct_grid(pts, ct_volume)
        rho = rho.view(pts.shape[0], pts.shape[1], 1)
    else:
        if model is None or pos_encoder is None: raise ValueError("model and pos_encoder must be provided.")
        pts_embedded = pos_encoder(pts.view(-1, 3))
        rho, _ = model(pts_embedded)
        rho = rho.view(pts.shape[0], pts.shape[1], 1)

    integral = (rho * delta_t.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
    pixel_intensity = I0 * torch.exp(-integral)

    return pixel_intensity

# --- 5. 主流程 ---
def main():
    ct_volume = load_and_preprocess_ct(CT_FILE_PATH)
    pos_encoder = PositionalEncoder(num_freqs=10).to(DEVICE)
    model = DiffRadMLP(pos_embed_dim=10 * 2 * 3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    camera_dist = 2.5
    c2w = torch.tensor([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, camera_dist],
        [0, 0, 0, 1]
    ], dtype=torch.float32, device=DEVICE)
    
    render_near = camera_dist - 1.2
    render_far = camera_dist + 1.2

    rays_o_all, rays_d_all = get_rays(H, W, FOCAL, c2w)
    rays_o_all, rays_d_all = rays_o_all.view(-1, 3), rays_d_all.view(-1, 3)

    print("Generating ground truth DRR from CT volume...")
    with torch.no_grad():
        gt_chunks = []
        for i in range(0, H * W, BATCH_SIZE):
            chunk_o, chunk_d = rays_o_all[i:i+BATCH_SIZE], rays_d_all[i:i+BATCH_SIZE]
            pixel_chunk = render_rays(
                rays_o=chunk_o, rays_d=chunk_d,
                near=render_near, far=render_far, n_samples=N_SAMPLES,
                use_ct_gt=True, ct_volume=ct_volume
            )
            gt_chunks.append(pixel_chunk)
        target_drr = torch.cat(gt_chunks, dim=0).view(H, W)
        
    gt_img_np = (np.clip(target_drr.cpu().numpy(), 0, 1) * 255).astype(np.uint8)
    imageio.imwrite("output_ct_demo/00_ground_truth_drr_from_ct.png", gt_img_np)
    print("Ground truth DRR saved.")

    print("Starting training MLP to reconstruct the DRR...")
    model.train()
    all_pixels_gt = target_drr.view(-1, 1)

    for i in range(ITERS):
        indices = torch.randint(0, H * W, (BATCH_SIZE,))
        batch_rays_o, batch_rays_d = rays_o_all[indices], rays_d_all[indices]
        batch_pixels_gt = all_pixels_gt[indices]

        rendered_pixels = render_rays(
            model=model, pos_encoder=pos_encoder,
            rays_o=batch_rays_o, rays_d=batch_rays_d,
            near=render_near, far=render_far, n_samples=N_SAMPLES
        )
        loss = F.l1_loss(rendered_pixels, batch_pixels_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Iter: {i}, Loss: {loss.item():.6f}")

        if i % 500 == 0 or i == ITERS - 1:
            model.eval()
            with torch.no_grad():
                rendered_chunks = []
                # ##################################################################
                # ## THE FIX IS HERE ##
                # ##################################################################
                for j in range(0, H * W, BATCH_SIZE):
                    chunk_o = rays_o_all[j:j + BATCH_SIZE]
                    chunk_d = rays_d_all[j:j + BATCH_SIZE]
                # ##################################################################
                    pixel_chunk = render_rays(
                        model=model, pos_encoder=pos_encoder,
                        rays_o=chunk_o, rays_d=chunk_d,
                        near=render_near, far=render_far, n_samples=N_SAMPLES
                    )
                    rendered_chunks.append(pixel_chunk)
                rendered_drr = torch.cat(rendered_chunks, dim=0).view(H, W)
                img_np = (np.clip(rendered_drr.cpu().numpy(), 0, 1) * 255).astype(np.uint8)
                imageio.imwrite(f"output_ct_demo/iter_{i:04d}.png", img_np)
            model.train()
    
    images = []
    fnames = sorted([f for f in os.listdir("output_ct_demo") if f.endswith(".png") and "iter" in f])
    for filename in fnames: images.append(imageio.imread(os.path.join("output_ct_demo", filename)))
    imageio.mimsave('output_ct_demo/training_progress.gif', images, fps=5)
    print("Training finished. Check the 'output_ct_demo' directory.")

if __name__ == '__main__':
    main()