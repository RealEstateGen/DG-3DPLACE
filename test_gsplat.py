import torch
import torch.nn as nn
from gsplat import rasterization

def test_raster(render_mode):
    num_gaussians = 2
    means3d = torch.zeros((num_gaussians, 3), device="cuda")
    quats = torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]], device="cuda")
    scales = torch.ones((num_gaussians, 3), device="cuda")
    opacities = torch.ones((num_gaussians, 1), device="cuda")
    colors = torch.ones((num_gaussians, 3), device="cuda")
    
    viewmat = torch.eye(4, device="cuda")
    projmat = torch.eye(4, device="cuda")
    H, W = 64, 64
    fx, fy, cx, cy = 32.0, 32.0, 32.0, 32.0
    
    try:
        renders, alphas, meta = rasterization(
            means3d=means3d,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmat=viewmat,
            projmat=projmat,
            near_plane=0.01,
            far_plane=100.0,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=W,
            height=H,
            render_mode=render_mode
        )
        print(f"\nMode: {render_mode}")
        print(f"Renders: shape={renders.shape}, dtype={renders.dtype}")
        print(f"Alphas: shape={alphas.shape}, dtype={alphas.dtype}")
        print(f"Meta keys: {list(meta.keys())}")
    except Exception as e:
        print(f"Error in {render_mode}: {e}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_raster('ED')
        test_raster('RGB+ED')
    else:
        print("CUDA not available")
