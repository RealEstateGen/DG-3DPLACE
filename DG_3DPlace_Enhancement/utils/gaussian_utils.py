"""
Gaussian Splatting specific utilities
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional


def load_ckpt_gaussians(ckpt_path: str) -> Dict:
    """
    Load Gaussian parameters from checkpoint.
    
    Args:
        ckpt_path: Path to checkpoint file
        
    Returns:
        Dictionary with Gaussian parameters
    """
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        gaussians = {
            'means': ckpt.get('means', None),
            'scales': ckpt.get('scales', None),
            'quats': ckpt.get('quats', None),
            'features_dc': ckpt.get('features_dc', None),
            'opacities': ckpt.get('opacities', None),
        }
        
        return gaussians
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return {}


def extract_object_gaussians(gaussians: Dict, object_indices: np.ndarray) -> Dict:
    """
    Extract Gaussian subset for placed object.
    
    Args:
        gaussians: Complete Gaussian parameters dict
        object_indices: Indices of object Gaussians
        
    Returns:
        Dictionary with object Gaussian parameters
    """
    object_gaussians = {}
    
    for key, value in gaussians.items():
        if value is not None and hasattr(value, 'shape'):
            object_gaussians[key] = value[object_indices]
        else:
            object_gaussians[key] = value
    
    return object_gaussians


def compute_gaussian_density_map(means: torch.Tensor, scales: torch.Tensor,
                                 resolution: Tuple[int, int]) -> np.ndarray:
    """
    Compute 2D density map from Gaussian parameters.
    
    Args:
        means: Gaussian means (N, 3)
        scales: Gaussian scales (N, 3)
        resolution: Output resolution (H, W)
        
    Returns:
        2D density map
    """
    h, w = resolution
    density_map = np.zeros((h, w))
    
    # Project Gaussians to 2D
    if means.is_cuda:
        means = means.cpu()
    if scales.is_cuda:
        scales = scales.cpu()
    
    means_np = means.detach().numpy()
    scales_np = scales.detach().numpy()
    
    # Simple projection: use x-y coordinates
    x = means_np[:, 0]
    y = means_np[:, 1]
    sx = scales_np[:, 0]
    sy = scales_np[:, 1]
    
    # Normalize to image space
    x_norm = ((x - x.min()) / (x.max() - x.min() + 1e-6) * w).astype(int)
    y_norm = ((y - y.min()) / (y.max() - y.min() + 1e-6) * h).astype(int)
    
    # Add Gaussian contribution
    for i in range(len(x_norm)):
        xi, yi = x_norm[i], y_norm[i]
        if 0 <= xi < w and 0 <= yi < h:
            # Simple Gaussian contribution
            radius = max(1, int(max(sx[i], sy[i]) * w * 0.01))
            y_start = max(0, yi - radius)
            y_end = min(h, yi + radius)
            x_start = max(0, xi - radius)
            x_end = min(w, xi + radius)
            
            yy, xx = np.ogrid[y_start:y_end, x_start:x_end]
            dist = (xx - xi)**2 + (yy - yi)**2
            gauss = np.exp(-dist / (2 * radius**2))
            density_map[y_start:y_end, x_start:x_end] += gauss
    
    return density_map / (density_map.max() + 1e-6)


def merge_gaussians(gaussians1: Dict, gaussians2: Dict, 
                   transition_mask: Optional[np.ndarray] = None) -> Dict:
    """
    Merge two Gaussian sets smoothly.
    
    Args:
        gaussians1: First Gaussian set (background)
        gaussians2: Second Gaussian set (object)
        transition_mask: Optional mask for smooth transition
        
    Returns:
        Merged Gaussian parameters
    """
    merged = {}
    
    for key in gaussians1.keys():
        if gaussians1[key] is None or gaussians2.get(key) is None:
            merged[key] = gaussians1[key]
            continue
        
        g1 = gaussians1[key]
        g2 = gaussians2[key]
        
        # Concatenate along Gaussian dimension
        if isinstance(g1, torch.Tensor) and isinstance(g2, torch.Tensor):
            merged[key] = torch.cat([g1, g2], dim=0)
        elif isinstance(g1, np.ndarray) and isinstance(g2, np.ndarray):
            merged[key] = np.concatenate([g1, g2], axis=0)
        else:
            merged[key] = g1
    
    return merged


def compute_gaussian_coverage(gaussians: Dict, resolution: Tuple[int, int],
                             coverage_threshold: float = 0.01) -> np.ndarray:
    """
    Compute coverage map from Gaussians.
    
    Args:
        gaussians: Gaussian parameters
        resolution: Target resolution
        coverage_threshold: Minimum contribution threshold
        
    Returns:
        Coverage mask (H, W)
    """
    h, w = resolution
    coverage = np.zeros((h, w))
    
    means = gaussians.get('means')
    scales = gaussians.get('scales')
    opacities = gaussians.get('opacities')
    
    if means is None:
        return coverage
    
    # Convert to numpy if needed
    if isinstance(means, torch.Tensor):
        means = means.detach().cpu().numpy()
    if isinstance(scales, torch.Tensor):
        scales = scales.detach().cpu().numpy()
    if isinstance(opacities, torch.Tensor):
        opacities = opacities.detach().cpu().numpy()
    
    # Project and accumulate
    for i in range(len(means)):
        x, y = means[i, 0], means[i, 1]
        sx, sy = scales[i, 0], scales[i, 1]
        op = opacities[i, 0] if opacities is not None else 1.0
        
        # Project to image space
        x_norm = int((x + 1) * w / 2)
        y_norm = int((y + 1) * h / 2)
        
        if 0 <= x_norm < w and 0 <= y_norm < h:
            radius = max(1, int(max(sx, sy) * max(w, h) * 0.05))
            y_start = max(0, y_norm - radius)
            y_end = min(h, y_norm + radius)
            x_start = max(0, x_norm - radius)
            x_end = min(w, x_norm + radius)
            
            yy, xx = np.ogrid[y_start:y_end, x_start:x_end]
            dist = (xx - x_norm)**2 + (yy - y_norm)**2
            gauss = np.exp(-dist / (2 * radius**2)) * op
            coverage[y_start:y_end, x_start:x_end] += gauss
    
    # Threshold
    coverage_mask = coverage > coverage_threshold
    
    return coverage_mask.astype(np.float32)
