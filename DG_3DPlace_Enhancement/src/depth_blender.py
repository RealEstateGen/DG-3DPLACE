"""
Depth Blender - Depth-aware blending for seamless object integration
"""

import numpy as np
from typing import Tuple, Optional
from scipy.ndimage import gaussian_filter


class DepthBlender:
    """
    Performs depth-aware blending to integrate objects seamlessly.
    Uses depth information to create natural transitions between object and scene.
    """
    
    def __init__(self):
        self.blend_cache = {}
    
    def compute_depth_based_blend_mask(self, object_depth: np.ndarray,
                                       scene_depth: np.ndarray,
                                       transition_softness: float = 2.0) -> np.ndarray:
        """
        Compute a blend mask based on depth relationships.
        
        Args:
            object_depth: Object depth map (H, W)
            scene_depth: Scene background depth (H, W)
            transition_softness: Gaussian blur radius for smooth transitions
            
        Returns:
            Blend mask (H, W) with smooth transitions
        """
        # Compute depth difference
        depth_diff = object_depth - scene_depth
        
        # Normalize to [0, 1]
        depth_diff_norm = np.clip((depth_diff / (np.max(np.abs(depth_diff)) + 1e-6)) + 0.5, 0, 1)
        
        # Apply smoothing
        blend_mask = gaussian_filter(depth_diff_norm, sigma=transition_softness)
        
        return np.clip(blend_mask, 0, 1)
    
    def blend_with_depth_feathering(self, object_image: np.ndarray,
                                    scene_background: np.ndarray,
                                    object_alpha: np.ndarray,
                                    feather_radius: int = 10) -> np.ndarray:
        """
        Blend object and background with depth-aware feathering.
        
        Args:
            object_image: Object image (H, W, 3)
            scene_background: Background scene (H, W, 3)
            object_alpha: Object alpha/transparency mask (H, W)
            feather_radius: Feathering distance in pixels
            
        Returns:
            Blended image (H, W, 3)
        """
        # Create feathered alpha mask
        if object_alpha.ndim == 3:
            object_alpha = object_alpha[:, :, 0]
        
        # Feather edges using distance transform
        from scipy.ndimage import distance_transform_edt
        
        alpha_mask = object_alpha > 0.5
        dist = distance_transform_edt(~alpha_mask)
        feather_mask = np.exp(-(dist ** 2) / (feather_radius ** 2))
        feather_mask = feather_mask * object_alpha
        
        # Expand to 3D
        feather_mask_3d = np.stack([feather_mask] * 3, axis=2)
        
        # Blend
        result = object_image * feather_mask_3d + scene_background * (1 - feather_mask_3d)
        
        return np.clip(result, 0, 1)
    
    def adaptive_blending(self, object_image: np.ndarray,
                         scene_background: np.ndarray,
                         object_depth: np.ndarray,
                         scene_depth: np.ndarray,
                         blend_hardness: float = 1.0) -> np.ndarray:
        """
        Adaptive blending based on depth discontinuity.
        
        Args:
            object_image: Object image
            scene_background: Background scene
            object_depth: Object depth map
            scene_depth: Scene depth map
            blend_hardness: How hard the depth transition is (higher = sharper)
            
        Returns:
            Blended image
        """
        # Compute depth-based blend mask
        depth_diff = np.abs(object_depth - scene_depth)
        
        # Create sigmoid-like transition
        max_diff = np.percentile(depth_diff[object_depth > 0], 95)
        blend_mask = 1.0 / (1.0 + np.exp(-blend_hardness * (depth_diff - max_diff / 2)))
        
        # Smooth the mask
        blend_mask = gaussian_filter(blend_mask, sigma=1.5)
        
        # Expand to 3D
        blend_mask_3d = np.stack([blend_mask] * 3, axis=2)
        
        # Blend images
        result = object_image * blend_mask_3d + scene_background * (1 - blend_mask_3d)
        
        return np.clip(result, 0, 1)
    
    def create_transition_zone(self, object_alpha: np.ndarray,
                              transition_width: int = 20) -> np.ndarray:
        """
        Create a smooth transition zone at object boundaries.
        
        Args:
            object_alpha: Object alpha mask (H, W)
            transition_width: Width of transition zone in pixels
            
        Returns:
            Smoothed alpha mask
        """
        from scipy.ndimage import distance_transform_edt, binary_dilation
        
        # Get object mask
        obj_mask = object_alpha > 0.5
        
        # Create distance transform
        dist_inside = distance_transform_edt(obj_mask)
        dist_outside = distance_transform_edt(~obj_mask)
        
        # Create smooth transition
        boundary_dist = np.minimum(dist_inside, dist_outside)
        smooth_alpha = np.exp(-(boundary_dist ** 2) / (transition_width ** 2))
        
        return np.clip(smooth_alpha, 0, 1)
    
    def bilateral_blend(self, object_image: np.ndarray,
                       scene_background: np.ndarray,
                       object_alpha: np.ndarray,
                       spatial_sigma: float = 5.0,
                       range_sigma: float = 0.1) -> np.ndarray:
        """
        Bilateral blending for edge-preserving integration.
        
        Args:
            object_image: Object image
            scene_background: Background
            object_alpha: Alpha mask
            spatial_sigma: Spatial extent of filter
            range_sigma: Color range for blending
            
        Returns:
            Blended image with preserved edges
        """
        from scipy.ndimage import gaussian_filter
        
        # Compute color difference
        color_diff = np.mean(np.abs(object_image - scene_background), axis=2)
        
        # Create bilateral weight
        spatial_weight = gaussian_filter(object_alpha, sigma=spatial_sigma)
        range_weight = np.exp(-(color_diff ** 2) / (range_sigma ** 2))
        
        # Combined weight
        weight = spatial_weight * range_weight
        weight = np.stack([weight] * 3, axis=2)
        
        # Blend
        result = object_image * weight + scene_background * (1 - weight)
        
        return np.clip(result, 0, 1)
    
    def inpaint_missing_regions(self, image: np.ndarray,
                               mask: np.ndarray,
                               inpaint_radius: int = 15) -> np.ndarray:
        """
        Inpaint missing or masked regions using surrounding context.
        
        Args:
            image: Input image
            mask: Areas to inpaint (1 = inpaint, 0 = keep)
            inpaint_radius: Radius for context collection
            
        Returns:
            Inpainted image
        """
        from scipy.ndimage import gaussian_filter
        
        # Simple inpainting: blur and blend
        inpaint_region = mask > 0.5
        blurred = gaussian_filter(image, sigma=inpaint_radius)
        
        result = image.copy()
        result[inpaint_region] = blurred[inpaint_region]
        
        return np.clip(result, 0, 1)
