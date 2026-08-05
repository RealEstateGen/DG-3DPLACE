"""
Shadow Generator - Generate realistic shadows for placed objects
"""

import torch
import numpy as np
from typing import Tuple, Optional
from scipy.ndimage import gaussian_filter


class ShadowGenerator:
    """
    Generates and integrates shadows for placed 3DGS objects.
    Creates depth-aware shadows that follow light direction and scene geometry.
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
    
    def generate_contact_shadow(self, depth_map: np.ndarray, 
                               light_direction: np.ndarray,
                               object_height: float = 0.5,
                               shadow_softness: float = 3.0,
                               shadow_intensity: float = 0.4) -> np.ndarray:
        """
        Generate contact shadows on the ground plane.
        
        Args:
            depth_map: Scene depth map (H, W)
            light_direction: Estimated light direction (2D or 3D)
            object_height: Approximate object height in scene units
            shadow_softness: Gaussian blur radius for soft shadows
            shadow_intensity: Shadow opacity (0-1)
            
        Returns:
            Shadow mask (H, W)
        """
        h, w = depth_map.shape
        
        # Create shadow projection based on light direction
        if len(light_direction.shape) == 1 and light_direction.shape[0] == 2:
            # 2D light direction
            light_dir_2d = light_direction / (np.linalg.norm(light_direction) + 1e-6)
            offset_y = int(light_dir_2d[0] * w * 0.05)
            offset_x = int(light_dir_2d[1] * h * 0.05)
        else:
            # Default offset
            offset_x, offset_y = 5, 10
        
        # Find ground plane (typically max depth in this context)
        ground_depth = np.percentile(depth_map, 95)
        
        # Create shadow mask based on proximity to ground
        distance_to_ground = np.abs(depth_map - ground_depth)
        shadow_mask = np.exp(-distance_to_ground / (object_height + 1e-6))
        
        # Apply softness
        shadow_mask = gaussian_filter(shadow_mask, sigma=shadow_softness)
        
        # Apply intensity
        shadow_mask = np.clip(shadow_mask * shadow_intensity, 0, 1)
        
        # Shift shadow in light direction
        shadow_mask = np.roll(shadow_mask, offset_x, axis=0)
        shadow_mask = np.roll(shadow_mask, offset_y, axis=1)
        
        return shadow_mask
    
    def generate_cast_shadow(self, object_depth: np.ndarray,
                            light_direction: np.ndarray,
                            scene_depth: np.ndarray,
                            max_shadow_distance: float = 2.0,
                            shadow_softness: float = 2.0) -> np.ndarray:
        """
        Generate cast shadows from object geometry.
        
        Args:
            object_depth: Depth map of placed object
            light_direction: Light direction vector
            scene_depth: Background scene depth
            max_shadow_distance: Maximum shadow projection distance
            shadow_softness: Gaussian blur for soft edges
            
        Returns:
            Cast shadow mask (H, W)
        """
        h, w = object_depth.shape
        
        # Normalize light direction
        if len(light_direction.shape) == 1:
            if light_direction.shape[0] == 2:
                # 2D direction - extend to 3D
                light_dir = np.array([light_direction[0], light_direction[1], 0.5])
            else:
                light_dir = light_direction
        else:
            light_dir = light_direction
        
        light_dir = light_dir / (np.linalg.norm(light_dir) + 1e-6)
        
        # Create shadow map by projecting object geometry along light direction
        shadow_map = np.zeros_like(object_depth)
        
        # For each object pixel, project shadow
        object_mask = object_depth > 0
        
        # Simple shadow projection: shift object based on light direction
        shift_y = int(light_dir[0] * max_shadow_distance * w)
        shift_x = int(light_dir[1] * max_shadow_distance * h)
        
        shadow_map = np.roll(object_mask.astype(float), shift_x, axis=0)
        shadow_map = np.roll(shadow_map, shift_y, axis=1)
        
        # Only keep shadow where object is "above" surface
        shadow_validity = object_depth[:, :, None] < scene_depth[:, :, None] if scene_depth.ndim == 3 else object_depth < scene_depth
        if isinstance(shadow_validity, np.ndarray):
            shadow_map *= shadow_validity.astype(float)
        
        # Apply softness
        shadow_map = gaussian_filter(shadow_map, sigma=shadow_softness)
        
        return np.clip(shadow_map, 0, 1)
    
    def apply_shadow_to_image(self, image: np.ndarray, 
                             shadow_mask: np.ndarray,
                             shadow_color: Tuple[float, float, float] = (0, 0, 0),
                             shadow_intensity: float = 0.5) -> np.ndarray:
        """
        Blend shadow onto image.
        
        Args:
            image: Input image (H, W, 3)
            shadow_mask: Shadow mask (H, W)
            shadow_color: Shadow color (R, G, B) in range [0, 1]
            shadow_intensity: How dark shadows should be
            
        Returns:
            Image with shadows applied
        """
        # Ensure shadow mask is 3D
        if shadow_mask.ndim == 2:
            shadow_mask = np.stack([shadow_mask] * 3, axis=2)
        
        # Darken image where shadow is present
        shadow_color = np.array(shadow_color)
        result = image * (1 - shadow_mask * shadow_intensity) + shadow_color * shadow_mask * shadow_intensity
        
        return np.clip(result, 0, 1)
    
    def blend_shadows_with_falloff(self, image: np.ndarray,
                                  shadow_mask: np.ndarray,
                                  falloff_distance: int = 5) -> np.ndarray:
        """
        Blend shadows with distance falloff for natural integration.
        
        Args:
            image: Input image
            shadow_mask: Shadow mask
            falloff_distance: Distance for shadow edge softening
            
        Returns:
            Image with softly integrated shadows
        """
        from scipy.ndimage import distance_transform_edt
        
        # Create distance falloff
        if shadow_mask.max() > 0:
            dist = distance_transform_edt(1 - shadow_mask)
            falloff = np.exp(-(dist ** 2) / (falloff_distance ** 2))
            
            # Apply with falloff
            result = image.copy()
            if shadow_mask.ndim == 2:
                for c in range(3):
                    result[:, :, c] *= (1 - falloff * 0.3)
            else:
                result *= (1 - falloff[:, :, None] * 0.3)
            
            return np.clip(result, 0, 1)
        
        return image
