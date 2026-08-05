"""
Lighting Harmonizer - Match and normalize lighting between scene and placed objects
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional


class LightingHarmonizer:
    """
    Analyzes scene lighting environment and applies corrections to placed objects
    to integrate them seamlessly with the background scene.
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
    def analyze_illumination(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict:
        """
        Analyze illumination characteristics of an image.
        
        Args:
            image: Input image (H, W, 3) in range [0, 1]
            mask: Optional binary mask to exclude regions
            
        Returns:
            Dictionary with illumination properties
        """
        if mask is None:
            mask = np.ones_like(image[:, :, 0])
        
        # Convert to grayscale
        gray = np.mean(image, axis=2)
        masked_gray = gray * mask
        
        # Calculate brightness statistics
        brightness = np.sum(masked_gray) / (np.sum(mask) + 1e-6)
        variance = np.var(masked_gray[mask > 0])
        
        # Estimate dominant light direction from gradients
        gy, gx = np.gradient(gray)
        light_dir = self._estimate_light_direction(gx, gy, mask)
        
        # Color temperature analysis (warm vs cool)
        avg_color = np.mean(image, axis=(0, 1))
        
        return {
            'brightness': float(brightness),
            'variance': float(variance),
            'light_direction': light_dir,
            'avg_color': avg_color,
            'color_balance': self._analyze_color_temperature(image, mask)
        }
    
    def _estimate_light_direction(self, gx: np.ndarray, gy: np.ndarray, 
                                   mask: np.ndarray) -> np.ndarray:
        """Estimate light direction from image gradients."""
        gx_masked = gx * mask
        gy_masked = gy * mask
        
        light_x = np.mean(gx_masked[mask > 0])
        light_y = np.mean(gy_masked[mask > 0])
        
        norm = np.sqrt(light_x**2 + light_y**2) + 1e-6
        return np.array([light_x / norm, light_y / norm])
    
    def _analyze_color_temperature(self, image: np.ndarray, mask: np.ndarray) -> Dict:
        """Analyze warm/cool color temperature of the scene."""
        r_avg = np.mean(image[:, :, 0][mask > 0])
        g_avg = np.mean(image[:, :, 1][mask > 0])
        b_avg = np.mean(image[:, :, 2][mask > 0])
        
        return {
            'red': float(r_avg),
            'green': float(g_avg),
            'blue': float(b_avg),
            'warmth': float(r_avg - b_avg)  # Positive = warm, negative = cool
        }
    
    def harmonize_brightness(self, object_image: np.ndarray, target_brightness: float,
                            intensity_factor: float = 0.8) -> np.ndarray:
        """
        Adjust object brightness to match scene brightness.
        
        Args:
            object_image: Object image (H, W, 3)
            target_brightness: Target brightness level from scene
            intensity_factor: How much to blend (0-1)
            
        Returns:
            Brightness-adjusted image
        """
        current_brightness = np.mean(object_image)
        brightness_ratio = (target_brightness + 1e-6) / (current_brightness + 1e-6)
        
        # Apply adjustment with blending
        adjusted = object_image * (1 + (brightness_ratio - 1) * intensity_factor)
        return np.clip(adjusted, 0, 1)
    
    def harmonize_color_temperature(self, object_image: np.ndarray, 
                                    scene_color_balance: Dict,
                                    intensity_factor: float = 0.7) -> np.ndarray:
        """
        Adjust object color temperature to match scene.
        
        Args:
            object_image: Object image (H, W, 3)
            scene_color_balance: Color balance dict from analyze_illumination
            intensity_factor: Adjustment strength (0-1)
            
        Returns:
            Color-temperature-adjusted image
        """
        # Simple color channel adjustment
        r_factor = 1.0 + (scene_color_balance['red'] - 0.5) * intensity_factor
        b_factor = 1.0 + (scene_color_balance['blue'] - 0.5) * intensity_factor
        
        adjusted = object_image.copy()
        adjusted[:, :, 0] *= r_factor  # Red channel
        adjusted[:, :, 2] *= b_factor  # Blue channel
        
        return np.clip(adjusted, 0, 1)
    
    def apply_ambient_occlusion_shadow(self, image: np.ndarray, depth_map: np.ndarray,
                                       shadow_intensity: float = 0.3) -> np.ndarray:
        """
        Apply subtle ambient occlusion shadows based on depth.
        
        Args:
            image: Input image
            depth_map: Depth map (H, W)
            shadow_intensity: Shadow strength (0-1)
            
        Returns:
            Image with AO shadows applied
        """
        # Normalize depth
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
        
        # Blur depth for softer shadows
        from scipy.ndimage import gaussian_filter
        depth_blurred = gaussian_filter(depth_normalized, sigma=2.0)
        
        # Create shadow mask (darker at occluded areas)
        shadow_mask = 1.0 - (depth_blurred * shadow_intensity)
        
        # Apply to all channels
        result = image.copy()
        for c in range(3):
            result[:, :, c] *= shadow_mask
        
        return result
