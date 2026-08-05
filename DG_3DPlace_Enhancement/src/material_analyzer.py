"""
Material Analyzer - Analyze and match material properties
"""

import numpy as np
from typing import Dict, Tuple, Optional


class MaterialAnalyzer:
    """
    Analyzes material properties of objects and scenes for seamless integration.
    """
    
    def __init__(self):
        self.material_properties = {}
    
    def analyze_surface_properties(self, image: np.ndarray, 
                                   mask: Optional[np.ndarray] = None) -> Dict:
        """
        Analyze surface properties (roughness, specularity, etc.)
        
        Args:
            image: Input image (H, W, 3)
            mask: Optional binary mask
            
        Returns:
            Dictionary with material properties
        """
        if mask is None:
            mask = np.ones_like(image[:, :, 0])
        
        # Calculate properties
        roughness = self._estimate_roughness(image, mask)
        specularity = self._estimate_specularity(image, mask)
        diffuse_reflectance = np.mean(image[mask > 0])
        
        return {
            'roughness': roughness,
            'specularity': specularity,
            'diffuse_reflectance': float(diffuse_reflectance),
            'is_metallic': specularity > 0.6,
            'is_glossy': roughness < 0.3
        }
    
    def _estimate_roughness(self, image: np.ndarray, mask: np.ndarray) -> float:
        """
        Estimate surface roughness from image variance.
        Rough surfaces have more uniform color variation.
        """
        masked_image = image * mask[:, :, None]
        
        # Calculate spatial variance
        r_var = np.var(masked_image[:, :, 0][mask > 0])
        g_var = np.var(masked_image[:, :, 1][mask > 0])
        b_var = np.var(masked_image[:, :, 2][mask > 0])
        
        avg_variance = (r_var + g_var + b_var) / 3
        
        # Normalize to [0, 1] range (empirical scaling)
        roughness = min(1.0, avg_variance * 2.0)
        
        return float(roughness)
    
    def _estimate_specularity(self, image: np.ndarray, mask: np.ndarray) -> float:
        """
        Estimate specularity from highlight intensity.
        """
        masked_image = image * mask[:, :, None]
        
        # Find bright regions that might be specular highlights
        brightness = np.mean(masked_image, axis=2)
        bright_threshold = np.percentile(brightness[mask > 0], 75)
        
        bright_pixels = brightness > bright_threshold
        specularity = np.mean(brightness[bright_pixels & (mask > 0)])
        
        return float(specularity)
    
    def match_material_to_scene(self, object_props: Dict, 
                               scene_props: Dict,
                               adjustment_strength: float = 0.7) -> Dict:
        """
        Calculate adjustments needed to match object material to scene.
        
        Args:
            object_props: Material properties of object
            scene_props: Material properties of scene
            adjustment_strength: How much to adjust (0-1)
            
        Returns:
            Adjustment parameters
        """
        diff_roughness = (scene_props['roughness'] - object_props['roughness']) * adjustment_strength
        diff_specularity = (scene_props['specularity'] - object_props['specularity']) * adjustment_strength
        
        return {
            'roughness_adjustment': float(diff_roughness),
            'specularity_adjustment': float(diff_specularity),
            'brightness_adjustment': float(scene_props['diffuse_reflectance'] - object_props['diffuse_reflectance']),
            'match_metallic': scene_props['is_metallic'],
            'match_glossy': scene_props['is_glossy']
        }
    
    def apply_roughness_adjustment(self, image: np.ndarray, 
                                  roughness_delta: float) -> np.ndarray:
        """
        Apply roughness adjustment to image.
        Increasing roughness = reduce specular highlights, increase diffuse.
        
        Args:
            image: Input image
            roughness_delta: Adjustment amount (-1 to 1)
            
        Returns:
            Adjusted image
        """
        if roughness_delta == 0:
            return image
        
        result = image.copy()
        
        if roughness_delta > 0:
            # Increase roughness: reduce highlights, increase variance
            from scipy.ndimage import gaussian_filter
            blurred = gaussian_filter(image, sigma=1.0)
            result = image * (1 - roughness_delta * 0.5) + blurred * (roughness_delta * 0.5)
        else:
            # Decrease roughness: enhance highlights
            brightness = np.mean(image, axis=2, keepdims=True)
            result = image * (1 + abs(roughness_delta) * 0.3)
        
        return np.clip(result, 0, 1)
    
    def apply_specularity_adjustment(self, image: np.ndarray,
                                    specularity_delta: float) -> np.ndarray:
        """
        Apply specularity adjustment to image.
        
        Args:
            image: Input image
            specularity_delta: Adjustment amount (-1 to 1)
            
        Returns:
            Adjusted image
        """
        if specularity_delta == 0:
            return image
        
        # Enhance or reduce bright regions
        brightness = np.mean(image, axis=2, keepdims=True)
        highlight_mask = brightness > 0.5
        
        result = image.copy()
        adjustment = 1 + specularity_delta * 0.3
        result[highlight_mask] *= adjustment
        
        return np.clip(result, 0, 1)
    
    def normalize_reflectance(self, image: np.ndarray,
                            target_reflectance: float) -> np.ndarray:
        """
        Normalize diffuse reflectance to target value.
        
        Args:
            image: Input image
            target_reflectance: Target average brightness (0-1)
            
        Returns:
            Normalized image
        """
        current_reflectance = np.mean(image)
        
        if current_reflectance > 0:
            scale_factor = target_reflectance / current_reflectance
            result = image * scale_factor
            return np.clip(result, 0, 1)
        
        return image
