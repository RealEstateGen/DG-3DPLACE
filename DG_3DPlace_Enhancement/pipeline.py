"""Main enhancement pipeline orchestrator."""

import numpy as np
from typing import Dict, Optional

from .src import (
    LightingHarmonizer,
    ShadowGenerator,
    MaterialAnalyzer,
    DepthBlender
)
from . import config


class EnhancementPipeline:
    """
    Complete enhancement pipeline orchestrator.
    Coordinates all enhancement modules for unified processing.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize enhancement pipeline.
        
        Args:
            config_path: Path to configuration YAML file
        """
        # Load configuration
        self.config = config.load_config(config_path)
        self.device = self.config['RENDERING'].get('device', 'cuda')
        
        # Initialize modules
        self.lighting_harmonizer = LightingHarmonizer(device=self.device)
        self.shadow_generator = ShadowGenerator(device=self.device)
        self.material_analyzer = MaterialAnalyzer()
        self.depth_blender = DepthBlender()
        
        # Results cache
        self.results = {}
    
    def enhance_placed_object(self,
                             object_image: np.ndarray,
                             background_image: np.ndarray,
                             object_depth: np.ndarray,
                             background_depth: np.ndarray,
                             object_mask: Optional[np.ndarray] = None,
                             light_direction: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Full enhancement pipeline for placed object.
        
        Args:
            object_image: Object image to enhance
            background_image: Background scene image
            object_depth: Object depth map
            background_depth: Background depth map
            object_mask: Optional binary object mask
            light_direction: Optional light direction vector
            
        Returns:
            Enhanced composite image
        """
        # Step 1: Analyze scene
        scene_illum = self.lighting_harmonizer.analyze_illumination(background_image)
        scene_material = self.material_analyzer.analyze_surface_properties(background_image)
        
        if light_direction is None:
            light_direction = scene_illum['light_direction']
        
        # Step 2: Harmonize object lighting
        cfg_light = self.config['LIGHTING']
        enhanced = object_image.copy()
        
        if cfg_light['brightness']['enabled']:
            enhanced = self.lighting_harmonizer.harmonize_brightness(
                enhanced,
                scene_illum['brightness'],
                intensity_factor=cfg_light['brightness']['intensity_factor']
            )
        
        if cfg_light['color_temperature']['enabled']:
            enhanced = self.lighting_harmonizer.harmonize_color_temperature(
                enhanced,
                scene_illum['color_balance'],
                intensity_factor=cfg_light['color_temperature']['intensity_factor']
            )
        
        # Step 3: Analyze and match materials
        cfg_material = self.config['MATERIAL']
        if cfg_material['analysis_enabled']:
            obj_material = self.material_analyzer.analyze_surface_properties(
                enhanced, object_mask
            )
            adjustments = self.material_analyzer.match_material_to_scene(
                obj_material, scene_material
            )
            
            enhanced = self.material_analyzer.apply_roughness_adjustment(
                enhanced,
                cfg_material['roughness_adjustment']
            )
        
        # Step 4: Generate shadows
        cfg_shadow = self.config['SHADOW']
        if cfg_shadow['contact_shadow']['enabled']:
            shadow_mask = self.shadow_generator.generate_contact_shadow(
                background_depth,
                light_direction,
                object_height=cfg_shadow['contact_shadow']['object_height'],
                shadow_softness=cfg_shadow['contact_shadow']['softness'],
                shadow_intensity=cfg_shadow['contact_shadow']['intensity']
            )
            
            enhanced = self.shadow_generator.apply_shadow_to_image(
                enhanced,
                shadow_mask,
                shadow_color=tuple(cfg_shadow['shadow_color']),
                shadow_intensity=cfg_shadow['shadow_intensity']
            )
        
        # Step 5: Depth-aware blending
        cfg_blend = self.config['BLENDING']
        if cfg_blend['blend_type'] == 'adaptive':
            enhanced = self.depth_blender.adaptive_blending(
                enhanced,
                background_image,
                object_depth,
                background_depth,
                blend_hardness=cfg_blend['blend_hardness']
            )
        elif cfg_blend['blend_type'] == 'feather':
            if object_mask is not None:
                enhanced = self.depth_blender.blend_with_depth_feathering(
                    enhanced,
                    background_image,
                    object_mask,
                    feather_radius=cfg_blend['transition_width']
                )
        
        # Step 6: Post-processing
        cfg_post = self.config['POST_PROCESSING']
        if cfg_post['bilateral_filter']['enabled']:
            from .utils import image_processing
            enhanced = image_processing.apply_bilateral_filter(
                enhanced,
                diameter=cfg_post['bilateral_filter']['diameter'],
                sigma_color=cfg_post['bilateral_filter']['sigma_color'],
                sigma_space=cfg_post['bilateral_filter']['sigma_space']
            )
        
        # Cache results
        self.results = {
            'enhanced': enhanced,
            'scene_illumination': scene_illum,
            'shadow_mask': shadow_mask if cfg_shadow['contact_shadow']['enabled'] else None,
            'scene_material': scene_material,
        }
        
        return enhanced
    
    def get_shadow_map(self) -> Optional[np.ndarray]:
        """Get generated shadow map."""
        return self.results.get('shadow_mask')
    
    def get_analysis_results(self) -> Dict:
        """Get scene analysis results."""
        return {
            'illumination': self.results.get('scene_illumination'),
            'material': self.results.get('scene_material'),
        }


# Convenience function
def enhance_scene(object_image: np.ndarray,
                 background_image: np.ndarray,
                 object_depth: np.ndarray,
                 background_depth: np.ndarray,
                 config_path: Optional[str] = None,
                 **kwargs) -> np.ndarray:
    """
    Quick enhancement function.
    
    Args:
        object_image: Object image
        background_image: Background image  
        object_depth: Object depth map
        background_depth: Background depth map
        config_path: Optional config file path
        **kwargs: Additional parameters
        
    Returns:
        Enhanced image
    """
    pipeline = EnhancementPipeline(config_path)
    return pipeline.enhance_placed_object(
        object_image,
        background_image,
        object_depth,
        background_depth,
        **kwargs
    )
