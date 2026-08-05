"""
Shadow and Lighting Enhancement Module for DG-3DPlace
Enhances shadows and lighting of placed 3DGS objects to integrate seamlessly with scene
"""

from .lighting_harmonizer import LightingHarmonizer
from .shadow_generator import ShadowGenerator
from .material_analyzer import MaterialAnalyzer
from .depth_blender import DepthBlender

__all__ = [
    'LightingHarmonizer',
    'ShadowGenerator',
    'MaterialAnalyzer',
    'DepthBlender'
]
