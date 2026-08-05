"""
Color matching utilities
"""

import numpy as np
from typing import Tuple, Dict


def match_color_spaces(source_image: np.ndarray, target_image: np.ndarray,
                      intensity_only: bool = False) -> np.ndarray:
    """
    Match color space of source to target.
    
    Args:
        source_image: Source image to adjust
        target_image: Target image to match
        intensity_only: If True, only match brightness, not color
        
    Returns:
        Color-matched source image
    """
    source_mean = np.mean(source_image, axis=(0, 1))
    target_mean = np.mean(target_image, axis=(0, 1))
    
    source_std = np.std(source_image, axis=(0, 1)) + 1e-6
    target_std = np.std(target_image, axis=(0, 1)) + 1e-6
    
    if intensity_only:
        # Match only overall brightness
        scale = np.mean(target_std) / np.mean(source_std)
        offset = np.mean(target_mean) - np.mean(source_mean)
        result = source_image * scale + offset
    else:
        # Match per-channel
        result = source_image.copy()
        for c in range(3):
            result[:, :, c] = (source_image[:, :, c] - source_mean[c]) * (target_std[c] / source_std[c]) + target_mean[c]
    
    return np.clip(result, 0, 1)


def adjust_color_temperature(image: np.ndarray, temperature_shift: float) -> np.ndarray:
    """
    Adjust color temperature of image.
    
    Args:
        image: Input image
        temperature_shift: Shift amount (-1 to 1, negative = cooler, positive = warmer)
        
    Returns:
        Color-adjusted image
    """
    result = image.copy()
    
    if temperature_shift > 0:
        # Warmer: increase red, decrease blue
        result[:, :, 0] *= (1 + temperature_shift * 0.3)  # Red
        result[:, :, 2] *= (1 - temperature_shift * 0.2)  # Blue
    elif temperature_shift < 0:
        # Cooler: decrease red, increase blue
        result[:, :, 0] *= (1 + temperature_shift * 0.2)  # Red
        result[:, :, 2] *= (1 - temperature_shift * 0.3)  # Blue
    
    return np.clip(result, 0, 1)


def adjust_saturation(image: np.ndarray, saturation_factor: float) -> np.ndarray:
    """
    Adjust color saturation.
    
    Args:
        image: Input image
        saturation_factor: Saturation multiplier (1.0 = no change, >1 = more saturated)
        
    Returns:
        Saturation-adjusted image
    """
    # Convert to HSV
    from PIL import Image as PILImage
    from PIL.ImageEnhance import Color
    
    try:
        # Use PIL if available
        img_pil = PILImage.fromarray((image * 255).astype(np.uint8))
        enhancer = Color(img_pil)
        enhanced = enhancer.enhance(saturation_factor)
        return np.array(enhanced) / 255.0
    except:
        # Fallback: simple luminance-preserving adjustment
        brightness = np.mean(image, axis=2, keepdims=True)
        result = brightness + (image - brightness) * saturation_factor
        return np.clip(result, 0, 1)


def white_balance(image: np.ndarray, gray_point: Tuple[float, float, float] = None) -> np.ndarray:
    """
    Apply white balance correction.
    
    Args:
        image: Input image
        gray_point: RGB gray point for correction, or None for auto
        
    Returns:
        White-balanced image
    """
    if gray_point is None:
        # Auto white balance: use average color
        avg_color = np.mean(image, axis=(0, 1))
        gray_point = np.mean(avg_color)
    
    # Normalize each channel to gray point
    result = image.copy()
    for c in range(3):
        channel_mean = np.mean(image[:, :, c])
        scale = gray_point / (channel_mean + 1e-6)
        result[:, :, c] *= scale
    
    return np.clip(result, 0, 1)


def color_to_grayscale(image: np.ndarray, weights: Tuple[float, float, float] = None) -> np.ndarray:
    """
    Convert color image to grayscale with custom weights.
    
    Args:
        image: Input image
        weights: Custom weights for R, G, B channels
        
    Returns:
        Grayscale image (H, W)
    """
    if weights is None:
        weights = (0.299, 0.587, 0.114)  # Standard weights
    
    return np.dot(image, weights)


def increase_contrast(image: np.ndarray, contrast_factor: float = 1.2) -> np.ndarray:
    """
    Increase image contrast.
    
    Args:
        image: Input image
        contrast_factor: Contrast multiplier (>1 = more contrast)
        
    Returns:
        Contrast-adjusted image
    """
    mean = np.mean(image)
    result = (image - mean) * contrast_factor + mean
    return np.clip(result, 0, 1)


def shadow_highlight_adjustment(image: np.ndarray,
                               shadow_adjustment: float = 0.0,
                               highlight_adjustment: float = 0.0) -> np.ndarray:
    """
    Adjust shadows and highlights independently.
    
    Args:
        image: Input image
        shadow_adjustment: Adjustment for dark areas (-1 to 1, negative = darker)
        highlight_adjustment: Adjustment for bright areas (-1 to 1, negative = darker)
        
    Returns:
        Adjusted image
    """
    result = image.copy()
    brightness = np.mean(image, axis=2)
    
    # Apply shadow adjustment
    if shadow_adjustment != 0:
        dark_mask = brightness < 0.33
        result[dark_mask] *= (1 + shadow_adjustment * 0.2)
    
    # Apply highlight adjustment
    if highlight_adjustment != 0:
        bright_mask = brightness > 0.67
        result[bright_mask] *= (1 + highlight_adjustment * 0.2)
    
    return np.clip(result, 0, 1)
