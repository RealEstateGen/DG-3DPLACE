"""
Utility functions for shadow and lighting enhancement
"""

import numpy as np
from typing import Tuple, Optional


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range.
    
    Args:
        image: Input image
        
    Returns:
        Normalized image
    """
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    elif image.max() > 1.0:
        return image / 255.0
    return image.astype(np.float32)


def denormalize_image(image: np.ndarray, target_dtype=np.uint8) -> np.ndarray:
    """
    Convert normalized image back to integer range.
    
    Args:
        image: Normalized image [0, 1]
        target_dtype: Target data type
        
    Returns:
        Image in target dtype
    """
    if target_dtype == np.uint8:
        return np.clip(image * 255, 0, 255).astype(np.uint8)
    return np.clip(image, 0, 1).astype(target_dtype)


def estimate_dominant_color(image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Estimate dominant color in image.
    
    Args:
        image: Input image (H, W, 3)
        mask: Optional mask to focus on specific regions
        
    Returns:
        Dominant color as RGB array
    """
    if mask is not None:
        masked_image = image * mask[:, :, None]
        dom_color = np.sum(masked_image, axis=(0, 1)) / (np.sum(mask) * 3 + 1e-6)
    else:
        dom_color = np.mean(image, axis=(0, 1))
    
    return dom_color


def compute_image_gradient(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute image gradients.
    
    Args:
        image: Input image
        
    Returns:
        Tuple of (gradient_y, gradient_x)
    """
    if image.ndim == 3:
        # Convert to grayscale
        gray = np.mean(image, axis=2)
    else:
        gray = image
    
    gradient_y, gradient_x = np.gradient(gray)
    return gradient_y, gradient_x


def extract_object_mask(image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Extract object mask from image (assumes dark background).
    
    Args:
        image: Input image
        threshold: Threshold for mask
        
    Returns:
        Binary mask
    """
    brightness = np.mean(image, axis=2)
    mask = brightness > threshold
    return mask


def apply_bilateral_filter(image: np.ndarray, diameter: int = 9,
                          sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """
    Apply bilateral filter (edge-preserving blur).
    Note: Requires cv2, falls back to Gaussian if unavailable.
    
    Args:
        image: Input image
        diameter: Diameter of pixel neighborhood
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space
        
    Returns:
        Filtered image
    """
    try:
        import cv2
        # Convert to uint8 if needed
        if image.dtype == np.float32 or image.dtype == np.float64:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image
        
        filtered = cv2.bilateralFilter(image_uint8, diameter, sigma_color, sigma_space)
        
        if image.dtype == np.float32 or image.dtype == np.float64:
            return filtered.astype(np.float32) / 255.0
        return filtered
    except ImportError:
        # Fallback to Gaussian
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(image, sigma=1.0)


def blend_images(img1: np.ndarray, img2: np.ndarray, alpha: float) -> np.ndarray:
    """
    Blend two images.
    
    Args:
        img1: First image
        img2: Second image
        alpha: Blend factor (0 = img2, 1 = img1)
        
    Returns:
        Blended image
    """
    return img1 * alpha + img2 * (1 - alpha)


def match_histogram(source: np.ndarray, target: np.ndarray,
                   mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Match histogram of source to target.
    
    Args:
        source: Source image to adjust
        target: Target image to match
        mask: Optional mask for specific regions
        
    Returns:
        Histogram-matched source image
    """
    from scipy.ndimage import percentile_filter
    
    if mask is not None:
        source_masked = source[mask > 0]
        target_masked = target[mask > 0]
    else:
        source_masked = source
        target_masked = target
    
    # Simple histogram matching: match mean and std
    source_mean = np.mean(source_masked)
    source_std = np.std(source_masked) + 1e-6
    target_mean = np.mean(target_masked)
    target_std = np.std(target_masked) + 1e-6
    
    # Normalize source to target
    normalized = (source - source_mean) / source_std
    matched = normalized * target_std + target_mean
    
    return np.clip(matched, 0, 1)
