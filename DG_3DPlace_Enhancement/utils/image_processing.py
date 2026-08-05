"""
Image processing utilities
"""

import numpy as np
from typing import Tuple, Optional, List
from scipy.ndimage import gaussian_filter


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image to target size.
    
    Args:
        image: Input image
        size: Target size (height, width)
        
    Returns:
        Resized image
    """
    from PIL import Image
    
    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
        if image.dtype == np.float32 or image.dtype == np.float64:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image
        
        pil_img = Image.fromarray(image_uint8 if image_uint8.ndim == 2 else image_uint8)
        pil_img = pil_img.resize((size[1], size[0]), Image.Resampling.LANCZOS)
        
        result = np.array(pil_img)
        if image.dtype == np.float32 or image.dtype == np.float64:
            return result.astype(np.float32) / 255.0
        return result
    
    return image


def crop_image(image: np.ndarray, bounds: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop image to specified bounds.
    
    Args:
        image: Input image
        bounds: (y_start, y_end, x_start, x_end)
        
    Returns:
        Cropped image
    """
    y_start, y_end, x_start, x_end = bounds
    return image[y_start:y_end, x_start:x_end]


def apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply Gaussian blur.
    
    Args:
        image: Input image
        sigma: Blur sigma
        
    Returns:
        Blurred image
    """
    return gaussian_filter(image, sigma=sigma)


def apply_unsharp_mask(image: np.ndarray, sigma: float = 1.0, strength: float = 1.0) -> np.ndarray:
    """
    Apply unsharp mask for sharpening.
    
    Args:
        image: Input image
        sigma: Gaussian blur sigma
        strength: Sharpening strength
        
    Returns:
        Sharpened image
    """
    blurred = gaussian_filter(image, sigma=sigma)
    sharpened = image + (image - blurred) * strength
    return np.clip(sharpened, 0, 1)


def dilate_mask(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """
    Dilate binary mask.
    
    Args:
        mask: Binary mask
        radius: Dilation radius
        
    Returns:
        Dilated mask
    """
    from scipy.ndimage import binary_dilation
    return binary_dilation(mask, iterations=radius).astype(np.float32)


def erode_mask(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """
    Erode binary mask.
    
    Args:
        mask: Binary mask
        radius: Erosion radius
        
    Returns:
        Eroded mask
    """
    from scipy.ndimage import binary_erosion
    return binary_erosion(mask, iterations=radius).astype(np.float32)


def morphological_open(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """
    Morphological opening (erosion followed by dilation).
    
    Args:
        mask: Binary mask
        radius: Structuring element size
        
    Returns:
        Opened mask
    """
    eroded = erode_mask(mask, radius)
    opened = dilate_mask(eroded, radius)
    return opened


def morphological_close(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """
    Morphological closing (dilation followed by erosion).
    
    Args:
        mask: Binary mask
        radius: Structuring element size
        
    Returns:
        Closed mask
    """
    dilated = dilate_mask(mask, radius)
    closed = erode_mask(dilated, radius)
    return closed


def connected_components(mask: np.ndarray, connectivity: int = 2) -> Tuple[np.ndarray, int]:
    """
    Label connected components in binary mask.
    
    Args:
        mask: Binary mask
        connectivity: 2 for 4-connectivity, 3 for 8-connectivity
        
    Returns:
        Labeled components and number of components
    """
    from scipy.ndimage import label
    labeled, num_components = label(mask)
    return labeled, num_components


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """
    Extract largest connected component from mask.
    
    Args:
        mask: Binary mask
        
    Returns:
        Mask with only largest component
    """
    labeled, num_components = connected_components(mask)
    
    if num_components == 0:
        return mask
    
    # Find largest component
    component_sizes = np.bincount(labeled.ravel())
    largest_label = np.argmax(component_sizes[1:]) + 1
    
    return (labeled == largest_label).astype(np.float32)


def gap_fill(mask: np.ndarray, max_gap_size: int = 3) -> np.ndarray:
    """
    Fill small gaps in mask.
    
    Args:
        mask: Binary mask
        max_gap_size: Maximum gap size to fill
        
    Returns:
        Mask with gaps filled
    """
    # Use morphological closing
    return morphological_close(mask, max_gap_size)


def edge_detection(image: np.ndarray, method: str = 'sobel') -> np.ndarray:
    """
    Detect edges in image.
    
    Args:
        image: Input image
        method: 'sobel', 'canny', or 'laplacian'
        
    Returns:
        Edge map
    """
    from scipy.ndimage import sobel, laplace
    
    if image.ndim == 3:
        image = np.mean(image, axis=2)
    
    if method == 'sobel':
        edges = np.sqrt(sobel(image, axis=0)**2 + sobel(image, axis=1)**2)
    elif method == 'laplacian':
        edges = np.abs(laplace(image))
    else:
        # Default to Sobel
        edges = np.sqrt(sobel(image, axis=0)**2 + sobel(image, axis=1)**2)
    
    # Normalize
    edges = edges / (edges.max() + 1e-6)
    return edges


def pyramid_down(image: np.ndarray, levels: int = 1) -> List[np.ndarray]:
    """
    Create Gaussian image pyramid.
    
    Args:
        image: Input image
        levels: Number of pyramid levels
        
    Returns:
        List of downsampled images
    """
    pyramid = [image]
    current = image.copy()
    
    for _ in range(levels - 1):
        blurred = gaussian_filter(current, sigma=1.0)
        downsampled = blurred[::2, ::2]
        pyramid.append(downsampled)
        current = downsampled
    
    return pyramid
