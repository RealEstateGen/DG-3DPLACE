#!/usr/bin/env python3
"""
Enhance 3DGS Scene with Shadow and Lighting Enhancement

This script takes 3DGS checkpoints from DG_3DPlace_Optimization and applies
the enhancement pipeline to improve object integration.

Usage:
    python enhance_3dgs_scene.py \
        --input-ckpt DG_3DPlace_Optimization/data/inputs/scene_with_initial_object.ckpt \
        --camera DG_3DPlace_Optimization/data/inputs/selected_camera.pt \
        --output outputs/enhanced_3dgs_render.png
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from datetime import datetime

# Add project root to path dynamically
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from DG_3DPlace_Enhancement.pipeline import EnhancementPipeline


def load_gsplat_checkpoint(ckpt_path, device='cuda'):
    """Load 3DGS checkpoint and extract gaussian tensors."""
    print(f"📦 Loading checkpoint: {os.path.basename(ckpt_path)}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    if 'pipeline' not in ckpt:
        raise ValueError("Checkpoint must contain 'pipeline' key")
    
    pipeline_dict = ckpt['pipeline']
    
    # Extract Gaussian parameters
    gaussians = {
        'means': pipeline_dict.get('_model.means'),
        'scales': pipeline_dict.get('_model.scales'),
        'quats': pipeline_dict.get('_model.quats'),
        'features_dc': pipeline_dict.get('_model.features_dc'),
        'features_rest': pipeline_dict.get('_model.features_rest'),
        'opacities': pipeline_dict.get('_model.opacities'),
    }
    
    print(f"   ✓ Loaded {len(gaussians['means'])} Gaussians")
    return ckpt, gaussians


def compute_global_rgb_transform(source_rgb, target_rgb):
    """Compute per-channel affine transform from source to target image."""
    source_flat = source_rgb.reshape(-1, 3)
    target_flat = target_rgb.reshape(-1, 3)

    src_mean = np.mean(source_flat, axis=0)
    src_std = np.std(source_flat, axis=0) + 1e-6
    tgt_mean = np.mean(target_flat, axis=0)
    tgt_std = np.std(target_flat, axis=0) + 1e-6

    gain = np.clip(tgt_std / src_std, 0.5, 1.5)
    bias = np.clip(tgt_mean - src_mean * gain, -0.25, 0.25)

    return gain.astype(np.float32), bias.astype(np.float32)


def apply_rgb_transform_to_features_dc(features_dc, gain, bias):
    """Apply per-channel affine RGB transform to SH-DC features."""
    if not torch.is_tensor(features_dc):
        raise TypeError("features_dc must be a torch.Tensor")

    if features_dc.shape[-1] != 3:
        raise ValueError(f"Expected features_dc last dim 3, got shape {tuple(features_dc.shape)}")

    gain_t = torch.tensor(gain, dtype=features_dc.dtype, device=features_dc.device)
    bias_t = torch.tensor(bias, dtype=features_dc.dtype, device=features_dc.device)

    # Approximate SH-DC to RGB mapping used elsewhere in this module.
    rgb = features_dc * 0.28 + 0.5
    rgb_enhanced = torch.clamp(rgb * gain_t + bias_t, 0.0, 1.0)
    return (rgb_enhanced - 0.5) / 0.28


def save_enhanced_checkpoint(input_ckpt, output_ckpt_path, gain, bias):
    """Save a new checkpoint with enhanced appearance encoded in features_dc."""
    if 'pipeline' not in input_ckpt:
        raise ValueError("Checkpoint must contain 'pipeline' key")

    pipeline_dict = input_ckpt['pipeline']
    if '_model.features_dc' not in pipeline_dict:
        raise KeyError("Checkpoint pipeline does not contain '_model.features_dc'")

    updated_features_dc = apply_rgb_transform_to_features_dc(
        pipeline_dict['_model.features_dc'],
        gain,
        bias,
    )
    pipeline_dict['_model.features_dc'] = updated_features_dc

    input_ckpt['enhancement_metadata'] = {
        'generator': 'DG_3DPlace_Enhancement/enhance_3dgs_scene.py',
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'transform_type': 'global_rgb_affine_on_features_dc',
        'gain': [float(x) for x in gain],
        'bias': [float(x) for x in bias],
    }

    os.makedirs(os.path.dirname(output_ckpt_path) or '.', exist_ok=True)
    torch.save(input_ckpt, output_ckpt_path)


def render_gaussians(gaussians, camera_params, width=1280, height=720, device='cuda'):
    """
    Render Gaussians using gsplat library
    
    Returns: (image, depth_map)
    """
    try:
        import gsplat
        from gsplat import rasterization
    except ImportError:
        print("⚠️  gsplat not available, using fallback rendering")
        return render_gaussians_fallback(gaussians, width, height)
    
    print(f"🎨 Rendering {width}×{height} with gsplat...")
    
    device = torch.device(device)
    
    # Extract camera parameters
    intrinsics = camera_params['intrinsics']  # (3, 3)
    extrinsics = camera_params['extrinsics_w2c']  # (4, 4)
    
    # Prepare Gaussians for rendering
    means = gaussians['means'].to(device).float()
    scales = gaussians['scales'].to(device).float()
    quats = gaussians['quats'].to(device).float()
    features_dc = gaussians['features_dc'].to(device).float()
    opacities = gaussians['opacities'].to(device).float()
    
    # Normalize quaternions
    quats = quats / quats.norm(dim=-1, keepdim=True)
    
    # Render using gsplat
    try:
        renders, alphas, meta = rasterization(
            means=means,
            quats=quats,
            scales=torch.exp(scales),  # scales are stored as log
            opacities=opacities,
            colors=features_dc,
            viewmats=torch.tensor(extrinsics, dtype=torch.float32, device=device).unsqueeze(0),
            Ks=torch.tensor(intrinsics, dtype=torch.float32, device=device).unsqueeze(0),
            width=width,
            height=height,
            sh_degree=None,
            backgrounds=torch.ones(1, 3, device=device),
        )
        
        rgb = renders[0].cpu().numpy()
        alpha = alphas[0].cpu().numpy()
        
        # Simple depth estimation from means z-coordinate
        depth = np.ones((height, width), dtype=np.float32) * 100.0
        
        print(f"   ✓ Rendered RGB: {rgb.shape}, range [{rgb.min():.2f}, {rgb.max():.2f}]")
        print(f"   ✓ Rendered Alpha: {alpha.shape}")
        
        return rgb, depth, alpha
        
    except Exception as e:
        print(f"   ✗ gsplat rendering failed: {e}")
        return render_gaussians_fallback(gaussians, width, height)


def render_gaussians_fallback(gaussians, width=1280, height=720):
    """Fallback: Simple projection-based rendering when gsplat unavailable"""
    print(f"🎨 Rendering {width}×{height} with fallback projection...")
    
    means = gaussians['means'].cpu().numpy()
    features_dc = gaussians['features_dc'].cpu().numpy()
    opacities = gaussians['opacities'].cpu().numpy().squeeze()
    scales = gaussians['scales'].cpu().numpy()
    
    # Simple projection: use x-y as image coords, z as depth
    image = np.ones((height, width, 3), dtype=np.float32) * 0.5
    depth = np.ones((height, width), dtype=np.float32) * 100.0
    
    # Normalize to image space
    x = means[:, 0]
    y = means[:, 1]
    z = means[:, 2]
    
    # Rough normalization
    x_norm = ((x - x.min()) / (x.max() - x.min() + 1e-6) * width).astype(int)
    y_norm = ((y - y.min()) / (y.max() - y.min() + 1e-6) * height).astype(int)
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
    
    # Rasterize Gaussians (simple version)
    for i in range(min(len(means), 100000)):
        xi, yi = x_norm[i], y_norm[i]
        
        if 0 <= xi < width and 0 <= yi < height:
            # Use SH degree 0 (DC) color
            color = np.clip(features_dc[i] * 0.28 + 0.5, 0, 1)  # SH to RGB
            
            # Gaussian splatting
            radius = max(1, int(scales[i].mean() * 20))
            y_start = max(0, yi - radius)
            y_end = min(height, yi + radius)
            x_start = max(0, xi - radius)
            x_end = min(width, xi + radius)
            
            yy, xx = np.ogrid[y_start:y_end, x_start:x_end]
            dist = (xx - xi)**2 + (yy - yi)**2
            gauss = np.exp(-dist / (2 * radius**2)) * opacities[i]
            
            # Blend
            gauss_3d = np.stack([gauss] * 3, axis=2)
            image[y_start:y_end, x_start:x_end] = (
                image[y_start:y_end, x_start:x_end] * (1 - gauss_3d) +
                color[np.newaxis, np.newaxis, :] * gauss_3d
            )
            
            # Update depth
            depth[y_start:y_end, x_start:x_end] = np.minimum(
                depth[y_start:y_end, x_start:x_end],
                z_norm[i]
            )
    
    alpha = np.ones((height, width), dtype=np.float32)
    print(f"   ✓ Fallback rendered RGB: {image.shape}, range [{image.min():.2f}, {image.max():.2f}]")
    
    return np.clip(image, 0, 1), np.clip(depth, 0, 100), alpha


def main():
    parser = argparse.ArgumentParser(
        description='Enhance 3DGS scene with shadow and lighting'
    )
    parser.add_argument(
        '--input-ckpt',
        type=str,
        default='/home/cse_g2/RealEstateGen/DG-3DPlace/DG_3DPlace_Optimization/data/inputs/scene_with_initial_object.ckpt',
        help='Path to input 3DGS checkpoint'
    )
    parser.add_argument(
        '--camera',
        type=str,
        default='/home/cse_g2/RealEstateGen/DG-3DPlace/DG_3DPlace_Optimization/data/inputs/selected_camera.pt',
        help='Path to camera parameters'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/enhanced_3dgs_render.png',
        help='Output image path'
    )
    parser.add_argument(
        '--output-ckpt',
        type=str,
        default='outputs/enhanced_3dgs_scene.ckpt',
        help='Output enhanced 3DGS checkpoint path'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=1280,
        help='Render width'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=720,
        help='Render height'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device (cuda or cpu)'
    )
    parser.add_argument(
        '--no-enhancement',
        action='store_true',
        help='Skip enhancement and just render'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("3DGS SCENE ENHANCEMENT PIPELINE")
    print("="*70)
    
    # Create output directories
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.output_ckpt) or '.', exist_ok=True)
    
    # Load camera parameters
    print(f"\n📷 Loading camera parameters: {os.path.basename(args.camera)}")
    camera_params = torch.load(args.camera, map_location='cpu', weights_only=False)
    render_width = camera_params.get('render_width', args.width)
    render_height = camera_params.get('render_height', args.height)
    print(f"   ✓ Camera resolution: {render_width}×{render_height}")
    print(f"   ✓ Camera position: {camera_params.get('position', 'N/A')}")
    
    # Load checkpoint
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    input_ckpt, gaussians = load_gsplat_checkpoint(args.input_ckpt, device=device.type)
    
    # Render scene
    rgb, depth, alpha = render_gaussians(
        gaussians,
        camera_params,
        width=render_width,
        height=render_height,
        device=device.type
    )
    
    # Save original render
    rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    orig_path = args.output.replace('.png', '_00_original.png')
    Image.fromarray(rgb_uint8).save(orig_path)
    print(f"\n✓ Original render saved: {orig_path}")
    
    # Apply enhancement
    enhanced = rgb.copy()
    shadow_mask = np.zeros_like(depth)

    if not args.no_enhancement:
        print(f"\n🎨 Applying enhancement pipeline...")
        
        # For full scene enhancement, we treat the whole render as needing enhancement
        # In a real scenario, you'd separate object and background Gaussians first
        
        try:
            pipeline = EnhancementPipeline()
            
            # Simple approach: enhance the whole rendered image
            # A more sophisticated approach would separate object and background Gaussians
            
            # Analyze scene
            scene_illum = pipeline.lighting_harmonizer.analyze_illumination(rgb)
            print(f"   ✓ Scene brightness: {scene_illum['brightness']:.3f}")
            print(f"   ✓ Light direction: {scene_illum['light_direction']}")
            
            # Generate shadows
            shadow_mask = pipeline.shadow_generator.generate_contact_shadow(
                depth,
                scene_illum['light_direction'],
                shadow_softness=3.0,
                shadow_intensity=0.3
            )
            print(f"   ✓ Shadow mask generated")
            
            # Apply light and color enhancement
            enhanced = pipeline.lighting_harmonizer.harmonize_brightness(
                rgb,
                scene_illum['brightness']
            )
            enhanced = pipeline.lighting_harmonizer.harmonize_color_temperature(
                enhanced,
                scene_illum['color_balance']
            )
            enhanced = pipeline.shadow_generator.apply_shadow_to_image(
                enhanced,
                shadow_mask,
                shadow_intensity=0.35
            )
            print(f"   ✓ Enhancement complete")
            
            # Save enhanced
            enhanced_uint8 = (np.clip(enhanced, 0, 1) * 255).astype(np.uint8)
            enh_path = args.output.replace('.png', '_01_enhanced.png')
            Image.fromarray(enhanced_uint8).save(enh_path)
            print(f"✓ Enhanced render saved: {enh_path}")
            
            # Save shadow map
            shadow_uint8 = (np.clip(shadow_mask, 0, 1) * 255).astype(np.uint8)
            shadow_path = args.output.replace('.png', '_02_shadow_map.png')
            Image.fromarray(shadow_uint8).save(shadow_path)
            print(f"✓ Shadow map saved: {shadow_path}")
            
        except Exception as e:
            print(f"   ✗ Enhancement failed: {e}")
            import traceback
            traceback.print_exc()

    # Export enhanced checkpoint
    gain, bias = compute_global_rgb_transform(rgb, enhanced)
    print(f"\n💾 Writing enhanced checkpoint...")
    print(f"   ✓ RGB gain: {gain}")
    print(f"   ✓ RGB bias: {bias}")
    save_enhanced_checkpoint(input_ckpt, args.output_ckpt, gain, bias)
    print(f"✓ Enhanced checkpoint saved: {args.output_ckpt}")
    
    print("\n" + "="*70)
    print("✓ 3DGS Scene Enhancement Complete!")
    print("="*70)
    print(f"\nNext steps:")
    print(f"  1. Use checkpoint: {args.output_ckpt}")
    print(f"  2. View previews in: {os.path.dirname(args.output) or '.'}")
    print(f"  3. Tune enhancement strength in config/default_config.yaml")
    print()


if __name__ == '__main__':
    main()
