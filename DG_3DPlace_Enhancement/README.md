# DG_3DPlace Enhancement (Simplified)

Minimal module to improve visual integration of a placed 3DGS object and export an enhanced 3DGS checkpoint.

## Kept Structure

```
DG_3DPlace_Enhancement/
├── enhance_3dgs_scene.py
├── pipeline.py
├── config/
│   ├── __init__.py
│   └── default_config.yaml
├── src/
│   ├── __init__.py
│   ├── lighting_harmonizer.py
│   ├── shadow_generator.py
│   ├── material_analyzer.py
│   └── depth_blender.py
└── utils/
    ├── __init__.py
    ├── color_matching.py
    ├── gaussian_utils.py
    └── image_processing.py
```

## Run

From the repository root:

```bash
python DG_3DPlace_Enhancement/enhance_3dgs_scene.py \
  --input-ckpt DG_3DPlace_Optimization/data/inputs/scene_with_initial_object.ckpt \
  --camera DG_3DPlace_Optimization/data/inputs/selected_camera.pt \
  --output DG_3DPlace_Enhancement/outputs/enhanced_3dgs_render.png \
  --output-ckpt DG_3DPlace_Enhancement/outputs/enhanced_3dgs_scene.ckpt
```

## Simple Web Viewer

To inspect a checkpoint in the browser, run:

```bash
python DG_3DPlace_Enhancement/viewer_3dgs_web.py \
  DG_3DPlace_Enhancement/outputs/enhanced_3dgs_scene.ckpt
```

Then open `http://127.0.0.1:8000` and use the slider to orbit the scene.

## Outputs

Main output:
- `<output-ckpt>`: enhanced 3DGS checkpoint (`.ckpt`)

Preview outputs:
- `<output>_00_original.png`
- `<output>_01_enhanced.png`
- `<output>_02_shadow_map.png`

Viewer output:
- Browser-rendered orbit view at `http://127.0.0.1:8000`

## Notes

- The checkpoint export updates `_model.features_dc` using a global RGB affine transform estimated from original vs enhanced render.
- Output folders are created automatically.
- Tune enhancement behavior in `config/default_config.yaml`.
