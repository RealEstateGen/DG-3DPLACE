"""Convert DG-3DPlace .ckpt files to standard 3DGS .ply format."""

import os
import torch
import numpy as np
from plyfile import PlyData, PlyElement


def ckpt_to_ply(ckpt_path: str, ply_path: str):
    print(f"[*] Loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pipe = ckpt["pipeline"]

    means = pipe["_model.means"].numpy()           # [N, 3]
    scales = pipe["_model.scales"].numpy()         # [N, 3]
    quats = pipe["_model.quats"].numpy()           # [N, 4]
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True).clip(min=1e-8)
    features_dc = pipe["_model.features_dc"].numpy()  # [N, 3]
    opacities = pipe["_model.opacities"].numpy()   # [N, 1]

    N = means.shape[0]

    # Center the scene at the origin so the viewer's default orbit point (0,0,0) matches
    # the scene centroid. The webapp uses @mkkellogg/gaussian-splats-3d which defaults
    # to initialCameraLookAt=[0,0,0]; without this shift the camera orbits empty space.
    sig_op = 1.0 / (1.0 + np.exp(-opacities.squeeze()))
    centroid = np.average(means, weights=sig_op, axis=0)
    means = means - centroid
    print(f"    centroid shift: {centroid.round(4)}")

    # features_rest may be shorter than N (object Gaussians have no SH rest)
    features_rest_raw = pipe["_model.features_rest"].numpy()  # [M, 15, 3]
    M, num_coeffs, channels = features_rest_raw.shape
    if M < N:
        pad = np.zeros((N - M, num_coeffs, channels), dtype=features_rest_raw.dtype)
        features_rest_raw = np.concatenate([features_rest_raw, pad], axis=0)

    # Transpose [N, 15, 3] → [N, 3, 15] → [N, 45] to match 3DGS PLY convention
    features_rest = features_rest_raw.transpose(0, 2, 1).reshape(N, -1)

    # Normals (always zero in 3DGS)
    normals = np.zeros((N, 3), dtype=np.float32)

    # Build PLY attributes list
    attrs = [
        ("x", means[:, 0]),
        ("y", means[:, 1]),
        ("z", means[:, 2]),
        ("nx", normals[:, 0]),
        ("ny", normals[:, 1]),
        ("nz", normals[:, 2]),
        ("f_dc_0", features_dc[:, 0]),
        ("f_dc_1", features_dc[:, 1]),
        ("f_dc_2", features_dc[:, 2]),
    ]
    for i in range(features_rest.shape[1]):
        attrs.append((f"f_rest_{i}", features_rest[:, i]))
    attrs.append(("opacity", opacities[:, 0]))
    attrs.append(("scale_0", scales[:, 0]))
    attrs.append(("scale_1", scales[:, 1]))
    attrs.append(("scale_2", scales[:, 2]))
    attrs.append(("rot_0", quats[:, 0]))
    attrs.append(("rot_1", quats[:, 1]))
    attrs.append(("rot_2", quats[:, 2]))
    attrs.append(("rot_3", quats[:, 3]))

    dtype = [(name, "f4") for name, _ in attrs]
    arr = np.empty(N, dtype=dtype)
    for name, data in attrs:
        arr[name] = data.astype(np.float32)

    el = PlyElement.describe(arr, "vertex")
    PlyData([el]).write(ply_path)
    print(f"[+] Saved {N:,} Gaussians → {ply_path}")


CONVERSIONS = [
    (
        "data/outputs/scene_refined.ckpt",
        "data/ply_exports/gtn_chair_scene.ply",
    ),
    (
        "data/outputs/scene_refined_car.ckpt",
        "data/ply_exports/asith_garden_car_scene.ply",
    ),
    (
        "data/outputs/scene_refined_bear.ckpt",
        "data/ply_exports/asith_garden_bear_scene.ply",
    ),
]

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base, "data/ply_exports"), exist_ok=True)
    for ckpt_rel, ply_rel in CONVERSIONS:
        ckpt_to_ply(
            os.path.join(base, ckpt_rel),
            os.path.join(base, ply_rel),
        )
    print("\n[✓] All conversions complete.")
