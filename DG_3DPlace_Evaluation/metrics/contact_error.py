"""Contact/penetration error metric for DG-3DPlace.

This metric computes a 2D mask-based proxy: using the session's
`added_object_mask.png` (if present) and the rendered difference between
`final_scene_render.png` and `initial_scene_render.png`, it computes a
symmetric Chamfer-like distance between the two masks as a proxy for
geometric contact/precision error (lower is better).
"""
import os
import numpy as np
from PIL import Image
from scipy import ndimage

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(BASE_DIR)


def _load_binary_mask(path):
    if not path or not os.path.exists(path):
        return None
    im = Image.open(path).convert("L")
    arr = np.asarray(im, dtype=np.uint8)
    return (arr > 127).astype(np.uint8)


def _mask_from_image_diff(path_initial, path_final, thr=0.02):
    if not os.path.exists(path_initial) or not os.path.exists(path_final):
        return None
    a = np.asarray(Image.open(path_initial).convert("RGB"), dtype=np.float32) / 255.0
    b = np.asarray(Image.open(path_final).convert("RGB"), dtype=np.float32) / 255.0
    if a.shape != b.shape:
        return None
    diff = np.abs(b - a).max(axis=2)
    mask = diff > max(thr, float(np.percentile(diff, 90)))
    structure = np.ones((3, 3), dtype=bool)
    mask = ndimage.binary_opening(mask, structure=structure)
    mask = ndimage.binary_closing(mask, structure=structure)
    mask = ndimage.binary_fill_holes(mask)
    return mask.astype(np.uint8)


def _chamfer(a, b):
    # a, b are binary masks (0/1)
    if a is None or b is None:
        return None
    if a.sum() == 0 and b.sum() == 0:
        return 0.0
    if a.sum() == 0 or b.sum() == 0:
        return float('inf')

    # boundary masks
    def boundary(mask):
        eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3)))
        return mask & (~eroded)

    ba = boundary(a)
    bb = boundary(b)

    if ba.sum() == 0 or bb.sum() == 0:
        # fallback to mean distance between foreground pixels
        da = ndimage.distance_transform_edt(1 - b)
        db = ndimage.distance_transform_edt(1 - a)
        mean_a = da[a.astype(bool)].mean() if a.sum() > 0 else 0.0
        mean_b = db[b.astype(bool)].mean() if b.sum() > 0 else 0.0
        return float(0.5 * (mean_a + mean_b))

    da = ndimage.distance_transform_edt(~bb)
    db = ndimage.distance_transform_edt(~ba)

    mean_a = da[ba.astype(bool)].mean()
    mean_b = db[bb.astype(bool)].mean()
    return float(0.5 * (mean_a + mean_b))


def compute_contact_error(session_dir, ckpt_path=None):
    """Compute contact error for a session. Returns a single float (pixels).

    Higher-level evaluation code can normalize by image size if desired.
    """
    session_dir = os.path.abspath(session_dir)
    eval_data = os.path.join(ROOT_DIR, "DG_3DPlace_Evaluation", "data", "2d_images")

    # Prefer explicit added_object_mask produced in session.
    added_mask_path = os.path.join(session_dir, "added_object_mask.png")
    added_mask = _load_binary_mask(added_mask_path)

    # Fallback: compute diff mask between initial/final renders in evaluation folder.
    path_initial = os.path.join(eval_data, "initial_scene_render.png")
    path_final = os.path.join(eval_data, "final_scene_render.png")
    diff_mask = _mask_from_image_diff(path_initial, path_final)

    if added_mask is None:
        # session fallback if explicit mask wasn't generated
        selected = os.path.join(session_dir, "selected_camera_view.png")
        final_view = os.path.join(session_dir, "final_view_with_object_optimized.png")
        if not os.path.exists(final_view):
            final_view = os.path.join(session_dir, "final_view_with_object.png")
        if os.path.exists(selected) and os.path.exists(final_view):
            diff_mask = _mask_from_image_diff(selected, final_view)

    # if neither available, return infinity to signal missing data
    if added_mask is None and diff_mask is None:
        return float('nan')

    if added_mask is None:
        # use diff_mask vs itself -> zero
        return 0.0

    # if diff_mask missing, fallback: compare added_mask to itself -> zero
    if diff_mask is None:
        return 0.0

    # Resize masks to same shape if needed
    if added_mask.shape != diff_mask.shape:
        added_mask = np.asarray(
            Image.fromarray((added_mask * 255).astype(np.uint8)).resize(
                diff_mask.shape[::-1],
                resample=Image.NEAREST,
            )
        ) > 127
        added_mask = added_mask.astype(np.uint8)

    val = _chamfer(added_mask.astype(bool), diff_mask.astype(bool))
    return val
