"""Hardcoded prompt presets for automated DG-3DPlace paper experiments.

Each preset includes the checkpoint to load so the runner can automate runs
across multiple scenes without changing the pipeline entry point.
"""

PROMPT_SPECS = [
    {
        "id": "bench_car_red",
        "ckpt_path": "ckpt/bench_park.ckpt",
        "prompt": "a red car in the middle",
        "target_text": "A real-estate scene with a red car in the middle",
        "object_class": "car",
    },
    {
        "id": "bench_wooden_table",
        "ckpt_path": "ckpt/bench_park.ckpt",
        "prompt": "a wooden table near the center",
        "target_text": "A real-estate scene with a wooden table near the center",
        "object_class": "table",
    },
    {
        "id": "bench_swing",
        "ckpt_path": "ckpt/bench_park.ckpt",
        "prompt": "a swing in the park",
        "target_text": "A real-estate scene with a swing in the park",
        "object_class": "swing",
    },
    {
        "id": "room_vase_table",
        "ckpt_path": "ckpt/bench_park.ckpt",
        "prompt": "a vase on the floor",
        "target_text": "A real-estate scene with a vase on the table",
        "object_class": "vase",
    },
    {
        "id": "room_sofa",
        "ckpt_path": "ckpt/bench_park.ckpt",
        "prompt": "a sofa",
        "target_text": "A real-estate scene with a sofa against the wall",
        "object_class": "sofa",
    },
]

SOURCE_TEXT = "A park scene"

BASELINE_PROMPT = {
    "id": "baseline",
    "prompt": "no added object, keep the scene unchanged",
    "target_text": "A real-estate scene",
    "object_class": "none",
}
