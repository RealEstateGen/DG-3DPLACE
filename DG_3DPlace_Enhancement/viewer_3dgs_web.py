#!/usr/bin/env python3
"""Very small web viewer for a 3DGS checkpoint.

Serves a single HTML page with a slider to orbit the checkpoint.
Automatically tries the next port if the requested one is busy.
"""

from __future__ import annotations

import argparse
import errno
import json
import math
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DG_3DPlace_Enhancement.enhance_3dgs_scene import load_gsplat_checkpoint, render_gaussians


class SceneCamera:
    """Camera with intrinsics and extrinsics for rendering."""
    
    def __init__(self, position, lookat, fov_deg, width, height):
        self.position = np.array(position, dtype=np.float64)
        self.lookat = np.array(lookat, dtype=np.float64)
        self.width = int(width)
        self.height = int(height)

        fov_rad = math.radians(float(fov_deg))
        fy = (height / 2.0) / math.tan(fov_rad / 2.0)
        fx = fy
        cx = width / 2.0
        cy = height / 2.0
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        forward = self.lookat - self.position
        forward = forward / (np.linalg.norm(forward) + 1e-6)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            right = np.cross(forward, world_up)
        right = right / (np.linalg.norm(right) + 1e-6)
        up = np.cross(right, forward)
        up = up / (np.linalg.norm(up) + 1e-6)

        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = np.column_stack([right, up, -forward])
        c2w[:3, 3] = self.position

        w2c = np.linalg.inv(c2w)
        w2c[1, :] *= -1
        w2c[2, :] *= -1
        self.w2c = w2c.astype(np.float32)


def load_scene(ckpt_path: str, device: str = "cpu"):
    """Load checkpoint and compute scene bounds."""
    ckpt, gaussians = load_gsplat_checkpoint(ckpt_path, device=device)
    means = gaussians["means"].detach().cpu().numpy()
    opacities = gaussians["opacities"].detach().cpu().numpy().squeeze()
    
    opacity_mask = 1.0 / (1.0 + np.exp(-opacities)) > 0.1
    visible_means = means[opacity_mask] if np.any(opacity_mask) else means

    scene_center = visible_means.mean(axis=0)
    extent = visible_means.max(axis=0) - visible_means.min(axis=0)
    orbit_radius = float(np.linalg.norm(extent)) * 1.5
    if not np.isfinite(orbit_radius) or orbit_radius < 1e-3:
        orbit_radius = 2.0

    return ckpt, gaussians, scene_center, orbit_radius


def render_angle(gaussians, scene_center, orbit_radius, angle_deg, width, height, device):
    """Render the scene at a given orbit angle."""
    angle = math.radians(float(angle_deg))
    camera_position = np.array([
        scene_center[0] + orbit_radius * math.cos(angle),
        scene_center[1] + orbit_radius * math.sin(angle),
        scene_center[2] + orbit_radius * 0.35,
    ])
    camera = SceneCamera(camera_position, scene_center, 55.0, width, height)
    rgb, _, _ = render_gaussians(
        gaussians,
        {"intrinsics": camera.K, "extrinsics_w2c": camera.w2c},
        width=width,
        height=height,
        device=device,
    )
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def build_html(title: str) -> str:
    """Build the HTML page."""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{ margin: 0; font-family: system-ui, sans-serif; background: #111; color: #eee; }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 16px; }}
    .card {{ background: #1b1b1b; border: 1px solid #2a2a2a; border-radius: 14px; padding: 16px; }}
    img {{ width: 100%; height: auto; display: block; border-radius: 12px; background: #000; }}
    .row {{ display: flex; gap: 12px; align-items: center; margin: 12px 0 16px; flex-wrap: wrap; }}
    input[type=range] {{ flex: 1; min-width: 240px; }}
    code {{ background: #262626; padding: 2px 6px; border-radius: 6px; }}
    button {{ background: #2f6feb; color: white; border: 0; border-radius: 10px; padding: 10px 14px; cursor: pointer; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{title}</h1>
    <div class="card">
      <div class="row">
        <label for="angle">Orbit angle</label>
        <input id="angle" type="range" min="0" max="359" value="0" />
        <code id="value">0°</code>
        <button id="reset">Reset</button>
      </div>
      <img id="view" alt="3DGS view" />
    </div>
  </div>
  <script>
    const slider = document.getElementById('angle');
    const value = document.getElementById('value');
    const view = document.getElementById('view');
    const reset = document.getElementById('reset');
    function refresh() {{
      value.textContent = `${{slider.value}}°`;
      view.src = `/render?angle=${{slider.value}}&t=${{Date.now()}}`;
    }}
    slider.addEventListener('input', refresh);
    reset.addEventListener('click', () => {{ slider.value = 0; refresh(); }});
    refresh();
  </script>
</body>
</html>"""


class ViewerState:
    """Holds loaded checkpoint and scene data."""
    
    def __init__(self, ckpt_path: str, width: int, height: int, device: str):
        self.ckpt_path = ckpt_path
        self.width = width
        self.height = height
        self.device = device
        _, self.gaussians, self.scene_center, self.orbit_radius = load_scene(ckpt_path, device=device)


class ViewerHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the viewer."""
    
    state: ViewerState = None  # type: ignore[assignment]

    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(build_html(f"3DGS Viewer - {os.path.basename(self.state.ckpt_path)}"))
            return

        if parsed.path == "/render":
            query = parse_qs(parsed.query)
            angle = float(query.get("angle", ["0"])[0])
            img = render_angle(
                self.state.gaussians,
                self.state.scene_center,
                self.state.orbit_radius,
                angle,
                self.state.width,
                self.state.height,
                self.state.device,
            )
            self._send_png(img)
            return

        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"Not found")

    def log_message(self, format, *args):  # noqa: A003
        """Suppress log messages."""
        return

    def _send_html(self, html: str):
        """Send HTML response."""
        payload = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_png(self, image: np.ndarray):
        """Send PNG image response."""
        from io import BytesIO
        buffer = BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")
        payload = buffer.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def bind_server(host: str, port: int, handler_cls, port_tries: int):
    """Try to bind to a port, falling back to the next available port."""
    last_exc = None
    for offset in range(max(1, port_tries)):
        candidate = port + offset
        try:
            server = HTTPServer((host, candidate), handler_cls)
            if offset > 0:
                print(f"Requested port {port} is busy, using {candidate} instead")
            return server, candidate
        except OSError as exc:
            last_exc = exc
            if exc.errno != errno.EADDRINUSE:
                raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Failed to bind HTTP server")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Tiny browser viewer for a 3DGS checkpoint")
    parser.add_argument("checkpoint", help="Path to the .ckpt file")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--width", type=int, default=1280, help="Render width")
    parser.add_argument("--height", type=int, default=720, help="Render height")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Render device")
    parser.add_argument("--port-tries", type=int, default=20, help="How many ports to try if first is busy")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    ViewerHandler.state = ViewerState(args.checkpoint, args.width, args.height, args.device)
    server, actual_port = bind_server(args.host, args.port, ViewerHandler, args.port_tries)
    
    print(f"Viewer running at http://{args.host}:{actual_port}")
    print(f"Loading checkpoint: {args.checkpoint}")
    server.serve_forever()


if __name__ == "__main__":
    main()
