"""Export a MuJoCo rollout to GIF and TensorBoard without a graphics server."""

import io
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import tyro
from PIL import Image, ImageDraw
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.compat.proto.tensor_pb2 import TensorProto
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from tensorboard.compat.proto.types_pb2 import DT_STRING
from tensorboard.plugins.image.metadata import create_summary_metadata
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from supermimic.preview import latest_checkpoint

WIDTH, HEIGHT = 720, 280


@dataclass
class Config:
  run_dir: Path
  motion_index: int = 0
  seconds: int = 6


def render_gif(tracks: dict, failure: int | None, iteration: int) -> bytes:
  frames = []
  stride = 3
  fps = tracks["fps"]
  for frame in range(0, len(tracks["reference"]), stride):
    image = Image.new("RGB", (WIDTH, HEIGHT), "#161c26")
    draw = ImageDraw.Draw(image)
    draw.line((WIDTH // 2, 0, WIDTH // 2, HEIGHT), fill="#465367", width=1)
    draw.line((0, 244, WIDTH, 244), fill="#465367", width=1)
    for panel, track in enumerate(("reference", "actual")):
      pose = tracks[track][frame]
      root_x, root_y, _ = pose[1]
      points = []
      for x, y, z in pose:
        dx, dy = x - root_x, y - root_y
        points.append(
          (
            180 + panel * 360 + 220 * (0.72 * dx - 0.69 * dy),
            244 - 220 * (z + 0.32 * (0.69 * dx + 0.72 * dy)),
          )
        )
      failed = panel == 1 and failure is not None and frame >= failure
      color = "#ff958a" if failed else ("#94baff" if panel else "#90dac9")
      for link, parent in enumerate(tracks["parents"]):
        if parent > 0:
          draw.line((points[parent], points[link]), fill=color, width=5)
      for x, y in points[1:]:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
      title = ("POLICY - FAILED" if failed else "POLICY") if panel else "REFERENCE"
      draw.text((12 + panel * 360, 10), f"{title}   {frame / fps:.2f}s", fill=color)
    draw.text((12, 255), f"Checkpoint {iteration}", fill="#b6c1d1")
    frames.append(image.quantize(colors=64))
  encoded = io.BytesIO()
  frames[0].save(
    encoded,
    format="GIF",
    save_all=True,
    append_images=frames[1:],
    duration=round(1000 * stride / fps),
    loop=0,
    disposal=2,
  )
  return encoded.getvalue()


def main() -> None:
  cfg = tyro.cli(Config)
  if not 1 <= cfg.seconds <= 30 or cfg.motion_index < 0:
    raise ValueError("Choose 1–30 seconds and a nonnegative motion index")
  candidate = latest_checkpoint(cfg.run_dir)
  if candidate is None:
    raise FileNotFoundError(f"No checkpoints in {cfg.run_dir}")
  iteration, checkpoint = candidate
  os.environ["MUJOCO_GL"] = "disable"
  os.environ.pop("PYOPENGL_PLATFORM", None)
  from supermimic.preview_render import render

  with tempfile.TemporaryDirectory(prefix="supermimic-preview-gif-") as temp:
    scratch = Path(temp)
    render(checkpoint, scratch, cfg.motion_index, cfg.seconds, "links")
    tracks = json.loads((scratch / "links.json").read_text())
    effect = json.loads((scratch / "effect.json").read_text())
  gif = render_gif(tracks, effect["first_failure_frame"], iteration)
  folder = cfg.run_dir / "previews"
  folder.mkdir(parents=True, exist_ok=True)
  target = folder / f"checkpoint_{iteration}_motion_{cfg.motion_index}.gif"
  target.write_bytes(gif)
  log_dir = cfg.run_dir / "tensorboard" / "preview-images"
  tag = f"rollout/motion_{cfg.motion_index}"
  summary = Summary(
    value=[
      Summary.Value(
        tag=tag,
        metadata=create_summary_metadata(
          display_name=tag, description="Reference and actual MuJoCo rollout"
        ),
        tensor=TensorProto(
          dtype=DT_STRING,
          tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=3)]),
          string_val=[str(WIDTH).encode(), str(HEIGHT).encode(), gif],
        ),
      )
    ]
  )
  writer = EventFileWriter(str(log_dir))
  try:
    writer.add_event(Event(wall_time=time.time(), step=iteration, summary=summary))
  finally:
    writer.close()
  print(f"动作回放 GIF: {target.resolve()}", flush=True)
  print(f"TensorBoard Images: {tag} (step {iteration})", flush=True)
  print(f"TensorBoard Log Dir: {log_dir.resolve()}", flush=True)
