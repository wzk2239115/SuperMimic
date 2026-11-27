"""Serve a local, automatically refreshed preview of a training run."""

import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Thread
from urllib.parse import SplitResult, urlsplit

import tyro


@dataclass
class Config:
  run_dir: Path
  """Directory where the trainer writes checkpoints and metrics."""
  port: int = 8765
  """Local HTTP port. Use 0 to choose an available port."""
  notebook_url: str | None = None
  """Current JupyterLab URL. Print the browser URL through its port proxy."""
  motion_index: int = 0
  """Index of the fixed reference action in the checkpoint's motion list."""
  seconds: int = 6
  """Length of the action preview, in simulation seconds."""


@dataclass(frozen=True)
class Preview:
  checkpoint: int
  metadata: dict
  video: bytes


def latest_checkpoint(run_dir: Path) -> tuple[int, Path] | None:
  checkpoints = [
    (int(path.stem.removeprefix("checkpoint_")), path)
    for path in run_dir.glob("checkpoint_*.pt")
    if path.stem.removeprefix("checkpoint_").isdigit()
  ]
  return max(checkpoints, default=None)


def latest_iteration(run_dir: Path) -> int | None:
  try:
    with (run_dir / "metrics.jsonl").open("rb") as stream:
      size = stream.seek(0, 2)
      stream.seek(max(0, size - 8192))
      for line in reversed(stream.read().splitlines()):
        try:
          return int(json.loads(line)["iteration"])
        except (ValueError, TypeError, KeyError):
          continue
  except OSError:
    pass
  return None


class RunPreview:
  def __init__(self, cfg: Config):
    self.cfg = cfg
    self.ready: Preview | None = None
    self.rendering: int | None = None
    self.error: str | None = None
    self.stop = Event()

  def watch(self) -> None:
    while not self.stop.is_set():
      candidate = latest_checkpoint(self.cfg.run_dir)
      if candidate is not None and (
        self.ready is None or candidate[0] != self.ready.checkpoint
      ):
        self.rendering = candidate[0]
        self.error = None
        try:
          self.render(*candidate)
        except (OSError, RuntimeError, ValueError) as error:
          self.error = str(error)
          print(f"Preview: {error}", file=sys.stderr, flush=True)
          self.stop.wait(15)
        finally:
          self.rendering = None
      self.stop.wait(4)

  def render(self, iteration: int, checkpoint: Path) -> None:
    python = Path(sys.executable)
    if sys.platform == "darwin":
      python = python.with_name("mjpython")
    with tempfile.TemporaryDirectory(prefix="supermimic-preview-") as directory:
      output = Path(directory)
      with (
        (output / "render.log").open("w+") as log,
        subprocess.Popen(
          [
            str(python),
            "-m",
            "supermimic.preview_render",
            str(checkpoint),
            str(output),
            str(self.cfg.motion_index),
            str(self.cfg.seconds),
          ],
          stdout=log,
          stderr=subprocess.STDOUT,
        ) as process,
      ):
        try:
          for _ in range(360):
            if process.poll() is not None:
              break
            if self.stop.wait(0.5):
              return
          if process.poll() is None:
            raise RuntimeError("Rendering did not finish within three minutes")
          if process.returncode:
            log.seek(0)
            print(log.read().rstrip(), file=sys.stderr, flush=True)
            raise RuntimeError(f"无法生成第 {iteration} 轮预览，详见终端日志。")
        finally:
          if process.poll() is None:
            process.terminate()
            try:
              process.wait(timeout=3)
            except subprocess.TimeoutExpired:
              process.kill()
      self.ready = Preview(
        iteration,
        json.loads((output / "effect.json").read_text()),
        (output / "tracking.mp4").read_bytes(),
      )

  def handler(self) -> type[BaseHTTPRequestHandler]:
    preview = self
    html = Path(__file__).with_name("preview.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
      def do_GET(self) -> None:
        path = urlsplit(self.path).path
        snapshot = preview.ready
        if path == "/":
          self.send("text/html; charset=utf-8", html)
        elif path == "/status":
          status = {
            "iteration": latest_iteration(preview.cfg.run_dir),
            "rendering": preview.rendering,
            "error": preview.error,
            "preview": None
            if snapshot is None
            else (snapshot.metadata | {"checkpoint": snapshot.checkpoint}),
          }
          self.send("application/json", json.dumps(status).encode())
        elif snapshot is not None and path == f"/video/{snapshot.checkpoint}.mp4":
          self.send("video/mp4", snapshot.video)
        else:
          self.send_error(404)

      def send(self, mime_type: str, content: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
          self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
          pass

      def log_message(self, *args: object) -> None:
        pass

    return Handler


def notebook_base_url(value: str) -> SplitResult:
  url = urlsplit(value.strip())
  if (
    url.scheme not in ("http", "https")
    or not url.hostname
    or url.username is not None
    or url.password is not None
  ):
    raise ValueError("Paste the full HTTP(S) URL of the JupyterLab page")
  path = re.split(r"/(?:lab|tree|notebooks|terminals)(?:/|$)", url.path, maxsplit=1)[0]
  path = re.sub(r"/proxy/(?:absolute/)?[0-9]+/?$", "", path)
  return url._replace(path=path.rstrip("/"), query="", fragment="")


def main() -> None:
  cfg = tyro.cli(Config)
  if not 0 <= cfg.port <= 65535 or cfg.motion_index < 0 or not 1 <= cfg.seconds <= 30:
    raise ValueError("Use a valid port, nonnegative motion index and 1–30 seconds")
  notebook_url = cfg.notebook_url or os.environ.get("SUPERMIMIC_NOTEBOOK_URL")
  public_base = notebook_base_url(notebook_url) if notebook_url else None
  preview = RunPreview(cfg)

  def interrupt(*_: object) -> None:
    raise KeyboardInterrupt

  signal.signal(signal.SIGTERM, interrupt)
  with ThreadingHTTPServer(("127.0.0.1", cfg.port), preview.handler()) as server:
    worker = Thread(target=preview.watch)
    worker.start()
    try:
      print(f"服务器本地预览: http://127.0.0.1:{server.server_port}/", flush=True)
      if public_base is not None:
        public = public_base._replace(
          path=f"{public_base.path}/proxy/{server.server_port}/"
        )
        print(f"复制到浏览器: {public.geturl()}", flush=True)
      server.serve_forever()
    except KeyboardInterrupt:
      pass
    finally:
      preview.stop.set()
      worker.join()


if __name__ == "__main__":
  main()
