"""Opt-in CUDA event and host timings during the steady measurement window."""

from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from statistics import median
from time import perf_counter
from typing import Any

import torch

GIB = 1024**3


class CudaDiagnostics:
  def __init__(self, device: str):
    self.device = device
    self.active = False
    self.started: float | None = None
    self.initial_steps = 0
    self.steps = 0
    self.iterations = 0
    self.wall_seconds = 0.0
    self.min_free_gib = float("inf")
    self.pending: list[tuple[str, float, torch.cuda.Event, torch.cuda.Event]] = []
    self.host_ms: dict[str, float] = defaultdict(float)
    self.stream_ms: dict[str, float] = defaultdict(float)

  def begin(self, steps: int) -> None:
    if self.started is None:
      torch.cuda.reset_peak_memory_stats(self.device)
      self.started = perf_counter()
      self.initial_steps = steps
    self.active = True

  @contextmanager
  def timed(self, label: str) -> Generator[None]:
    if not self.active:
      yield
      return
    stream = torch.cuda.current_stream(self.device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    before = perf_counter()
    try:
      yield
    finally:
      elapsed_ms = (perf_counter() - before) * 1000
      end.record(stream)
      self.pending.append((label, elapsed_ms, start, end))

  def finish(self, steps: int) -> None:
    assert self.started is not None
    self.wall_seconds = perf_counter() - self.started
    self.iterations += 1
    self.steps = steps - self.initial_steps
    self.active = False
    # The outer profiling phases already synchronize at their boundaries.
    for label, host_ms, start, end in self.pending:
      self.host_ms[label] += host_ms
      self.stream_ms[label] += start.elapsed_time(end)
    self.pending.clear()
    self.min_free_gib = min(
      self.min_free_gib, torch.cuda.mem_get_info(self.device)[0] / GIB
    )

  def result(self, rank: int) -> dict[str, Any]:
    return {
      "rank": rank,
      "iterations": self.iterations,
      "env_steps": self.steps,
      "wall_seconds": self.wall_seconds,
      "host_ms_per_iteration": {
        key: value / self.iterations for key, value in self.host_ms.items()
      },
      "cuda_stream_ms_per_iteration": {
        key: value / self.iterations for key, value in self.stream_ms.items()
      },
      "torch_peak_allocated_gib": torch.cuda.max_memory_allocated(self.device) / GIB,
      "torch_peak_reserved_gib": torch.cuda.max_memory_reserved(self.device) / GIB,
      "min_sampled_device_free_gib": self.min_free_gib,
    }


def summarize_ranks(ranks: list[dict[str, Any]]) -> dict[str, Any]:
  summary: dict[str, Any] = {}
  for key in (
    "phase_wall_ms_per_iteration",
    "host_ms_per_iteration",
    "cuda_stream_ms_per_iteration",
  ):
    labels = sorted({name for rank in ranks for name in rank[key]})
    summary[key] = {}
    for label in labels:
      values = [(rank["rank"], rank[key].get(label, 0.0)) for rank in ranks]
      slowest, largest = max(values, key=lambda value: value[1])
      summary[key][label] = {
        "median": round(median(value for _, value in values), 3),
        "max": round(largest, 3),
        "max_rank": slowest,
      }
  return summary | {
    "env_steps_per_second": sum(rank["env_steps"] for rank in ranks)
    / max(rank["wall_seconds"] for rank in ranks),
    "min_sampled_device_free_gib": min(
      rank["min_sampled_device_free_gib"] for rank in ranks
    ),
  }
