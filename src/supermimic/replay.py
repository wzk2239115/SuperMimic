"""Per-environment temporal replay with terminal next states kept before reset."""

from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import Tensor


@dataclass
class ReplayBatch:
  states: Tensor
  next_states: Tensor
  actions: Tensor
  context: Tensor
  terminated: Tensor
  history: Tensor
  clips: Tensor
  frames: Tensor
  valid: Tensor | None = None

  def to(self, device: str) -> "ReplayBatch":
    tensors = {field.name: getattr(self, field.name) for field in fields(self)}
    return ReplayBatch(
      **{
        name: tensor.to(device) if tensor is not None else None
        for name, tensor in tensors.items()
      }
    )


class TemporalReplay:
  def __init__(
    self,
    capacity: int,
    num_envs: int,
    state_dim: int,
    context_dim: int,
    obs_dim: int,
    device: str,
  ):
    self.capacity = capacity
    self.num_envs = num_envs
    self.cursor = 0
    self.count = 0
    self.device = device
    self.states = torch.empty(capacity, num_envs, state_dim, device=device)
    self.next_states = torch.empty_like(self.states)
    self.context = torch.empty(capacity, num_envs, context_dim, device=device)
    self.actions = torch.empty(capacity, num_envs, 22, device=device)
    self.observations = torch.empty(capacity, num_envs, obs_dim, device=device)
    self.terminated = torch.empty(capacity, num_envs, device=device)
    self.boundaries = torch.ones(capacity, num_envs, dtype=torch.bool, device=device)
    self.clips = torch.empty(capacity, num_envs, dtype=torch.long, device=device)
    self.frames = torch.empty_like(self.clips)

  def state_dict(self) -> dict[str, Any]:
    """Save only initialized rows, in chronological order, on CPU."""
    oldest = self.cursor if self.count == self.capacity else 0
    rows = (oldest + torch.arange(self.count, device=self.device)) % self.capacity
    result: dict[str, Any] = {
      "capacity": self.capacity,
      "num_envs": self.num_envs,
      "count": self.count,
    }
    for name in self._storage_names():
      result[name] = getattr(self, name)[rows].cpu()
    return result

  @staticmethod
  def _storage_names() -> tuple[str, ...]:
    return (
      "states",
      "next_states",
      "context",
      "actions",
      "observations",
      "terminated",
      "boundaries",
      "clips",
      "frames",
    )

  def load_state_dict(self, state: dict[str, Any]) -> None:
    if state["capacity"] != self.capacity or state["num_envs"] != self.num_envs:
      raise ValueError("Replay capacity/environment count differs from checkpoint")
    count = state["count"]
    if not isinstance(count, int) or not 0 <= count <= self.capacity:
      raise ValueError("Invalid replay row count")
    for name in self._storage_names():
      target, source = getattr(self, name), state[name]
      if source.shape != (count, *target.shape[1:]) or source.dtype != target.dtype:
        raise ValueError(f"Invalid replay shard tensor: {name}")
    for name in self._storage_names():
      getattr(self, name)[:count].copy_(state[name])
    self.count, self.cursor = count, count % self.capacity
    # Physics is reset on resume. No sample may join the old and new episodes.
    if count:
      self.boundaries[(self.cursor - 1) % self.capacity] = True

  def add(
    self,
    states: Tensor,
    next_states: Tensor,
    actions: Tensor,
    context: Tensor,
    observation: Tensor,
    terminated: Tensor,
    boundary: Tensor,
    clips: Tensor,
    frames: Tensor,
  ) -> None:
    i = self.cursor
    for storage, value in (
      (self.states, states),
      (self.next_states, next_states),
      (self.actions, actions),
      (self.context, context),
      (self.observations, observation),
      (self.terminated, terminated),
      (self.boundaries, boundary),
      (self.clips, clips),
      (self.frames, frames),
    ):
      storage[i].copy_(value.detach())
    self.cursor = (self.cursor + 1) % self.capacity
    self.count = min(self.capacity, self.count + 1)

  def sample(self, batch_size: int, window: int = 8, history: int = 10) -> ReplayBatch:
    if min(batch_size, window, history) < 1:
      raise ValueError("Positive replay batch, window and history sizes required")
    if not self.count:
      raise ValueError("Replay has too few temporal samples")
    oldest = self.cursor if self.count == self.capacity else 0
    starts = torch.randint(self.count, (batch_size,), device=self.device)
    envs = torch.randint(self.num_envs, (batch_size,), device=self.device)
    env_grid = envs[:, None]
    offsets = torch.arange(window, device=self.device)
    positions = starts[:, None] + offsets
    available = positions < self.count
    rows = (oldest + positions.clamp_max(self.count - 1)) % self.capacity
    # Keep the terminal transition. Pad later rows with it, without supervising
    # them or reading the reset episode, including at the newest replay row.
    ends = torch.where(self.boundaries[rows, env_grid] & available, offsets, window)
    last = torch.minimum(ends.amin(1), self.count - 1 - starts)
    valid = offsets <= last[:, None]
    step_rows = (
      oldest + starts[:, None] + torch.minimum(offsets, last[:, None])
    ) % self.capacity

    positions = starts[:, None] + torch.arange(1 - history, 1, device=self.device)
    rows = (oldest + positions.clamp_min(0)) % self.capacity
    # A boundary belongs to the preceding transition. Missing observations
    # repeat the earliest available frame of this episode, as collection does.
    previous_end = (
      self.boundaries[rows, env_grid] & (positions >= 0) & (positions < starts[:, None])
    )
    first = torch.where(previous_end, positions + 1, 0).amax(1)
    history_rows = (oldest + torch.maximum(positions, first[:, None])) % self.capacity
    return ReplayBatch(
      self.states[step_rows, env_grid],
      self.next_states[step_rows, env_grid],
      self.actions[step_rows, env_grid],
      self.context[step_rows[:, 0], envs],
      self.terminated[step_rows, env_grid],
      self.observations[history_rows, env_grid],
      self.clips[step_rows[:, 0], envs],
      self.frames[step_rows[:, 0], envs],
      valid,
    )
