"""Networks independently specified from paper Tables A.4 and A.5."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Literal

import torch
from torch import Tensor, nn

from supermimic.normalization import (
  MLPPrecision,
  NetworkProfile,
  RunningNorm,
  run_mlp,
)
from supermimic.robot import (
  Q1Robot,
  normalize_quat,
  quat_apply,
  quat_inverse,
  quat_mul,
)
from supermimic.state_representation import (
  HeadingState,
  heading_quat,
  quaternion_log,
)

Activation = Literal["elu", "silu"]
Representation = Literal["legacy", "heading"]


def mlp(
  inputs: int,
  hidden: tuple[int, ...],
  outputs: int,
  activation: Activation = "elu",
  layer_norm: bool = False,
) -> nn.Sequential:
  layers: list[nn.Module] = []
  for width in hidden:
    layers.append(nn.Linear(inputs, width))
    if layer_norm:
      layers.append(nn.LayerNorm(width))
    layers.append(nn.SiLU() if activation == "silu" else nn.ELU())
    inputs = width
  layers.append(nn.Linear(inputs, outputs))
  return nn.Sequential(*layers)


@contextmanager
def frozen(*modules: nn.Module) -> Generator[None, None, None]:
  """Freeze weights while retaining Jacobians with respect to tensor inputs."""
  parameters = [p for m in modules for p in m.parameters()]
  old = [p.requires_grad for p in parameters]
  try:
    for p in parameters:
      p.requires_grad_(False)
    yield
  finally:
    for p, flag in zip(parameters, old, strict=True):
      p.requires_grad_(flag)


def state_scale(robot: Q1Robot) -> Tensor:
  return torch.tensor([1.0] * 29 + [3.0] * 6 + [10.0] * 22 + [1.0] * 22)


def goal_features(robot: Q1Robot, references: Tensor) -> Tensor:
  """Reference-only goal: never inject the robot's privileged root pose."""
  rotations = references[..., 3:7]
  # A deterministic hemisphere removes arbitrary reference quaternion signs.
  rotations = torch.where(rotations[..., :1] < 0, -rotations, rotations)
  return torch.cat(
    (
      references[..., :3],
      rotations,
      references[..., 7:29] - robot.default_q,
      references[..., 29:32] / 3,
      references[..., 32:35] / 3,
      references[..., 35:57] / 10,
    ),
    dim=-1,
  ).flatten(1)


class WorldModel(nn.Module):
  scale: Tensor
  context_scale: Tensor
  termination_scale: Tensor
  legacy_termination_scale: Tensor

  def __init__(
    self,
    robot: Q1Robot,
    hidden: tuple[int, ...] = (2048, 2048, 2048),
    *,
    activation: Activation = "elu",
    representation: Representation = "legacy",
    network_profile: NetworkProfile = "legacy",
  ):
    super().__init__()
    self.register_buffer("scale", state_scale(robot))
    self.register_buffer(
      "legacy_termination_scale", torch.tensor([0.25, 0.8]), persistent=False
    )
    self.representation = representation
    if network_profile not in ("legacy", "running_silu"):
      raise ValueError("Unknown network profile")
    running = network_profile == "running_silu"
    if running and representation != "heading":
      raise ValueError("Running SiLU dynamics require heading representation")
    self.network_profile = network_profile
    self.mlp_precision: MLPPrecision = "fp32"
    self.heading_state = HeadingState() if representation == "heading" else None
    self.modeled_action_delay = robot.modeled_action_delay
    if self.modeled_action_delay:
      context_scale = torch.ones(robot.context_dim)
      context_scale[robot.context_layout.delay_s] = 0.016
      self.register_buffer("context_scale", context_scale)
    inputs = robot.state_dim + 22 + robot.context_dim
    outputs = 56 if self.heading_state is not None else robot.physical_dim
    self.input_norm = RunningNorm(inputs) if running else nn.Identity()
    self.output_norm = RunningNorm(outputs, eps=1e-4) if running else nn.Identity()
    self.dynamics = mlp(
      inputs, hidden, outputs, "silu" if running else activation, layer_norm=running
    )
    self.termination = mlp(
      4 if self.heading_state is not None else 2,
      (64, 64) if self.heading_state is not None else (32, 32),
      1,
      activation,
    )
    if self.heading_state is not None:
      self.register_buffer(
        "termination_scale",
        torch.tensor(
          [
            robot.reward_scaling.termination_height,
            robot.reward_scaling.termination_tilt,
          ]
        ),
      )
    terminal_last = self.termination[-1]
    assert isinstance(terminal_last, nn.Linear)
    nn.init.constant_(terminal_last.bias, -4)
    last = self.dynamics[-1]
    assert isinstance(last, nn.Linear)
    if self.heading_state is not None:
      nn.init.zeros_(last.weight)
    else:
      nn.init.normal_(last.weight, std=0.001)
    nn.init.zeros_(last.bias)

  def input_features(self, state: Tensor, action: Tensor, context: Tensor) -> Tensor:
    if self.heading_state is not None:
      model_context = (
        context / self.context_scale if self.modeled_action_delay else context
      )
      return torch.cat((self.heading_state.features(state), action, model_context), -1)
    x = torch.cat((torch.zeros_like(state[:, :2]), state[:, 2:] / self.scale[2:]), -1)
    return torch.cat((x, action, context), -1)

  def forward(self, state: Tensor, action: Tensor, context: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
    residual = run_mlp(
      self.dynamics,
      self.input_norm(self.input_features(state, action, context)),
      self.mlp_precision,
    )
    if isinstance(self.output_norm, RunningNorm):
      residual = self.output_norm.denormalize(residual)
    if self.heading_state is not None:
      return self.heading_state.integrate(state, residual, action)
    physical = state[:, :57] + 0.02 * self.scale[:57] * residual
    return torch.cat(
      (physical[:, :3], normalize_quat(physical[:, 3:7]), physical[:, 7:], action), -1
    )

  def output_target(self, state: Tensor, successor: Tensor) -> Tensor:
    """Invert the existing heading integrator, without changing its physics."""
    if self.heading_state is None:
      return (successor[..., :57] - state[..., :57]) / (0.02 * self.scale[:57])
    rotation = normalize_quat(state[..., 3:7])
    following = normalize_quat(successor[..., 3:7])
    heading = heading_quat(rotation)
    inverse_heading = quat_inverse(heading)
    gyro = quat_apply(quat_inverse(rotation), state[..., 32:35])
    next_gyro = quat_apply(quat_inverse(following), successor[..., 32:35])
    increments = torch.cat(
      (
        quat_apply(inverse_heading, successor[..., 29:32] - state[..., 29:32]),
        next_gyro - gyro,
        successor[..., 35:57] - state[..., 35:57],
        quat_apply(
          inverse_heading,
          successor[..., :3] - state[..., :3] - 0.02 * successor[..., 29:32],
        ),
        quaternion_log(quat_mul(quat_inverse(rotation), following)) - 0.02 * next_gyro,
        successor[..., 7:29] - state[..., 7:29] - 0.02 * successor[..., 35:57],
      ),
      -1,
    )
    return increments / self.heading_state.output_scale

  def advance_context(self, context: Tensor, executed_action: Tensor) -> Tensor:
    if not self.modeled_action_delay:
      return context
    return torch.cat((context[..., :-22], executed_action), -1)

  def termination_logits(self, robot: Q1Robot, state: Tensor, target: Tensor) -> Tensor:
    if self.heading_state is not None:
      features = robot.termination_features(state, target) / self.termination_scale
      return run_mlp(
        self.termination, torch.cat((features, features.abs()), -1), self.mlp_precision
      ).squeeze(-1)
    features = robot.termination_features(
      state, target
    ) / self.legacy_termination_scale.to(state)
    return run_mlp(self.termination, features, self.mlp_precision).squeeze(-1)


class MotionPolicy(nn.Module):
  obs_scale: Tensor

  def __init__(
    self,
    robot: Q1Robot,
    hidden: tuple[int, ...] = (2048, 1024, 512, 256),
    *,
    activation: Activation = "elu",
    network_profile: NetworkProfile = "legacy",
  ):
    super().__init__()
    if network_profile not in ("legacy", "running_silu"):
      raise ValueError("Unknown network profile")
    self.network_profile = network_profile
    self.mlp_precision: MLPPrecision = "fp32"
    running = network_profile == "running_silu"
    activation = "elu" if running else activation
    self.obs_norm = RunningNorm(robot.obs_dim) if running else nn.Identity()
    self.goal_norm = RunningNorm(10 * robot.physical_dim) if running else nn.Identity()
    self.encoder = mlp(10 * robot.physical_dim, hidden, 32, activation)
    self.decoder = mlp(32 + 10 * robot.obs_dim, hidden, 22, activation)
    self.register_buffer(
      "obs_scale", torch.tensor([1.0] * 22 + [10.0] * 22 + [3.0] * 3 + [1.0] * 25)
    )
    last = self.decoder[-1]
    assert isinstance(last, nn.Linear)
    nn.init.normal_(last.weight, std=0.001)
    nn.init.zeros_(last.bias)

  def forward(self, history: Tensor, goal: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
    return self.decode(history, self.encode_goal(goal))

  def encode_goal(self, goal: Tensor) -> Tensor:
    return run_mlp(self.encoder, self.goal_norm(goal), self.mlp_precision).tanh()

  def decode(self, history: Tensor, encoding: Tensor) -> Tensor:
    features = self.obs_norm(history / self.obs_scale).flatten(1)
    return run_mlp(
      self.decoder, torch.cat((encoding, features), -1), self.mlp_precision
    )


class Critic(nn.Module):
  scale: Tensor

  def __init__(
    self,
    robot: Q1Robot,
    hidden: tuple[int, ...] = (256, 128),
    *,
    activation: Activation = "elu",
    network_profile: NetworkProfile = "legacy",
  ):
    super().__init__()
    self.register_buffer("scale", state_scale(robot))
    if network_profile not in ("legacy", "running_silu"):
      raise ValueError("Unknown network profile")
    self.network_profile = network_profile
    self.mlp_precision: MLPPrecision = "fp32"
    running = network_profile == "running_silu"
    activation = "elu" if running else activation
    inputs = robot.state_dim + 10 * robot.physical_dim + robot.context_dim
    self.input_norm = RunningNorm(inputs) if running else nn.Identity()
    self.network = mlp(
      inputs,
      hidden,
      1,
      activation,
    )

  def forward(self, state: Tensor, goal: Tensor, context: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
    return run_mlp(
      self.network,
      self.input_norm(self.input_features(state, goal, context)),
      self.mlp_precision,
    ).squeeze(-1)

  def input_features(self, state: Tensor, goal: Tensor, context: Tensor) -> Tensor:
    return torch.cat((state / self.scale, goal, context), -1)
