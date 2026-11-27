"""Independently train SuperMimic on explicit Q1 V2 XML and named Q1 motions."""

import hashlib
import importlib.metadata
import json
import math
import platform
import random
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

from supermimic.algorithm import SuperMimic
from supermimic.checkpoint import (
  StopRequest,
  capture_rng,
  load_checkpoint,
  restore_rng,
  save_checkpoint,
)
from supermimic.collection import Q1Collector
from supermimic.cuda_diagnostics import CudaDiagnostics, summarize_ranks
from supermimic.data import Q1Motions, ReturnResetSampler
from supermimic.distributed import Distributed
from supermimic.imagination_terminal import MIN_SURVIVAL, TERMINAL_HANDLING
from supermimic.models import (
  Critic,
  MotionPolicy,
  WorldModel,
)
from supermimic.normalization import MLPPrecision, NetworkProfile, RunningNorm
from supermimic.randomization import RandomizationCfg
from supermimic.replay import ReplayBatch, TemporalReplay
from supermimic.reward_tuning import RewardTuningCfg
from supermimic.robot import Q1Robot
from supermimic.simulator import Q1Simulator
from supermimic.warm_start import (
  load_warm_start,
  restore_warm_start,
  warm_start_identity,
)


@dataclass
class TrainCfg:
  robot_xml: Path
  motion_dir: Path
  output_dir: Path
  motion_pattern: str = "*.npz"
  motion_manifest: Path | None = None
  """Optional list of reference .npz paths relative to motion_dir."""
  max_motions: int = 8
  # An alphabetical prefix of a large library is one narrow slice of one corpus
  # -- the first few thousand BONES clips are neighbouring takes of the same
  # sessions. A seeded draw spreads the selection over every imported source.
  motion_selection: Literal["random", "first"] = "random"
  motion_seed: int = 0
  motion_memory_gb: float = 8.0
  motion_storage: Literal["shared", "device", "device-fp16"] = "shared"
  motion_cache_dir: Path | None = None
  device: str = "auto"
  # Environment and batch counts are totals across every rank, so a run on one
  # GPU and the same run on eight are the same experiment. Each must divide by
  # the rank count.
  num_envs: int = 8
  cpu_sim_workers: int = 1
  cpu_torch_threads: int = 4
  iterations: int = 20
  replay_steps: int = 1000
  model_batch_size: int = 128
  policy_batch_size: int = 128
  model_batches: int = 10
  horizon: int = 32
  # Default zero retains existing runs; fresh example uses 500.
  model_warmup_iterations: int = 0
  critic_epochs: int = 5
  critic_minibatches: int = 4
  learning_rate: float = 2e-4
  action_scale: float = 0.5
  use_tn: bool = True
  action_noise: float = 0.0
  clean_policy_actions: bool = False
  action_lag: bool = False
  modeled_action_delay: bool = False
  network_activation: Literal["elu", "silu"] = "elu"
  dynamics_representation: Literal["legacy", "heading"] = "legacy"
  network_profile: NetworkProfile = "legacy"
  mlp_precision: MLPPrecision = "fp32"
  normalize_policy_objective: bool = False
  normalize_model_objective: bool = False
  randomize: bool = True
  small_networks: bool = False
  save_interval: int = 100
  initial_save_iteration: int = 0
  seed: int = 0
  profile_iterations: int = 0
  profile_every: int = 0
  profile_cuda_ranks: bool = False
  compile_networks: bool = False
  compile_collection: bool = False
  fuse_model_loss: bool = False
  cache_reference_heads: bool = False
  cache_goals: bool = False
  time_major_rollout: bool = False
  capture_reset: bool = False
  capture_imagination: bool = False
  compile_critic: bool = False
  fused_optimizers: bool = False
  physics_timestep: float | None = None
  q1_reward_scaling: bool = False
  reward_arm_joint_weight: float = 0.0
  reward_arm_pd_target_weight: float = 0.0
  reward_arm_error_std: float = 0.8
  reward_body_orientation_std: float | None = None
  # Defer tensor/gradient checks to the learning-update boundary to avoid a
  # host/device synchronization for every world-model and critic minibatch.
  runtime_checks: Literal["per_minibatch", "per_update"] = "per_update"
  collective_timeout_seconds: int = 300
  resume: Path | None = None
  warm_start: Path | None = None
  warm_start_progress: bool = False
  checkpoint_replay: bool = False
  resume_replay: bool = True


def available_motions(cfg: TrainCfg) -> list[Path]:
  if cfg.motion_manifest is None:
    return sorted(cfg.motion_dir.glob(cfg.motion_pattern))
  paths = []
  seen: set[Path] = set()
  for line in cfg.motion_manifest.read_text(encoding="utf-8").splitlines():
    entry = line.partition("#")[0].strip()
    if not entry:
      continue
    path = (cfg.motion_dir / entry).resolve()
    if path.suffix != ".npz" or not path.is_relative_to(cfg.motion_dir.resolve()):
      raise ValueError(f"Expected a .npz inside the motion directory: {entry}")
    if path in seen or not path.is_file():
      raise ValueError(f"Missing or duplicate motion in manifest: {entry}")
    paths.append(path)
    seen.add(path)
  if not paths:
    raise ValueError(f"No motions in {cfg.motion_manifest}")
  return paths


def select_motions(cfg: TrainCfg, available: list[Path]) -> list[Path]:
  """Choose which of the library's clips this run trains on.

  Drawn from the sorted listing with an explicit seed, so every rank of a
  data-parallel run selects the same clips and identity.json alone is enough to
  reproduce the choice.
  """
  if not cfg.max_motions or cfg.max_motions >= len(available):
    return available
  if cfg.motion_selection == "first":
    return available[: cfg.max_motions]
  order = random.Random(cfg.motion_seed).sample(range(len(available)), cfg.max_motions)
  return [available[index] for index in sorted(order)]


def rank_configuration(
  cfg: TrainCfg, group: Distributed, visible_gpus: int
) -> TrainCfg:
  """Resolve auto separately from an explicit CPU request, then split totals."""
  if cfg.device not in ("cpu", "mps", "auto") and not cfg.device.startswith("cuda"):
    raise ValueError("Use auto, cpu, mps or cuda")
  if cfg.device == "auto":
    cfg = replace(cfg, device="cuda:0" if visible_gpus else "cpu")
  if cfg.device == "mps" and group.enabled:
    raise ValueError("MPS training requires one process")
  if group.enabled:
    # The launcher assigns the GPU; a device in the config would make every rank
    # crowd onto the same one. An explicit CPU device is left alone, so the
    # collective ordering can be validated without reserving GPUs.
    if cfg.device != "cpu":
      cfg = replace(cfg, device=f"cuda:{group.local_rank}")
    cfg = replace(
      cfg,
      num_envs=group.share(cfg.num_envs, "Environment count"),
      model_batch_size=group.share(cfg.model_batch_size, "Model batch"),
      policy_batch_size=group.share(cfg.policy_batch_size, "Policy batch"),
    )
  if cfg.device.startswith("cuda"):
    index = torch.device(cfg.device).index or 0
    if not 0 <= index < visible_gpus:
      raise ValueError(f"Device {cfg.device} is outside {visible_gpus} visible GPUs")
    cfg = replace(cfg, device=f"cuda:{index}")
  return cfg


def train(cfg: TrainCfg) -> Path:
  requested = cfg
  group = Distributed.from_environment(validation_mode=cfg.runtime_checks)
  cfg = rank_configuration(cfg, group, torch.cuda.device_count())
  group.initialize(cfg.device, cfg.collective_timeout_seconds)
  try:
    with StopRequest() as stop:
      return _train(cfg, group, requested, stop)
  finally:
    group.shutdown()


def _train(
  cfg: TrainCfg, group: Distributed, requested: TrainCfg, stop: StopRequest
) -> Path:
  if (
    min(
      cfg.num_envs,
      cfg.iterations,
      cfg.model_batch_size,
      cfg.policy_batch_size,
      cfg.model_batches,
      cfg.horizon,
      cfg.critic_epochs,
      cfg.critic_minibatches,
      cfg.save_interval,
    )
    < 1
  ):
    raise ValueError("Counts, batches and intervals must be positive")
  if cfg.model_warmup_iterations < 0:
    raise ValueError("Model warmup must be nonnegative")
  if cfg.cpu_torch_threads < 1 or cfg.cpu_sim_workers < 1:
    raise ValueError("CPU worker and Torch thread counts must be positive")
  if cfg.replay_steps < 10 or cfg.max_motions < 0 or cfg.learning_rate <= 0:
    raise ValueError("Invalid replay size, motion limit or learning rate")
  if cfg.motion_memory_gb <= 0:
    raise ValueError("Allow a positive motion state budget")
  if cfg.profile_every < 0 or not 0 <= cfg.profile_iterations <= cfg.iterations:
    raise ValueError("Profile at most every iteration of this run")
  if cfg.profile_cuda_ranks and (
    not cfg.device.startswith("cuda") or not cfg.profile_iterations
  ):
    raise ValueError("Per-rank CUDA diagnostics require CUDA and profile_iterations")
  profiler = CudaDiagnostics(cfg.device) if cfg.profile_cuda_ranks else None
  group.cuda_diagnostics = profiler
  mps = cfg.device == "mps"
  if mps:
    if not torch.backends.mps.is_available():
      raise RuntimeError("MPS is unavailable in this process")
    if any(
      (
        cfg.compile_networks,
        cfg.compile_collection,
        cfg.capture_imagination,
        cfg.capture_reset,
      )
    ):
      raise ValueError("MPS training uses eager execution without graph capture")
    if cfg.mlp_precision != "fp32":
      raise ValueError("Select fp32 for MPS training")
  if cfg.action_noise < 0 or cfg.action_scale <= 0:
    raise ValueError("Invalid Q1 action noise or scale")
  if cfg.runtime_checks not in ("per_minibatch", "per_update"):
    raise ValueError("Use per_minibatch or per_update runtime checks")
  if cfg.resume is not None and cfg.warm_start is not None:
    raise ValueError("Use resume or weights-only warm_start, not both")
  if cfg.warm_start_progress and cfg.warm_start is None:
    raise ValueError("warm_start_progress requires a weights-only warm_start source")
  reward_tuning = RewardTuningCfg(
    arm_joint_weight=cfg.reward_arm_joint_weight,
    arm_pd_target_weight=cfg.reward_arm_pd_target_weight,
    arm_error_std=cfg.reward_arm_error_std,
    body_orientation_std=cfg.reward_body_orientation_std,
    action_scale=cfg.action_scale,
  )
  with group.checked("run directory"):
    if cfg.output_dir.exists():
      raise FileExistsError("Use a fresh output directory, including when resuming")
  resumed, rank_state, warm_started = None, None, None
  if cfg.resume is not None:
    with group.checked("checkpoint loading"):
      resumed, rank_state = load_checkpoint(cfg.resume, group)
  if cfg.warm_start is not None:
    with group.checked("weights-only checkpoint loading"):
      warm_started = load_warm_start(cfg.warm_start)
  # Identical weights, then divergent exploration: every rank builds the same
  # networks from the same seed, and only afterwards offsets its own stream so
  # the ranks do not simulate the same environments.
  torch.manual_seed(cfg.seed)
  if cfg.device == "cpu" or mps:
    torch.set_num_threads(cfg.cpu_torch_threads)
  else:
    torch.backends.cuda.matmul.allow_tf32 = True
  with group.checked("Q1 assets and motion preparation"):
    robot = Q1Robot(
      cfg.robot_xml,
      reward_scaling="q1_proportional" if cfg.q1_reward_scaling else None,
      modeled_action_delay=cfg.modeled_action_delay,
    ).to(cfg.device)
    available = available_motions(cfg)
    available_count = len(available)
    paths = select_motions(cfg, available)
    motions = Q1Motions(
      robot,
      paths,
      device=cfg.device,
      memory_budget_gb=cfg.motion_memory_gb,
      storage=cfg.motion_storage,
      cache_dir=cfg.motion_cache_dir,
    )
    lengths = motions.lengths()
    motions.prepare()
  sampler = ReturnResetSampler(lengths, motions.dt)
  model_hidden = (64, 64) if cfg.small_networks else (2048, 2048, 2048)
  actor_hidden = (64, 64) if cfg.small_networks else (2048, 1024, 512, 256)
  critic_hidden = (64, 32) if cfg.small_networks else (256, 128)
  with group.checked("network allocation"):
    model = WorldModel(
      robot,
      model_hidden,
      activation=cfg.network_activation,
      representation=cfg.dynamics_representation,
      network_profile=cfg.network_profile,
    ).to(cfg.device)
    actor = MotionPolicy(
      robot,
      actor_hidden,
      activation=cfg.network_activation,
      network_profile=cfg.network_profile,
    ).to(cfg.device)
    critic = Critic(
      robot,
      critic_hidden,
      activation=cfg.network_activation,
      network_profile=cfg.network_profile,
    ).to(cfg.device)
    if cfg.mlp_precision not in ("fp32", "bf16"):
      raise ValueError("Unknown MLP precision")
    precision = cfg.mlp_precision if cfg.device != "cpu" else "fp32"
    if precision == "bf16":
      with torch.cuda.device(cfg.device):
        if not torch.cuda.is_bf16_supported():
          raise ValueError("GPU does not support bf16; select fp32 explicitly")
    for network in (model, actor, critic):
      network.mlp_precision = precision
    algorithm = SuperMimic(
      robot,
      motions,
      model,
      actor,
      critic,
      cfg.learning_rate,
      horizon=cfg.horizon,
      distributed=group,
      cache_reference_heads=cfg.cache_reference_heads,
      cache_goals=cfg.cache_goals,
      time_major_rollout=cfg.time_major_rollout,
      fuse_model_loss=cfg.fuse_model_loss,
      reward_tuning=reward_tuning,
      clean_policy_actions=cfg.clean_policy_actions,
      normalize_policy_objective=cfg.normalize_policy_objective,
      normalize_model_objective=cfg.normalize_model_objective,
      capture_imagination=cfg.capture_imagination,
      compile_critic=cfg.compile_critic,
      fused_optimizers=cfg.fused_optimizers,
    )
    if cfg.compile_networks:
      algorithm.enable_compile()
  collection_device = "cpu" if mps else cfg.device
  with group.checked("simulator and replay allocation"):
    physics_robot = robot
    collection_actor = actor
    collection_motions = motions
    if mps:
      physics_robot = Q1Robot(
        cfg.robot_xml,
        reward_scaling="q1_proportional" if cfg.q1_reward_scaling else None,
        modeled_action_delay=cfg.modeled_action_delay,
      )
      collection_actor = MotionPolicy(
        physics_robot,
        actor_hidden,
        activation=cfg.network_activation,
        network_profile=cfg.network_profile,
      )
      if motions.device != "cpu":
        collection_motions = Q1Motions(
          physics_robot, paths, storage="shared", cache_dir=cfg.motion_cache_dir
        )
        collection_motions.prepare()
    collector = Q1Collector(
      physics_robot,
      collection_actor,
      motions.dt,
      reward_tuning=reward_tuning,
      clean_policy_actions=cfg.clean_policy_actions,
    )
    if cfg.compile_collection:
      collector.enable_compile()
    simulator = Q1Simulator(
      physics_robot,
      cfg.num_envs,
      collection_device,
      cfg.action_scale,
      randomization=RandomizationCfg(),
      use_tn=cfg.use_tn,
      physics_timestep=cfg.physics_timestep,
      capture_reset=cfg.capture_reset,
      cpu_sim_workers=cfg.cpu_sim_workers,
    )
    replay = TemporalReplay(
      cfg.replay_steps,
      cfg.num_envs,
      robot.state_dim,
      robot.context_dim,
      robot.obs_dim,
      collection_device,
    )
  history = torch.empty(cfg.num_envs, 10, robot.obs_dim, device=collection_device)
  clips = torch.empty(cfg.num_envs, dtype=torch.long, device=collection_device)
  frames = torch.empty_like(clips)
  # Each step keeps the bin it belongs to and the ceiling of its own frame, so a
  # return observed late in a clip is not measured against what a frame a second
  # earlier could have earned.
  episode_traces: list[list[tuple[int, float, float]]] = [
    [] for _ in range(cfg.num_envs)
  ]
  pending_bins: list[int] = []
  pending_progress: list[float] = []
  length_device = lengths.to(collection_device)
  local_env_steps = int(rank_state.get("local_env_steps", 0)) if rank_state else 0
  # Phase attribution puts an explicit device barrier on each boundary, so the
  # reported seconds are comparable to one another but sum to slightly more
  # than an uninstrumented iteration, which overlaps the phases.
  phase_seconds: dict[str, float] = {}
  iteration_phases: dict[str, float] = {}
  profiling = False

  @contextmanager
  def phase(name: str) -> Generator[None]:
    if not profiling:
      yield
      return
    if mps:
      torch.mps.synchronize()
    elif cfg.device.startswith("cuda"):
      torch.cuda.synchronize(cfg.device)
    start = time.perf_counter()
    yield
    if mps:
      torch.mps.synchronize()
    elif cfg.device.startswith("cuda"):
      torch.cuda.synchronize(cfg.device)
    elapsed = time.perf_counter() - start
    iteration_phases[name] = iteration_phases.get(name, 0.0) + elapsed
    if iteration >= cfg.iterations - cfg.profile_iterations:
      phase_seconds[name] = phase_seconds.get(name, 0.0) + elapsed

  def sync_collection_actor() -> None:
    if mps:
      collection_actor.load_state_dict(actor.state_dict())

  def sample(batch_size: int, window: int, history: int) -> ReplayBatch:
    batch = replay.sample(batch_size, window, history)
    return batch.to(cfg.device) if mps else batch

  def reset(ids: torch.Tensor) -> None:
    c, f, _ = sampler.sample(len(ids))
    c, f = c.to(collection_device), f.to(collection_device)
    reference = collection_motions.sequence(
      c, f, torch.zeros(1, dtype=torch.long, device=collection_device)
    )[:, 0]
    simulator.reset(ids, reference, cfg.randomize)
    clips[ids], frames[ids] = c, f
    observation = physics_robot.observation(
      simulator.state()[ids], simulator.context[ids]
    )
    observation += torch.randn_like(observation) * physics_robot.noise_scale()
    history[ids] = observation[:, None].expand(-1, 10, -1)
    for index in ids.tolist():
      episode_traces[index].clear()

  @torch.no_grad()
  def collect() -> float:
    nonlocal local_env_steps
    state = simulator.state()
    context = simulator.context
    if cfg.modeled_action_delay:
      context = context.clone()
    references = collection_motions.sequence(
      clips, frames, torch.arange(1, 11, device=collection_device)
    )
    with group.timed("collector_actor"):
      action = collector.action(history, references)
    executed = action + cfg.action_noise * torch.randn_like(action)
    with group.timed("physics_step"):
      next_state = simulator.step(executed)
    with group.timed("collector_heads"):
      reward, terminated, observation = collector.heads(
        state,
        next_state,
        references[:, 0],
        action if cfg.clean_policy_actions else executed,
        simulator.context,
        history[:, -1, -22:] if cfg.clean_policy_actions else None,
      )
    boundary = terminated | (frames + 1 >= length_device[clips] - 1)
    replay.add(
      state,
      simulator.transition_state,
      executed,
      context,
      history[:, -1],
      terminated.float(),
      boundary | simulator.pushed,
      clips,
      frames,
    )
    with group.timed("collection_readback"):
      trace_bins = sampler.bin_ids(clips, frames).tolist()
      trace_ceilings = sampler.frame_ceiling(clips, frames).tolist()
      rewards = reward.cpu().tolist()
    for index, (bin_id, ceiling, value) in enumerate(
      zip(trace_bins, trace_ceilings, rewards, strict=True)
    ):
      episode_traces[index].append((bin_id, ceiling, value))
    frames.add_(1)
    local_env_steps += cfg.num_envs
    observation += torch.randn_like(observation) * physics_robot.noise_scale()
    history.copy_(torch.cat((history[:, 1:], observation[:, None]), 1))
    with group.timed("find_reset_ids"):
      ids = boundary.nonzero().flatten()
    if len(ids):
      for index in ids.tolist():
        future = 0.0
        for bin_id, ceiling, value in reversed(episode_traces[index]):
          future = value + 0.99 * future
          pending_bins.append(bin_id)
          pending_progress.append(future / ceiling)
      with group.timed("reset_worlds"):
        reset(ids)
    return float(reward.mean())

  def flush_curriculum() -> None:
    """Apply the buffered reset statistics, exactly once per iteration.

    Sharing the curriculum is a collective, and every rank must reach the same
    collectives in the same order. Episodes end whenever they end, and the
    replay top-up below collects a rank-dependent number of extra steps, so
    neither is a safe place to synchronize. One flush at a fixed point in the
    iteration is: batching a whole iteration's samples into one exponential
    step is what `update` already does within a step.
    """
    sampler.update(
      torch.tensor(pending_bins, dtype=torch.long),
      torch.tensor(pending_progress, dtype=torch.float32),
      reduce=group.sum_ if group.enabled else None,
    )
    pending_bins.clear()
    pending_progress.clear()
    sampler.advance_past()

  config = {
    k: str(v) if isinstance(v, Path) else v for k, v in asdict(requested).items()
  }
  identity = {
    "xml_sha256": robot.sha256,
    "asset_sha256": robot.asset_sha256,
    "joint_names": robot.joint_names,
    "control_dt": motions.dt,
    "reward_scaling": robot.reward_identity(),
    "reward_tuning": asdict(reward_tuning),
    "training_semantics": {
      "clean_policy_actions": cfg.clean_policy_actions,
      "action_lag": cfg.action_lag,
      "horizon": cfg.horizon,
      "model_warmup_iterations": cfg.model_warmup_iterations,
      "normalize_policy_objective": cfg.normalize_policy_objective,
      "normalize_model_objective": cfg.normalize_model_objective,
      "imagination_terminal_handling": TERMINAL_HANDLING,
      "imagination_min_survival": MIN_SURVIVAL,
    },
    "physics_timestep": physics_robot.model.opt.timestep,
    "xml_timestep": physics_robot.model.opt.timestep,
    "solver_tolerance": (
      float(simulator.warp_model.opt.tolerance.numpy().reshape(-1)[0])
      if cfg.device.startswith("cuda")
      else physics_robot.model.opt.tolerance
    ),
    "source_file_sha256": {
      path.name: hashlib.sha256(path.read_bytes()).hexdigest()
      for path in sorted(Path(__file__).parent.glob("*.py"))
    },
    "software_versions": {
      name: importlib.metadata.version(name)
      for name in ("torch", "mujoco", "mujoco-warp", "warp-lang", "numpy")
    }
    | {"python": platform.python_version()},
    "collection_device": collection_device,
    "device_name": (
      torch.cuda.get_device_name(cfg.device)
      if cfg.device.startswith("cuda")
      else platform.processor()
    ),
    "state_dim": robot.state_dim,
    "obs_dim": robot.obs_dim,
    "context_dim": robot.context_dim,
    "randomization": asdict(simulator.randomization),
    "mlp_precision_actual": precision,
    "network_architecture": {
      "activation": cfg.network_activation,
      "representation": cfg.dynamics_representation,
      "modeled_action_delay": cfg.modeled_action_delay,
      **(
        {"network_profile": cfg.network_profile}
        if cfg.network_profile != "legacy"
        else {}
      ),
    },
    "model_hidden": model_hidden,
    "actor_hidden": actor_hidden,
    "critic_hidden": critic_hidden,
    "available_motions": available_count,
    "selected_motions": len(paths),
    "motion_selection": cfg.motion_selection,
    "motion_seed": cfg.motion_seed,
    "motion_paths": [str(p.resolve()) for p in paths],
    "motion_lengths": lengths.tolist(),
    "kp": simulator.kp.tolist(),
    "kd": simulator.kd.tolist(),
    "use_tn": cfg.use_tn,
    "torque_speed_peak_nm": simulator.torque_peaks.tolist(),
    "torque_speed_knee_rad_s": simulator.torque_knees.tolist(),
    "torque_speed_cutoff_rad_s": simulator.torque_cutoffs.tolist(),
    "default_q": robot.default_q.tolist(),
    "calibration_semantics": (
      "episode_default=xml_default+offset; observation=q-episode_default"
    ),
    "world_size": group.world_size,
    "num_envs_per_rank": cfg.num_envs,
    "global_parallel_envs": requested.num_envs,
    "model_batch_size_per_rank": cfg.model_batch_size,
    "policy_batch_size_per_rank": cfg.policy_batch_size,
    "motion_storage": cfg.motion_storage,
  }
  start_iteration = 0
  with group.checked("learner and rank state restoration"):
    if resumed is not None:
      saved_architecture = resumed["identity"].get(
        "network_architecture",
        {
          "activation": "elu",
          "representation": "legacy",
          "modeled_action_delay": False,
        },
      )
      if saved_architecture != identity["network_architecture"]:
        raise ValueError("Checkpoint network representation changed; start a fresh run")
      for name in (
        "asset_sha256",
        "joint_names",
        "state_dim",
        "obs_dim",
        "context_dim",
        "model_hidden",
        "actor_hidden",
        "critic_hidden",
        "motion_paths",
        "motion_lengths",
        "control_dt",
        "kp",
        "kd",
        "default_q",
        "num_envs_per_rank",
        "model_batch_size_per_rank",
        "policy_batch_size_per_rank",
      ):
        if resumed["identity"][name] != identity[name]:
          raise ValueError(f"Checkpoint identity changed: {name}")
      if resumed["config"]["action_scale"] != cfg.action_scale:
        raise ValueError("Checkpoint action scale changed")
      if resumed["config"].get("use_tn", False) != cfg.use_tn:
        raise ValueError("Checkpoint torque-speed control changed")
      if resumed["config"].get("q1_reward_scaling", False) != cfg.q1_reward_scaling:
        raise ValueError("Checkpoint Q1 reward scaling changed")
      saved_identity = resumed["identity"]
      if saved_identity.get("randomization") != identity["randomization"]:
        raise ValueError("Checkpoint domain randomization changed; use a new run")
      saved_semantics = saved_identity.get(
        "training_semantics",
        {
          name: resumed["config"].get(name, default)
          for name, default in (
            ("clean_policy_actions", False),
            ("action_lag", True),
            ("normalize_policy_objective", False),
            ("normalize_model_objective", False),
          )
        },
      )
      saved_semantics = {
        "horizon": 32,
        "model_warmup_iterations": 0,
        **saved_semantics,
      }
      if saved_semantics != identity["training_semantics"]:
        raise ValueError(
          "Checkpoint training semantics changed; use a new warm-start run"
        )
      saved_timestep = saved_identity.get(
        "physics_timestep", saved_identity.get("xml_timestep")
      )
      if saved_timestep != identity["physics_timestep"]:
        raise ValueError("Checkpoint physics timestep changed")
      if (
        saved_identity.get("reward_scaling", identity["reward_scaling"])
        != (identity["reward_scaling"])
      ):
        raise ValueError("Checkpoint reward parameters changed")
      saved_tuning = saved_identity.get(
        "reward_tuning",
        asdict(RewardTuningCfg(action_scale=resumed["config"]["action_scale"])),
      )
      if saved_tuning != identity["reward_tuning"]:
        raise ValueError("Checkpoint reward tuning changed; use a new warm-start run")
      if "initialization" in saved_identity:
        identity["initialization"] = saved_identity["initialization"]
      if cfg.use_tn:
        for name in (
          "torque_speed_peak_nm",
          "torque_speed_knee_rad_s",
          "torque_speed_cutoff_rad_s",
        ):
          if resumed["identity"].get(name) != identity[name]:
            raise ValueError(f"Checkpoint motor curve changed: {name}")
      for name, module in (("model", model), ("actor", actor), ("critic", critic)):
        module.load_state_dict(resumed[name])
        optimizer = getattr(algorithm, f"{name}_optimizer")
        optimizer.load_state_dict(resumed[f"{name}_optimizer"])
      start_iteration = int(resumed["iteration"])
      if not 0 <= start_iteration < cfg.iterations:
        raise ValueError("--iterations must exceed the checkpoint iteration")
      if rank_state is not None:
        sampler.load_state_dict(rank_state["curriculum"])
        pending_bins.extend(rank_state["pending_bins"])
        pending_progress.extend(rank_state["pending_progress"])
        if cfg.resume_replay and rank_state.get("replay") is not None:
          replay.load_state_dict(rank_state["replay"])
    if warm_started is not None:
      restore_warm_start(
        warm_started,
        identity,
        model,
        actor,
        action_scale=cfg.action_scale,
        use_tn=cfg.use_tn,
      )
      assert cfg.warm_start is not None
      if cfg.warm_start_progress:
        start_iteration = int(warm_started["iteration"])
        global_steps = int(warm_started["global_env_steps"])
        if not 0 <= start_iteration < cfg.iterations or global_steps < 0:
          raise ValueError(
            "Warm-start progress must precede the cumulative iteration target"
          )
        if global_steps % group.world_size:
          raise ValueError("Warm-start environment steps must divide across ranks")
        local_env_steps = global_steps // group.world_size
      identity["initialization"] = warm_start_identity(
        cfg.warm_start, warm_started, preserve_progress=cfg.warm_start_progress
      )
  group.broadcast_parameters((model, actor, critic))
  with group.checked("RNG restoration and physics reset"):
    rank_seed = cfg.seed + 1_000_003 * group.rank
    torch.manual_seed(rank_seed)
    random.seed(rank_seed)
    np.random.seed(rank_seed % 2**32)
    if rank_state is not None:
      restore_rng(rank_state["rng"], cfg.device)
    sync_collection_actor()
    reset(torch.arange(cfg.num_envs, device=collection_device))
  with group.checked("run creation"):
    if group.main:
      cfg.output_dir.mkdir(parents=True, exist_ok=False)
      (cfg.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
      (cfg.output_dir / "identity.json").write_text(
        json.dumps(identity, indent=2) + "\n"
      )
  writer = None
  with group.checked("log initialization"):
    if group.main:
      writer = SummaryWriter(str(cfg.output_dir / "tensorboard"))
  start_time = time.monotonic()
  completed_iteration = start_iteration

  def checkpoint(include_replay: bool = False) -> Path:
    # A failed deferred update must never publish weights or optimizer state.
    group.validate_update()
    group.assert_parameters_synced((model, actor, critic))
    global_steps = int(group.sum_(torch.tensor(local_env_steps, device=cfg.device)))
    learner = {
      "config": config,
      "identity": identity,
      "iteration": completed_iteration,
      "global_env_steps": global_steps,
      "model": model.state_dict(),
      "actor": actor.state_dict(),
      "critic": critic.state_dict(),
      "robot": robot.state_dict(),
      "model_optimizer": algorithm.model_optimizer.state_dict(),
      "actor_optimizer": algorithm.actor_optimizer.state_dict(),
      "critic_optimizer": algorithm.critic_optimizer.state_dict(),
    }
    with group.checked("checkpoint state preparation"):
      local = {
        "rng": capture_rng(cfg.device),
        "curriculum": sampler.state_dict(),
        "local_env_steps": local_env_steps,
        "pending_bins": list(pending_bins),
        "pending_progress": list(pending_progress),
        "replay": (
          replay.state_dict() if cfg.checkpoint_replay or include_replay else None
        ),
      }
    path = save_checkpoint(
      cfg.output_dir / f"checkpoint_{completed_iteration}.pt", learner, local, group
    )
    return path

  if group.main:
    print(
      json.dumps(
        {
          "robot": "q1_v3",
          "selected_motions": len(paths),
          "ranks": group.world_size,
          "num_envs": cfg.num_envs * group.world_size,
          "num_envs_per_rank": cfg.num_envs,
          "obs_dim": robot.obs_dim,
          "small_networks": cfg.small_networks,
        }
      ),
      flush=True,
    )
  try:
    # Warmup and extra collection are bounded; dense failures give a useful error.
    for _ in range(40):
      if group.any(stop.requested):
        return checkpoint(include_replay=True)
      with group.checked("warmup collection"):
        collect()
    flush_curriculum()
    if group.main:
      print(
        json.dumps(
          {"warmup_collection_complete": 40, "compile_networks": cfg.compile_networks}
        ),
        flush=True,
      )
    for iteration in range(start_iteration, cfg.iterations):
      if group.any(stop.requested):
        checkpoint(include_replay=True)
        break
      # Time only the last iterations, so lazy kernel compilation and the
      # first replay fills stay out of the measurement.
      update_started = time.monotonic()
      actor_update = iteration >= cfg.model_warmup_iterations
      iteration_phases.clear()
      profiling = iteration >= cfg.iterations - cfg.profile_iterations or bool(
        cfg.profile_every and iteration > 0 and iteration % cfg.profile_every == 0
      )
      if profiler is not None and iteration >= cfg.iterations - cfg.profile_iterations:
        profiler.begin(local_env_steps)
      with phase("collect"):
        # 0.4 updates per collected control step: alternate 2 and 3 steps.
        with group.checked("collection"):
          sync_collection_actor()
          collection_reward = sum(collect() for _ in range(2 + iteration % 2)) / (
            2 + iteration % 2
          )
        flush_curriculum()
        model_batch = None
        policy_batch = None
        with group.checked("replay readiness"):
          for attempt in range(500):
            try:
              model_batch = sample(cfg.model_batch_size, window=8, history=1)
              if actor_update:
                policy_batch = sample(cfg.policy_batch_size, window=1, history=10)
              break
            except ValueError:
              if attempt == 499:
                raise RuntimeError(
                  "No valid replay windows; inspect reset poses and controls"
                ) from None
              collect()
      assert model_batch is not None
      algorithm.update_normalizers(model_batch)
      factor = 0.03 + 0.97 * 0.5 * (
        1 + math.cos(math.pi * min(iteration, 250000) / 250000)
      )
      for optimizer in (
        algorithm.model_optimizer,
        algorithm.actor_optimizer,
        algorithm.critic_optimizer,
      ):
        for param_group in optimizer.param_groups:
          param_group["lr"] = cfg.learning_rate * factor
      model_metrics = {}
      model_tensors: dict[str, torch.Tensor] = {}
      staged_batches = None
      if cfg.runtime_checks == "per_update":
        with group.checked("model replay sampling"):
          staged_batches = [model_batch] + [
            sample(cfg.model_batch_size, 8, 1) for _ in range(cfg.model_batches - 1)
          ]
      group.begin_update()
      with phase("model"):
        for update in range(cfg.model_batches):
          if staged_batches is not None:
            batch = staged_batches[update]
          else:
            with group.checked("model replay sampling"):
              batch = model_batch if update == 0 else sample(cfg.model_batch_size, 8, 1)
          model_tensors = algorithm.fit_model_tensors(batch)
        model_metrics = {name: float(value) for name, value in model_tensors.items()}
      with phase("policy"):
        metrics = model_metrics
        if actor_update:
          assert policy_batch is not None
          metrics = metrics | algorithm.improve_policy(
            policy_batch,
            cfg.critic_epochs,
            cfg.critic_minibatches,
            profile_mps=profiling and mps,
          )
        metrics["actor_update"] = float(actor_update)
      group.validate_update()
      metrics["collection_reward"] = collection_reward
      if iteration % 10 == 0:
        for name, norm in (
          ("model_input", model.input_norm),
          ("model_output", model.output_norm),
          ("observation", actor.obs_norm),
          ("goal", actor.goal_norm),
          ("critic", critic.input_norm),
        ):
          if isinstance(norm, RunningNorm):
            metrics[f"normalizer_samples/{name}"] = float(norm.count)
      if group.enabled:
        # Losses and rewards are per-rank samples of one global quantity, so the
        # log reports their mean rather than whatever rank 0 happened to see.
        names = sorted(metrics)
        metrics = dict(zip(names, group.mean([metrics[n] for n in names]), strict=True))
      metrics |= {
        "iteration": iteration + 1,
        "startup_update": iteration == start_iteration,
        "update_seconds": time.monotonic() - update_started,
        "elapsed_seconds": time.monotonic() - start_time,
        "global_env_steps": int(
          group.sum_(torch.tensor(local_env_steps, device=cfg.device))
        ),
      }
      metrics.update(
        {f"{name}_seconds": value for name, value in iteration_phases.items()}
      )
      if profiling and mps:
        metrics["mps_allocated_gib"] = torch.mps.current_allocated_memory() / 1024**3
        metrics["mps_driver_gib"] = torch.mps.driver_allocated_memory() / 1024**3
      if profiling and cfg.device.startswith("cuda"):
        metrics["torch_allocated_gib"] = (
          torch.cuda.memory_allocated(cfg.device) / 1024**3
        )
        metrics["torch_reserved_gib"] = torch.cuda.memory_reserved(cfg.device) / 1024**3
        metrics["cuda_free_gib"] = torch.cuda.mem_get_info(cfg.device)[0] / 1024**3
      completed_iteration = iteration + 1
      with group.checked("training log"):
        if group.main:
          assert writer is not None
          with (cfg.output_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
          for name, value in metrics.items():
            writer.add_scalar(name, value, iteration + 1)
          print(json.dumps(metrics), flush=True)
      if profiler is not None and profiler.active:
        profiler.finish(local_env_steps)
      stopping = group.any(stop.requested)
      if (
        (iteration + 1) % cfg.save_interval == 0
        or iteration + 1 == cfg.initial_save_iteration
        or (cfg.compile_networks and iteration == start_iteration)
        or iteration + 1 == cfg.iterations
        or stopping
      ):
        checkpoint(include_replay=stopping or iteration + 1 == cfg.iterations)
      if stopping:
        break
  finally:
    if writer is not None:
      writer.close()
  rank_summary = {}
  if profiler is not None and profiler.iterations:
    local_profile = profiler.result(group.rank)
    local_profile["phase_wall_ms_per_iteration"] = {
      name: 1000 * seconds / profiler.iterations
      for name, seconds in phase_seconds.items()
    }
    ranks = group.objects(local_profile)
    if group.main:
      rank_summary = summarize_ranks(ranks)
      (cfg.output_dir / "rank_performance.json").write_text(
        json.dumps({"summary": rank_summary, "ranks": ranks}, indent=2) + "\n"
      )
  if cfg.profile_iterations and phase_seconds and group.main:
    total = sum(phase_seconds.values())
    report = {
      "profiled_iterations": profiler.iterations
      if profiler is not None
      else cfg.profile_iterations,
      "device": cfg.device,
      "ranks": group.world_size,
      "num_envs": cfg.num_envs * group.world_size,
      "small_networks": cfg.small_networks,
      "seconds_per_iteration": {
        name: value
        / (profiler.iterations if profiler is not None else cfg.profile_iterations)
        for name, value in phase_seconds.items()
      },
      "share": {name: value / total for name, value in phase_seconds.items()},
    }
    if rank_summary:
      report["env_steps_per_second"] = rank_summary["env_steps_per_second"]
      report["min_sampled_device_free_gib"] = rank_summary[
        "min_sampled_device_free_gib"
      ]
    (cfg.output_dir / "throughput.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
  return cfg.output_dir / f"checkpoint_{completed_iteration}.pt"


def main() -> None:
  train(tyro.cli(TrainCfg))


if __name__ == "__main__":
  main()
