"""Whole-rollout math and CPU fallback do not require a CUDA context."""

from dataclasses import replace
from typing import Literal

import pytest
import torch
from test_reward_training import environment as environment
from test_reward_training import small_learner
from test_runtime_training import assert_state_equal, make_batch

from supermimic.algorithm import gae_returns
from supermimic.data import Q1Motions
from supermimic.distributed import Distributed
from supermimic.imagination_graph import ImaginationGraph
from supermimic.reward_tuning import RewardTuningCfg
from supermimic.robot import Q1Robot


@pytest.mark.parametrize("horizon", [4, 32])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("new_semantics", [False, True])
@pytest.mark.parametrize("mode", ["per_minibatch", "per_update"])
def test_whole_rollout_math_matches_original_complete_updates_exactly(
  environment: tuple[Q1Robot, Q1Motions],
  horizon: int,
  cached: bool,
  new_semantics: bool,
  mode: Literal["per_minibatch", "per_update"],
):
  robot, motions = environment
  torch.manual_seed(872)
  batch = make_batch(robot, motions, 1)
  # Include an imminent reference ending and different clean/executed history.
  batch.frames[0] = int(motions.lengths()[0]) - 2
  batch.states[:, :, 57:] = 0.3
  batch.history[:, -1, -22:] = -0.2
  learners = []
  for graph_path in (False, True):
    torch.manual_seed(913)
    learner = small_learner(
      robot,
      motions,
      RewardTuningCfg(arm_joint_weight=0.05 if new_semantics else 0.0),
    )
    learner.horizon = horizon
    learner.cache_goals = cached
    learner.cache_reference_heads = cached
    learner.clean_policy_actions = new_semantics
    learner.normalize_policy_objective = new_semantics
    learner.normalize_model_objective = new_semantics
    learner.distributed = Distributed.detached(validation_mode=mode)
    if graph_path:
      learner.imagination_graph = ImaginationGraph(learner, gae_returns)
    learners.append(learner)
  for update in range(2):
    metrics = []
    for learner in learners:
      torch.manual_seed(653 + update)
      learner.distributed.begin_update()
      model_metrics = learner.fit_model_tensors(batch)
      policy_metrics = (
        learner.improve_policy(batch, 2, 2)
        if learner.imagination_graph is None
        else learner._improve_policy_captured(batch, 2, 2)
      )
      learner.distributed.validate_update()
      metrics.append((model_metrics, policy_metrics))
    assert_state_equal(metrics[0], metrics[1])
    for name in ("model", "actor", "critic"):
      assert_state_equal(
        getattr(learners[0], name).state_dict(),
        getattr(learners[1], name).state_dict(),
      )
      assert_state_equal(
        getattr(learners[0], f"{name}_optimizer").state_dict(),
        getattr(learners[1], f"{name}_optimizer").state_dict(),
      )
      for left, right in zip(
        getattr(learners[0], name).parameters(),
        getattr(learners[1], name).parameters(),
        strict=True,
      ):
        assert_state_equal(left.grad, right.grad)


def test_cpu_opt_in_falls_back_without_capture(
  environment: tuple[Q1Robot, Q1Motions],
  monkeypatch: pytest.MonkeyPatch,
):
  robot, motions = environment
  learner = small_learner(robot, motions, RewardTuningCfg())
  graph = ImaginationGraph(learner, gae_returns)
  learner.imagination_graph = graph

  def unexpected_capture(*args, **kwargs):
    raise AssertionError("CPU training must not initialize CUDA capture")

  monkeypatch.setattr(graph, "run", unexpected_capture)
  result = learner.improve_policy(make_batch(robot, motions, 1), 1, 1)
  assert result["actor_grad_norm"] > 0
  assert graph.graph is None
  assert not torch.cuda.is_initialized()


def test_tensor_rollout_keeps_analytic_input_gradients_with_frozen_weights(
  environment: tuple[Q1Robot, Q1Motions],
):
  robot, motions = environment
  learner = small_learner(robot, motions, RewardTuningCfg(arm_joint_weight=0.05))
  learner.clean_policy_actions = True
  graph = ImaginationGraph(learner, gae_returns)
  inputs = graph.prepare(make_batch(robot, motions, 1))
  inputs = replace(
    inputs,
    state=inputs.state.clone().requires_grad_(),
    history=inputs.history.clone().requires_grad_(),
  )
  output = graph.forward_backward(inputs)
  for value in (inputs.state.grad, inputs.history.grad):
    assert value is not None and torch.isfinite(value).all() and value.abs().sum() > 0
  assert all(parameter.grad is None for parameter in learner.model.parameters())
  assert all(parameter.grad is None for parameter in learner.critic.parameters())
  assert all(parameter.requires_grad for parameter in learner.model.parameters())
  assert all(parameter.requires_grad for parameter in learner.critic.parameters())
  assert not output.targets.requires_grad
  assert not output.states.requires_grad and not output.goals.requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_eager_cuda_graph_captures_and_reuses_actor_gradients(
  environment: tuple[Q1Robot, Q1Motions],
):
  robot, motions = environment
  batch = make_batch(robot, motions, 1).to("cuda:0")
  robot.to("cuda:0")
  learner = small_learner(robot, motions, RewardTuningCfg(arm_joint_weight=0.05))
  for module in (learner.model, learner.actor, learner.critic):
    module.to("cuda:0")
    module.mlp_precision = "bf16" if torch.cuda.is_bf16_supported() else "fp32"
  learner.imagination_graph = ImaginationGraph(learner, gae_returns)

  for _ in range(2):
    results = learner.improve_policy(batch, 1, 1)
    assert torch.isfinite(torch.tensor(list(results.values()))).all()
    assert results["actor_grad_norm"] > 0
  assert learner.compiled_backend is None
  assert learner.imagination_graph.graph is not None
