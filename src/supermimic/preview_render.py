"""Render a short checkpoint rollout in a separate MuJoCo process."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from supermimic.play import PlayCfg, play
from supermimic.record import RecordCfg, diagnostic_summary, render_video


def render(checkpoint: Path, output: Path, motion: int, seconds: int) -> None:
  rollout = output / "rollout.npz"
  result = play(
    PlayCfg(
      checkpoint=checkpoint,
      motion_index=motion,
      output=rollout,
      viewer="off",
      stop_on_failure=False,
      max_steps=seconds * 50,
    ),
    cpu_threads=1,
  )
  with np.load(rollout, allow_pickle=False) as data:
    actual = data["states"][:, :29]
    reference = np.concatenate((actual[:1], data["reference_qpos"]))
    summary = diagnostic_summary(
      actual, reference, data["step_rewards"], data["step_failures"]
    )

  learner = torch.load(checkpoint, map_location="cpu", weights_only=True)
  xml = Path(learner["config"]["robot_xml"])
  render_video(
    RecordCfg(checkpoint, output, output, width=400, height=260),
    xml,
    actual,
    reference,
    summary["first_failure_frame"],
    reference_first=True,
  )
  (output / "effect.json").write_text(
    json.dumps(summary | {"motion": result["motion"]})
  )


if __name__ == "__main__":
  _, checkpoint, output, motion, seconds = sys.argv
  render(Path(checkpoint), Path(output), int(motion), int(seconds))
