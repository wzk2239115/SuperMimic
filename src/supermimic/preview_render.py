"""Render a short checkpoint rollout in a separate MuJoCo process."""

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

from supermimic.play import PlayCfg, play
from supermimic.record import RecordCfg, diagnostic_summary, render_video


def world_links(xml: Path, actual: np.ndarray, reference: np.ndarray) -> dict:
  model = mujoco.MjModel.from_xml_path(str(xml))
  data = mujoco.MjData(model)

  def positions(track: np.ndarray) -> list:
    result = []
    for pose in track:
      data.qpos[:] = pose
      mujoco.mj_kinematics(model, data)
      result.append(data.xpos.round(5).tolist())
    return result

  return {
    "fps": 50,
    "parents": model.body_parentid.tolist(),
    "actual": positions(actual),
    "reference": positions(reference),
  }


def render(
  checkpoint: Path, output: Path, motion: int, seconds: int, mode: str
) -> None:
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
  if mode == "links":
    (output / "links.json").write_text(
      json.dumps(
        world_links(xml, actual, reference), separators=(",", ":"), allow_nan=False
      )
    )
  else:
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
  _, checkpoint, output, motion, seconds, mode = sys.argv
  render(Path(checkpoint), Path(output), int(motion), int(seconds), mode)
