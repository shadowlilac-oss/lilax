"""Parse the public Trainer checkpoint selection contract."""

from dataclasses import dataclass

from etils import epath


@dataclass(frozen=True)
class CheckpointRequest:
  resume: bool
  step: int | None = None
  directory: str | None = None


def checkpoint_request(resume_from_checkpoint, checkpoint_step):
  if checkpoint_step is not None:
    if isinstance(checkpoint_step, bool) or not isinstance(checkpoint_step, int) or checkpoint_step < 0:
      raise ValueError("checkpoint_step must be a nonnegative integer.")
    if resume_from_checkpoint is not None and resume_from_checkpoint is not False:
      raise ValueError("checkpoint_step and resume_from_checkpoint are mutually exclusive.")
    return CheckpointRequest(True, checkpoint_step)
  if resume_from_checkpoint is None or resume_from_checkpoint is False:
    return CheckpointRequest(False)
  if resume_from_checkpoint is True:
    return CheckpointRequest(True)
  if isinstance(resume_from_checkpoint, str) and resume_from_checkpoint:
    path = epath.Path(resume_from_checkpoint)
    if path.name.isdigit():
      return CheckpointRequest(True, int(path.name), str(path.parent))
    return CheckpointRequest(True, directory=str(path))
  raise ValueError("resume_from_checkpoint must be a path, True, False or None.")
