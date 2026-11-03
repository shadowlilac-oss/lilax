"""Profile artifact staging and upload."""

import trackio
import wandb

from lilax.operations.profiling import stage_xprof_logdir


def log_profile_artifacts(logdir, step, process_index, trackio_enabled, wandb_run):
  if process_index != 0 or not (trackio_enabled or wandb_run is not None):
    return
  with stage_xprof_logdir(logdir, step) as local_logdir:
    metadata = {"step": int(step), "logdir": str(logdir)}
    if trackio_enabled:
      artifact = trackio.Artifact(name=f"xprof-step-{step}", type="profile", metadata=metadata)
      artifact.add_dir(local_logdir)
      trackio.log_artifact(artifact)
    if wandb_run is not None:
      artifact = wandb.Artifact(name=f"xprof-step-{step}", type="profile", metadata=metadata)
      artifact.add_dir(local_logdir, policy="mutable")
      wandb_run.log_artifact(artifact)
