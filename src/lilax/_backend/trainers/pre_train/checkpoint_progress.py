"""Validate training progress restored from checkpoints."""


def validate_completed_steps(completed_steps: int, config_steps: int):
  """Raise if training has already completed up to config_steps."""
  if completed_steps >= config_steps:
    raise RuntimeError(
        f"Requested training up to step {config_steps}, but a checkpoint already exists at step {completed_steps - 1} "
        f"(which means {completed_steps} steps have been completed). "
        f"Did you mean to continue training past step {completed_steps} (you should set steps > {completed_steps}) "
        f"or to not load the checkpoint (use enable_checkpointing=False?)"
    )
