"""Run the existing M3 profiling recipe from a configured Trainer."""

from lilax._backend.m3.train import loop


def profile_m3(trainer) -> float:
  """Profile synthetic training with the existing M3 model-family restrictions."""
  return loop.train_loop(trainer.config)
