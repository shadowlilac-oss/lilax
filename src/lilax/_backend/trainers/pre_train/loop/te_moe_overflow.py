"""Windowed host check that aborts training when TE MoE receive capacity overflowed."""

import jax
import numpy as np

from lilax._backend.utils import max_logging


def track_te_moe_overflow(config, step, metrics, te_moe_overflow_window, prof):
  te_moe_overflow_window.append((
      int(step), metrics["scalar"]["learning/te_moe_capacity_overflow"],
      metrics["scalar"]["learning/te_moe_max_total_recv_tokens"],
      metrics["scalar"]["learning/te_moe_recv_capacity_per_rank"],
  ))
  check_overflow = len(te_moe_overflow_window) == config.te_ep_overflow_check_every_n_steps or step == config.steps - 1
  if check_overflow:
    checked_window = jax.device_get(tuple(te_moe_overflow_window))
    overflowing_steps = [entry for entry in checked_window if bool(np.asarray(entry[1]))]
    te_moe_overflow_window.clear()
  else:
    overflowing_steps = []
  if overflowing_steps:
    overflow_step = overflowing_steps[0][0]
    observed = max(int(np.asarray(entry[2])) for entry in overflowing_steps)
    capacity = min(int(np.asarray(entry[3])) for entry in overflowing_steps)
    prof.deactivate()
    message = (
        "TE MoE receive capacity overflow at training step "
        f"{overflow_step} (detected at the {config.te_ep_overflow_check_every_n_steps}-step "
        f"check ending at step {step}): observed padded receive demand {observed} exceeds "
        f"recv_capacity_per_rank {capacity} (ragged_buffer_factor={config.ragged_buffer_factor}). "
        "The optimizer update was skipped; increase ragged_buffer_factor, "
        "or set it to -1 to reserve worst-case dropless capacity."
    )
    max_logging.error(message)
    raise RuntimeError(message)
