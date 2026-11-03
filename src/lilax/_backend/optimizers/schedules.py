"""Learning rate schedules used by MaxText training."""

import jax.numpy as jnp
import optax

from lilax._backend.configs import types


def create_learning_rate_schedule(config):
  """Creates a learning rate schedule with warmup and decay."""

  def make_cos_schedule(init_lr, final_lr, len_steps):
    def schedule(step):
      pct = step / (len_steps - 1) if len_steps > 1 else 1.0
      a = 0.5 * (jnp.cos(jnp.pi * pct) + 1)
      lr = init_lr * a + final_lr * (1 - a)
      return lr

    return schedule

  lr = config.learning_rate
  final_lr = lr * config.learning_rate_final_fraction
  warmup_steps = int(config.learning_rate_schedule_steps * config.warmup_steps_fraction)
  constant_zero_steps = config.steps - config.learning_rate_schedule_steps

  pieces = []
  boundaries = []

  if warmup_steps > 0:
    warmup_schedule = optax.linear_schedule(init_value=0.0, end_value=lr, transition_steps=warmup_steps)
    pieces.append(warmup_schedule)
    boundaries.append(warmup_steps)

  if config.lr_schedule_type == types.LearningRateScheduleType.COSINE:
    cos_steps = config.learning_rate_schedule_steps - warmup_steps
    if cos_steps > 0:
      cos_schedule = make_cos_schedule(lr, final_lr, cos_steps)
      pieces.append(cos_schedule)
      boundaries.append(warmup_steps + cos_steps)
  else:
    decay_steps = int(config.learning_rate_schedule_steps * config.wsd_decay_steps_fraction)
    stable_steps = config.learning_rate_schedule_steps - warmup_steps - decay_steps

    if stable_steps > 0:
      stable_schedule = optax.constant_schedule(lr)
      pieces.append(stable_schedule)
      boundaries.append(warmup_steps + stable_steps)
    if decay_steps > 0:
      if config.wsd_decay_style == types.WsdDecayStyle.LINEAR:
        decay_schedule = optax.linear_schedule(init_value=lr, end_value=final_lr, transition_steps=decay_steps - 1)
      else:
        decay_schedule = make_cos_schedule(lr, final_lr, decay_steps)
      pieces.append(decay_schedule)
      boundaries.append(warmup_steps + stable_steps + decay_steps)

  if constant_zero_steps > 0:
    constant_schedule = optax.constant_schedule(0.0)
    pieces.append(constant_schedule)
    boundaries.append(config.learning_rate_schedule_steps)

  return optax.join_schedules(pieces, boundaries)
