"""Continuous-time Flow Match diffusion noise scheduler."""

from functools import partial
from typing import Optional, Tuple, Union

import flax
import jax
import jax.numpy as jnp

from .scheduling_utils import FlaxSchedulerMixin, FlaxSchedulerOutput, register_to_config


@flax.struct.dataclass
class FlowMatchSchedulerState:
  sigmas: jnp.ndarray
  timesteps: jnp.ndarray
  linear_timesteps_weights: Optional[jnp.ndarray]
  training: bool
  num_inference_steps: int

  @classmethod
  def create(cls):
    return cls(sigmas=None, timesteps=None, linear_timesteps_weights=None, training=False, num_inference_steps=0)


@flax.struct.dataclass(frozen=False)
class FlaxFlowMatchSchedulerOutput(FlaxSchedulerOutput):
  state: FlowMatchSchedulerState


class FlaxFlowMatchScheduler(FlaxSchedulerMixin):
  dtype: jnp.dtype

  @property
  def has_state(self) -> bool:
    return True

  @register_to_config
  def __init__(self, num_train_timesteps: int = 1000, shift: float = 3.0, sigma_max: float = 1.0,
               sigma_min: float = 0.003 / 1.002, inverse_timesteps: bool = False, extra_one_step: bool = False,
               reverse_sigmas: bool = False, use_dynamic_shifting: bool = False, time_shift_type: str = "linear",
               dtype: jnp.dtype = jnp.float32):
    self.dtype = dtype

  def create_state(self) -> FlowMatchSchedulerState:
    return FlowMatchSchedulerState.create()

  def set_timesteps(self, state: FlowMatchSchedulerState, num_inference_steps: int = 100, shape: Tuple = None,
                    denoising_strength: float = 1.0, training: bool = False,
                    shift: Optional[float] = None) -> FlowMatchSchedulerState:
    current_shift = shift if shift is not None else self.config.shift
    sigma_start = self.config.sigma_min + (self.config.sigma_max - self.config.sigma_min) * denoising_strength
    if self.config.extra_one_step:
      sigmas = jnp.linspace(sigma_start, self.config.sigma_min, num_inference_steps + 1, dtype=self.dtype)[:-1]
    else:
      sigmas = jnp.linspace(sigma_start, self.config.sigma_min, num_inference_steps, dtype=self.dtype)
    if self.config.inverse_timesteps:
      sigmas = jnp.flip(sigmas, axis=0)
    if getattr(self.config, "use_dynamic_shifting", False):
      if getattr(self.config, "time_shift_type", "exponential") == "exponential":
        sigmas = jnp.exp(current_shift) / (jnp.exp(current_shift) + (1 / jnp.clip(sigmas, 1e-7, 1.0) - 1))
      else:
        sigmas = current_shift * sigmas / (1 + (current_shift - 1) * sigmas)
    else:
      sigmas = current_shift * sigmas / (1 + (current_shift - 1) * sigmas)
    if self.config.reverse_sigmas:
      sigmas = 1 - sigmas
    timesteps = sigmas * self.config.num_train_timesteps
    linear_timesteps_weights = None
    if training:
      linear_timesteps_weights = self._calculate_training_weights(timesteps, num_inference_steps)
    return state.replace(sigmas=sigmas, timesteps=timesteps, linear_timesteps_weights=linear_timesteps_weights,
                         training=training, num_inference_steps=num_inference_steps)

  def _calculate_training_weights(self, timesteps: jnp.ndarray, num_inference_steps: int) -> jnp.ndarray:
    x = timesteps
    y = jnp.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
    y_shifted = y - jnp.min(y)
    bsmntw_weighing = y_shifted * (num_inference_steps / jnp.sum(y_shifted))
    return bsmntw_weighing

  def sample_timesteps(self, timestep_rng, batch_size):
    t = jax.random.uniform(timestep_rng, (batch_size,))
    t_shifted = (t * self.config.shift) / (1 + (self.config.shift - 1) * t)
    return t_shifted * self.config.num_train_timesteps

  def apply_flow_match(self, noise: jnp.ndarray, batch_images: jnp.ndarray,
                       timesteps: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    t = timesteps.astype(jnp.float32) / self.config.num_train_timesteps
    broadcast_shape = (-1,) + (1,) * (batch_images.ndim - 1)
    t = t.reshape(broadcast_shape)
    sigma = (1 - t) * self.config.sigma_min + t * self.config.sigma_max
    noisy_latents = (1 - sigma) * batch_images + sigma * noise
    target = noise - batch_images
    schedule = self.set_timesteps(self.create_state(), num_inference_steps=self.config.num_train_timesteps, training=True)
    training_weights = self.training_weight(schedule, jnp.reshape(timesteps, (-1,)))
    return noisy_latents, target, training_weights

  def _find_timestep_id(self, state: FlowMatchSchedulerState, timestep: jnp.ndarray) -> jnp.ndarray:
    timestep = jnp.asarray(timestep, dtype=state.timesteps.dtype)
    if timestep.ndim == 0:
      return jnp.argmin(jnp.abs(state.timesteps - timestep))
    diffs = jnp.abs(state.timesteps[None, :] - timestep[:, None])
    return jnp.argmin(diffs, axis=1)

  @partial(jax.jit, static_argnums=(0, 5, 6))
  def step(self, state: FlowMatchSchedulerState, model_output: jnp.ndarray, timestep: jnp.ndarray,
           sample: jnp.ndarray, to_final: bool = False, return_dict: bool = True,
           step_index: Optional[int] = None) -> Union[FlaxFlowMatchSchedulerOutput, Tuple]:
    if step_index is not None:
      timestep_id = step_index
    else:
      timestep_id = self._find_timestep_id(state, timestep)
    sigma = state.sigmas[timestep_id]

    def get_next_sigma():
      return state.sigmas[timestep_id + 1]

    def get_final_sigma():
      val = 1.0 if (self.config.inverse_timesteps or self.config.reverse_sigmas) else 0.0
      return jnp.full_like(state.sigmas[timestep_id + 1], val, dtype=sigma.dtype)

    is_final_step = to_final or jnp.all(timestep_id + 1 >= state.timesteps.shape[0])
    sigma_next = jax.lax.cond(is_final_step, get_final_sigma, get_next_sigma)
    if jnp.ndim(timestep) != 0:
      broadcast_shape = (-1,) + (1,) * (sample.ndim - 1)
      sigma = sigma.reshape(broadcast_shape)
      sigma_next = sigma_next.reshape(broadcast_shape)
    prev_sample = sample + model_output * (sigma_next - sigma)
    if not return_dict:
      return (prev_sample, state)
    return FlaxFlowMatchSchedulerOutput(prev_sample=prev_sample, state=state)

  def return_to_timestep(self, state: FlowMatchSchedulerState, timestep: jnp.ndarray, sample: jnp.ndarray,
                         sample_stablized: jnp.ndarray) -> jnp.ndarray:
    timestep_id = self._find_timestep_id(state, timestep)
    sigma = state.sigmas[timestep_id]
    if jnp.ndim(timestep) != 0:
      sigma = sigma.reshape((-1,) + (1,) * (sample.ndim - 1))
    return (sample - sample_stablized) / sigma

  def add_noise(self, state: FlowMatchSchedulerState, original_samples: jnp.ndarray, noise: jnp.ndarray,
                timesteps: jnp.ndarray) -> jnp.ndarray:
    if state.sigmas is None or state.timesteps is None:
      raise ValueError("Scheduler's `sigmas` and `timesteps` are not set. Please call `set_timesteps` before `add_noise`.")
    timestep_ids = self._find_timestep_id(state, timesteps)
    sigmas = state.sigmas[timestep_ids]
    broadcast_shape = (-1,) + (1,) * (original_samples.ndim - 1)
    sigmas = sigmas.reshape(broadcast_shape)
    return (1 - sigmas) * original_samples + sigmas * noise

  def training_target(self, sample: jnp.ndarray, noise: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
    return noise - sample

  def training_weight(self, state: FlowMatchSchedulerState, timestep: jnp.ndarray) -> jnp.ndarray:
    timestep_ids = self._find_timestep_id(state, timestep)
    return state.linear_timesteps_weights[timestep_ids]

  def __len__(self) -> int:
    return self.config.num_train_timesteps
