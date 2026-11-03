"""Diffusion noise scheduler mixins and output dataclasses."""

from dataclasses import dataclass
import functools
import inspect
import math
from typing import Tuple

from diffusers.utils import BaseOutput
import flax
import jax.numpy as jnp

SCHEDULER_CONFIG_NAME = "scheduler_config.json"


@dataclass
class FlaxSchedulerOutput(BaseOutput):
  prev_sample: jnp.ndarray


class SchedulerConfig(dict):
  def __getattr__(self, name):
    try:
      return self[name]
    except KeyError as e:
      raise AttributeError(name) from e


def register_to_config(init):
  signature = inspect.signature(init)

  @functools.wraps(init)
  def wrapper(self, *args, **kwargs):
    bound = signature.bind(self, *args, **kwargs)
    bound.apply_defaults()
    ignore = {"self", *getattr(self, "ignore_for_config", ())}
    self.config = SchedulerConfig({k: v for k, v in bound.arguments.items() if k not in ignore})
    init(self, *args, **kwargs)

  return wrapper


class FlaxSchedulerMixin:
  ignore_for_config = ["dtype"]

  @classmethod
  def from_config(cls, config: dict, **kwargs):
    params = set(inspect.signature(cls.__init__).parameters) - {"self"}
    scheduler = cls(**{k: v for k, v in {**config, **kwargs}.items() if k in params})
    state = scheduler.create_state() if getattr(scheduler, "has_state", False) else None
    return scheduler, state


def broadcast_to_shape_from_left(x: jnp.ndarray, shape: Tuple[int]) -> jnp.ndarray:
  assert len(shape) >= x.ndim
  return jnp.broadcast_to(x.reshape(x.shape + (1,) * (len(shape) - x.ndim)), shape)


def betas_for_alpha_bar(num_diffusion_timesteps: int, max_beta=0.999, dtype=jnp.float32) -> jnp.ndarray:
  def alpha_bar(time_step):
    return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

  betas = []
  for i in range(num_diffusion_timesteps):
    t1 = i / num_diffusion_timesteps
    t2 = (i + 1) / num_diffusion_timesteps
    betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
  return jnp.array(betas, dtype=dtype)


def rescale_betas_zero_snr(betas):
  alphas = 1.0 - betas
  alphas_cumprod = jnp.cumprod(alphas, axis=0)
  alphas_bar_sqrt = jnp.sqrt(alphas_cumprod)
  alphas_bar_sqrt_0 = jnp.copy(alphas_bar_sqrt[0])
  alphas_bar_sqrt_T = jnp.copy(alphas_bar_sqrt[-1])
  alphas_bar_sqrt -= alphas_bar_sqrt_T
  alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)
  alphas_bar = alphas_bar_sqrt**2
  alphas = alphas_bar[1:] / alphas_bar[:-1]
  alphas = jnp.concatenate([alphas_bar[0:1], alphas])
  betas = 1 - alphas
  return betas


@flax.struct.dataclass
class CommonSchedulerState:
  alphas: jnp.ndarray
  betas: jnp.ndarray
  alphas_cumprod: jnp.ndarray

  @classmethod
  def create(cls, scheduler):
    config = scheduler.config
    if config.trained_betas is not None:
      betas = jnp.asarray(config.trained_betas, dtype=scheduler.dtype)
    elif config.beta_schedule == "linear":
      betas = jnp.linspace(config.beta_start, config.beta_end, config.num_train_timesteps, dtype=scheduler.dtype)
    elif config.beta_schedule == "scaled_linear":
      betas = jnp.linspace(config.beta_start**0.5, config.beta_end**0.5, config.num_train_timesteps, dtype=scheduler.dtype) ** 2
    elif config.beta_schedule == "squaredcos_cap_v2":
      betas = betas_for_alpha_bar(config.num_train_timesteps, dtype=scheduler.dtype)
    else:
      raise NotImplementedError(
          f"beta_schedule {config.beta_schedule} is not implemented for scheduler {scheduler.__class__.__name__}"
      )
    if config.get("rescale_zero_terminal_snr", False):
      betas = rescale_betas_zero_snr(betas)
    alphas = 1.0 - betas
    alphas_cumprod = jnp.cumprod(alphas, axis=0)
    return cls(alphas=alphas, betas=betas, alphas_cumprod=alphas_cumprod)


def get_sqrt_alpha_prod(state: CommonSchedulerState, original_samples: jnp.ndarray, noise: jnp.ndarray,
                        timesteps: jnp.ndarray):
  alphas_cumprod = state.alphas_cumprod
  sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
  sqrt_alpha_prod = sqrt_alpha_prod.flatten()
  sqrt_alpha_prod = broadcast_to_shape_from_left(sqrt_alpha_prod, original_samples.shape)
  sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
  sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
  sqrt_one_minus_alpha_prod = broadcast_to_shape_from_left(sqrt_one_minus_alpha_prod, original_samples.shape)
  return sqrt_alpha_prod, sqrt_one_minus_alpha_prod


def add_noise_common(state: CommonSchedulerState, original_samples: jnp.ndarray, noise: jnp.ndarray,
                     timesteps: jnp.ndarray):
  sqrt_alpha_prod, sqrt_one_minus_alpha_prod = get_sqrt_alpha_prod(state, original_samples, noise, timesteps)
  return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise


def get_velocity_common(state: CommonSchedulerState, sample: jnp.ndarray, noise: jnp.ndarray, timesteps: jnp.ndarray):
  sqrt_alpha_prod, sqrt_one_minus_alpha_prod = get_sqrt_alpha_prod(state, sample, noise, timesteps)
  return sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
