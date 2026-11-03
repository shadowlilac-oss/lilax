"""Minimal M3 training loop for collecting profiles."""

import functools
import math
import time

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
import optax

from lilax._backend.common import profiler
from lilax._backend.m3 import models as m3_models
from lilax._backend.m3.core.sharding import create_mesh
from lilax._backend.m3.infra import scan as m3_scan
from lilax._backend.optimizers import optimizers
from lilax._backend.utils import max_logging

_BATCH_AXES = ("data", "fsdp")


def make_optimizer(config) -> optax.GradientTransformation:
  tx = optimizers.get_optimizer(config, config.learning_rate)
  if config.gradient_clipping_threshold > 0:
    return optax.chain(optax.clip_by_global_norm(config.gradient_clipping_threshold), tx)
  return tx


@functools.lru_cache(maxsize=32)
def _synthetic_batch_fn(batch_size: int, seq_len: int, vocab_size: int, mesh: Mesh):
  def body(key: jax.Array) -> dict[str, jax.Array]:
    tokens = jax.random.randint(key, (batch_size, seq_len + 1), 0, vocab_size, dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch_size, seq_len))
    return {"inputs": tokens[:, :-1], "targets": tokens[:, 1:], "positions": positions}

  return jax.jit(body, out_shardings=NamedSharding(mesh, P(_BATCH_AXES)))


def make_synthetic_batch(config, mesh: Mesh, key: jax.Array) -> dict[str, jax.Array]:
  fn = _synthetic_batch_fn(config.global_batch_size_to_train_on, config.max_target_length, config.vocab_size, mesh)
  return fn(key)


def loss_fn(model: nnx.Module, batch: dict[str, jax.Array]) -> jax.Array:
  logits = model(batch["inputs"], batch["positions"])
  losses = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), batch["targets"])
  return jnp.mean(losses)


@nnx.jit(donate_argnums=(0, 1))
def train_step(model: nnx.Module, optimizer: nnx.Optimizer, batch: dict[str, jax.Array]) -> jax.Array:
  loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
  optimizer.update(model, grads)
  return loss


def _log_step(step: int, loss: jax.Array, last_done: float, tokens_per_step: int) -> tuple[float, float]:
  loss = float(loss)
  done = time.perf_counter()
  step_time = done - last_done
  max_logging.log(f"step {step}: loss {loss:.4f}, {step_time:.3f}s, {tokens_per_step / step_time:,.0f} tokens/s")
  return loss, done


def train_loop(config) -> float:
  """Build and profile an M3 model, returning the last loss."""
  mesh = create_mesh()
  max_logging.log(f"m3 loop: mesh {dict(mesh.shape)} over {len(jax.devices())} device(s)")
  if config.gradient_accumulation_steps > 1:
    raise ValueError(
        "m3 loop does not accumulate gradients yet; "
        f"got gradient_accumulation_steps={config.gradient_accumulation_steps}."
    )
  batch_size = config.global_batch_size_to_train_on
  batch_shards = math.prod(mesh.shape[axis] for axis in _BATCH_AXES)
  if batch_size == 0 or batch_size % batch_shards:
    raise ValueError(
        f"per_device_batch_size={config.per_device_batch_size} gives a global batch of {batch_size}, which does not "
        f"split over the {batch_shards} shards of mesh axes {_BATCH_AXES}."
    )

  with jax.set_mesh(mesh):
    model = m3_models.create_model(config, mesh, rngs=nnx.Rngs(config.init_weights_seed))
    model = m3_scan.maybe_apply_scan(model, config)
    optimizer = nnx.Optimizer(model, make_optimizer(config), wrt=nnx.Param)
    num_params = sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
    tokens_per_step = config.global_batch_size_to_train_on * config.max_target_length
    max_logging.log(
        f"m3 loop: {config.model_name}, {num_params / 1e9:.3f}B params, "
        f"batch {config.global_batch_size_to_train_on} x {config.max_target_length} tokens"
    )
    prof = profiler.Profiler(config, offset_step=0)
    data_key = jax.random.key(config.init_weights_seed)
    loss = float("nan")
    step_fn = nnx.cached_partial(train_step, model, optimizer)
    pending = None
    last_done = time.perf_counter()
    for step in range(config.steps):
      batch = make_synthetic_batch(config, mesh, jax.random.fold_in(data_key, step))
      prof.maybe_activate_profiler(step, pending)
      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        step_loss = step_fn(batch)
      if pending is not None:
        loss, last_done = _log_step(*pending, last_done, tokens_per_step)
      pending = (step, step_loss)
      prof.maybe_deactivate_profiler(step, pending)
    if pending is not None:
      loss, last_done = _log_step(*pending, last_done, tokens_per_step)
    if config.profiler:
      max_logging.log(f"m3 loop: profile written under {config.tensorboard_dir}")
    return loss
