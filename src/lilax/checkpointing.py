"""Checkpoint operations backed by MaxText's checkpoint managers."""

import datetime
import os
import time

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from lilax._backend.checkpoint_conversion.compare_hf_ckpt import compare_hf_ckpt
from lilax._backend.checkpoint_conversion.inspect_checkpoint import inspect_hf, inspect_maxtext, inspect_orbax
from lilax._backend.checkpoint_conversion.load_and_quantize_checkpoint import (
    load_and_quantize_checkpoint as _load_and_quantize_checkpoint,
)
from lilax._backend.checkpoint_conversion.reshard_checkpoint import reshard_checkpoint
from lilax._backend.common import emergency_checkpointing, train_state_nnx
from lilax._backend.common.checkpointing import (
    EmergencyCheckpointManager,
    EmergencyReplicatorCheckpointManager,
    all_steps,
    checkpoint_exception_guard,
    create_orbax_checkpoint_manager,
    is_saving_in_progress,
    latest_step,
    load_checkpoint_metadata,
    load_params_from_path,
    load_state_if_possible,
    maybe_save_checkpoint,
    reached_preemption,
    save_checkpoint,
    save_params_to_path,
    wait_until_finished,
)
from lilax._backend.common.emergency_checkpointing import (
    create_emergency_checkpoint_manager,
    create_replicator_checkpoint_manager,
)
from lilax._backend.configs.runtime import resolve_runtime_config
from lilax._backend.trainers.pre_train.train import get_first_step
from lilax._backend.training_engine.checkpointing import (
    CheckpointManager as EngineCheckpointManager,
    CheckpointRestoreError as EngineCheckpointRestoreError,
    CheckpointState as EngineCheckpointState,
)
from lilax._backend.utils import max_logging, maxtext_utils, maxtext_utils_nnx, model_creation_utils, train_utils
from lilax._backend.utils.generate_param_only_checkpoint import generate_decode_checkpoint as _generate_decode_checkpoint
from lilax._backend.utils.layerwise_quantization import quantize_checkpoint as _quantize_checkpoint
from lilax._backend.utils.model_creation_utils import from_config
from lilax.configuration import MaxTextConfig, RLConfig


def generate_decode_checkpoint(config):
  """Convert a training checkpoint into a parameter-only decode checkpoint."""
  if isinstance(config, (MaxTextConfig, RLConfig)):
    config = resolve_runtime_config(config)
  return _generate_decode_checkpoint(config)


def quantize_checkpoint(config, rng=None):
  """Quantize a parameter checkpoint using the layerwise utility."""
  if isinstance(config, (MaxTextConfig, RLConfig)):
    config = resolve_runtime_config(config)
  return _quantize_checkpoint(config, rng=rng)


def load_and_quantize_checkpoint(config):
  """Load and quantize a parameter checkpoint with the inference engine."""
  if isinstance(config, (MaxTextConfig, RLConfig)):
    config = resolve_runtime_config(config)
  return _load_and_quantize_checkpoint(config)


def _as_abstract_leaf(leaf):
  if isinstance(leaf, jax.Array):
    return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=leaf.sharding)
  return leaf


def checkpoint_loop(config):
  """Save and optionally restore checkpoints at the configured cadence."""
  if isinstance(config, (MaxTextConfig, RLConfig)):
    config = resolve_runtime_config(config)
  train_utils.validate_train_config(config)
  init_rng = jax.random.PRNGKey(config.init_weights_seed)
  mesh = maxtext_utils.get_mesh_from_config(config)
  rngs = maxtext_utils_nnx.create_nnx_rngs(config, rng_key=init_rng)
  model = from_config(config, mesh=mesh, rngs=rngs)
  _, tx = train_utils.create_training_optimizer(config, model)
  create_model, _ = model_creation_utils.create_nnx_abstract_model(config, mesh)

  def init_state_fn():
    nnx_model = create_model()
    wrt = (
        getattr(nnx, "LoRAParam", nnx.Param)
        if getattr(getattr(config, "lora", None), "enable_lora", False)
        else nnx.Param
    )
    optimizer = nnx.Optimizer(nnx_model, tx, wrt=wrt)
    return train_state_nnx.TrainStateNNX(nnx_model, optimizer)

  checkpoint_manager = train_utils.create_checkpoint_manager(config, mesh, init_state_fn)
  jax.experimental.multihost_utils.sync_global_devices("Barrier before load")
  state, _, _, _, _ = maxtext_utils.setup_training_state(None, config, mesh, checkpoint_manager, init_state_fn)

  jax.block_until_ready(state)
  state = add_entropy_to_checkpoint(state)
  start_step = get_first_step(model, state)
  for step in np.arange(start_step, config.steps):
    if checkpoint_manager is not None:
      start_time = datetime.datetime.now()
      jax.experimental.multihost_utils.sync_global_devices("Barrier before save")
      state_to_save = train_state_nnx.to_linen_checkpoint_dict(state.to_pure_dict())
      if save_checkpoint(checkpoint_manager, int(step), state_to_save):
        wait_until_finished(checkpoint_manager)
        end_time = datetime.datetime.now()
        if jax.process_index() == 0:
          max_logging.log(
              "STANDALONE CHECKPOINTER : Checkpoint saved in" f" {end_time - start_time} ,step {step}, on host 0"
          )
          elapsed_time = datetime.datetime.now() - start_time
          time_to_wait = config.standalone_checkpointer_per_step_interval - elapsed_time.total_seconds()
          if time_to_wait > 0:
            time.sleep(time_to_wait)
        jax.experimental.multihost_utils.sync_global_devices("Barrier after step")
        if config.standalone_checkpointer_enable_restore_in_loop:
          if jax.process_index() == 0 and config.standalone_checkpointer_drop_page_cache_before_restore:
            max_logging.log("STANDALONE CHECKPOINTER : Dropping OS page cache before restore...")
          if config.standalone_checkpointer_drop_page_cache_before_restore:
            os.system("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'")
          restore_start = datetime.datetime.now()
          abstract_state = jax.tree_util.tree_map(_as_abstract_leaf, state_to_save)
          if isinstance(checkpoint_manager, (EmergencyCheckpointManager, EmergencyReplicatorCheckpointManager)):
            restored_state = emergency_checkpointing.restore(checkpoint_manager, int(step), abstract_state)
          else:
            restored_state = checkpoint_manager.load_checkpointables(int(step), {"items": abstract_state})["items"]
          jax.block_until_ready(restored_state)
          restore_end = datetime.datetime.now()
          if jax.process_index() == 0:
            max_logging.log(
                f"STANDALONE CHECKPOINTER : Checkpoint restored in {restore_end - restore_start} ,step {step}, on host 0"
            )
  return state


def add_entropy_to_checkpoint(state):
  """Replace optimizer moments with functions of the model parameters."""
  if hasattr(state, "model"):
    if isinstance(state, nnx.Module):
      params = nnx.state(state.model, nnx.Param)
    else:
      params = state.model.filter(nnx.Param) if hasattr(state.model, "filter") else state.model
    new_mu = jax.tree_util.tree_map(lambda k: jnp.cos(1000 * k), params)
    new_nu = jax.tree_util.tree_map(lambda k: jnp.sin(1000 * k), params)
    if isinstance(state, nnx.Module):
      opt = state.optimizer
      opt.opt_state = (opt.opt_state[0]._replace(mu=new_mu, nu=new_nu),) + tuple(opt.opt_state[1:])
    else:
      state.optimizer.opt_state[0].mu = new_mu
      state.optimizer.opt_state[0].nu = new_nu
    return state
  opt_0 = state.opt_state[0]
  opt_0 = opt_0._replace(mu=jax.tree_util.tree_map(lambda k: jnp.cos(1000 * k), state.params))
  opt_0 = opt_0._replace(nu=jax.tree_util.tree_map(lambda k: jnp.sin(1000 * k), state.params))
  new_opt = [opt_0] + list(state.opt_state[1:])
  return state.replace(opt_state=new_opt)
