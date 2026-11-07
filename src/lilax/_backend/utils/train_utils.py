"""Utils that are only interesting for training in MaxText."""

import functools
import subprocess
import threading
import time

from flax import nnx
from flax.core.spmd import logical_axis_rules
import jax
from jax.experimental import multihost_utils
import numpy as np
import optax
import orbax.checkpoint.pathways as ocp_pathways

from lilax._backend.common.checkpoints import checkpointing
from lilax._backend.common.checkpoints import emergency_checkpointing
from lilax._backend.common import train_state_nnx
from lilax._backend.common.common_types import ReorderStrategy
from lilax._backend.common.data_loader import create_dataloader
from lilax._backend.common.goodput import GoodputEvent, maybe_record_goodput
from lilax._backend.input_pipeline import multihost_dataloading
from lilax._backend.input_pipeline.input_pipeline_interface import create_data_iterator
from lilax._backend.optimizers import optimizers
from lilax._backend.trainers.diloco import diloco
from lilax._backend.distributed import device_mesh
from lilax._backend.trainers.diloco import diloco_sharding
from lilax._backend.utils import elastic_utils
from lilax._backend.utils import lora_utils
from lilax._backend.observability import max_logging
from lilax._backend.utils import maxtext_utils
from lilax._backend.distributed import sharding
from lilax._backend.distributed import state_abstraction
from lilax._backend.layers.context_parallel import sequence_reorder
from lilax._backend.observability.diagnostics import mesh_report
from lilax._backend.observability.diagnostics import sharding_report
from lilax._backend.models.creation import nnx_abstract_model
from lilax._backend.integration.transformer_engine import moe_bootstrap
from lilax._backend.trainers.setup import state_setup
from lilax._backend.trainers.setup.compilation import jit_eval_step, jit_train_and_eval_step, jit_train_step
from lilax._backend.trainers.setup.rampup_batch import create_rampup_manager
from lilax._backend.trainers.pre_train import checkpoint_progress


def prepare_before_run_start(
    config, state, learning_rate_schedule, start_step, mesh, data_loader, shaped_batch, shaped_eval_batch=None
):
  """Run optional setup before the timed run without touching the dataset."""
  timings = []
  if config.warm_input_reshard_before_run_start:
    start = time.perf_counter()
    local_rows = config.global_batch_size_to_load // jax.process_count()

    def synthetic_global(leaf):
      local = np.zeros((local_rows,) + tuple(leaf.shape[1:]), dtype=leaf.dtype)
      return multihost_dataloading._form_global_array((), local, mesh)

    warm = jax.device_put(jax.tree.map(synthetic_global, shaped_batch), data_loader.input_data_shardings)
    jax.block_until_ready(warm)
    del warm
    timings.append(f"input reshard warm {time.perf_counter() - start:.3f} s")

  if config.warm_eval_input_reshard_before_run_start and shaped_eval_batch is not None:
    start = time.perf_counter()
    local_rows_eval = config.global_batch_size_to_load_eval // jax.process_count()

    def synthetic_global_eval(leaf):
      local = np.zeros((local_rows_eval,) + tuple(leaf.shape[1:]), dtype=leaf.dtype)
      return multihost_dataloading._form_global_array((), local, mesh)

    eval_input_sharding = sharding.get_input_data_sharding(config, mesh, rules=config.logical_axis_rules_for_eval)
    warm_eval = jax.device_put(jax.tree.map(synthetic_global_eval, shaped_eval_batch), eval_input_sharding)
    jax.block_until_ready(warm_eval)
    del warm_eval
    timings.append(f"eval input reshard warm {time.perf_counter() - start:.3f} s")

  if config.block_state_before_run_start:
    start = time.perf_counter()
    jax.block_until_ready(state)
    timings.append(f"state ready {time.perf_counter() - start:.3f} s")
    start = time.perf_counter()
    jax.block_until_ready(learning_rate_schedule(start_step))
    timings.append(f"lr schedule warm {time.perf_counter() - start:.3f} s")
    start = time.perf_counter()
    multihost_utils.sync_global_devices("pre_run_start")
    timings.append(f"barrier {time.perf_counter() - start:.3f} s")

  if timings:
    max_logging.log("Pre-run_start setup: " + ", ".join(timings))


def start_eval_cache_prefill(eval_data_iterator):
  """Fill a tf.data evaluation cache on a background thread when available."""
  inner = eval_data_iterator
  for _ in range(8):
    if hasattr(inner, "dataloader") or not hasattr(inner, "data_iterator"):
      break
    inner = inner.data_iterator
  dataset = getattr(inner, "dataloader", None)
  if dataset is None or not hasattr(dataset, "as_numpy_iterator"):
    max_logging.log(
        f"eval_cache_prefill_in_background: eval loader {type(eval_data_iterator).__name__} -> {type(inner).__name__} "
        "is not a tf.data pipeline; skipped"
    )
    return None

  def run():
    start = time.perf_counter()
    count = 0
    try:
      for _ in dataset.as_numpy_iterator():
        count += 1
      max_logging.log(f"eval cache prefill: {count} batches in {time.perf_counter() - start:.3f} s")
    except Exception as error:
      max_logging.log(f"eval cache prefill failed ({error!r}); the first eval will read the eval files itself")

  thread = threading.Thread(target=run, name="eval_cache_prefill", daemon=True)
  thread.start()
  return thread


def create_training_optimizer(config, model, mesh=None):
  """Create the optimizer and learning rate schedule."""
  learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(config)
  tx = optimizers.get_optimizer(config, learning_rate_schedule, model, mesh=mesh)
  return learning_rate_schedule, tx


def create_checkpoint_manager(config, mesh, init_state_fn):
  """Create the configured checkpoint manager."""
  logger = checkpointing.setup_checkpoint_logger(config)
  if config.enable_multi_tier_checkpointing:
    checkpoint_manager = emergency_checkpointing.create_replicator_checkpoint_manager(
        config.local_checkpoint_directory, config.local_checkpoint_period, mesh, config.colocated_python_checkpointing
    )
  elif config.enable_emergency_checkpoint:
    abstract_state, _, _ = state_abstraction.get_abstract_state(config, mesh, init_state_fn, is_training=True)
    checkpoint_manager = emergency_checkpointing.create_emergency_checkpoint_manager(
        config.local_checkpoint_directory, config.checkpoint_dir, mesh, abstract_state,
        config.local_checkpoint_period, config.checkpoint_period, logger,
    )
  else:
    use_ocdbt = config.checkpoint_storage_use_ocdbt
    use_zarr3 = config.checkpoint_storage_use_zarr3
    if config.enable_single_controller and not config.colocated_python_checkpointing:
      use_ocdbt, use_zarr3 = False, False
    checkpoint_dir = config.checkpoint_dir if config.enable_checkpointing else ""
    checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
        checkpoint_dir,
        config.enable_checkpointing,
        config.async_checkpointing,
        config.checkpoint_period,
        config.dataset_type,
        logger,
        use_ocdbt,
        use_zarr3,
        config.enable_continuous_checkpointing,
        config.max_num_checkpoints_to_keep,
        config.checkpoint_storage_concurrent_gb,
        config.enable_autocheckpoint,
        config.checkpoint_todelete_subdir,
        config.checkpoint_todelete_full_path,
        config.checkpoint_storage_target_data_file_size_bytes,
        colocated_python_checkpointing=config.enable_single_controller and config.colocated_python_checkpointing,
    )

  if checkpoint_manager is not None and config.enable_single_controller and config.colocated_python_checkpointing:
    max_logging.log("Registering colocated python array handler")
    checkpointing_impl = ocp_pathways.CheckpointingImpl.from_options(use_colocated_python=True)
    ocp_pathways.register_type_handlers(
        use_single_replica_array_handler=config.enable_single_replica_ckpt_restoring,
        checkpointing_impl=checkpointing_impl,
    )
  return checkpoint_manager


class _ReorderedDataIterator:
  """Apply a reorder function to each batch."""

  def __init__(self, reorder_fn, data_iterator):
    self.reorder_fn = reorder_fn
    self.data_iterator = data_iterator

  def __iter__(self):
    return self

  def __next__(self):
    return self.reorder_fn(next(self.data_iterator))

  def reset(self):
    self.data_iterator.reset()


def _reorder_data_iterator_for_loader(reorder_fn, data_iterator):
  """Wrap an iterator or each iterator in a list with the reorder view."""
  if isinstance(data_iterator, list):
    return [_ReorderedDataIterator(reorder_fn, iterator) for iterator in data_iterator]
  return _ReorderedDataIterator(reorder_fn, data_iterator)


_UNSET_CHECKPOINT_MANAGER = object()


def setup_train_loop(
    config, recorder, devices=None, elastic_retry_cache=None, *, model=None, mesh=None,
    runtime=None, data_iterator=None, eval_data_iterator=None, checkpoint_step=-1,
    checkpoint_manager=_UNSET_CHECKPOINT_MANAGER, selected_source=None,
    tx=None, learning_rate_schedule=None,
):
  """Set up training state, checkpointing, model, mesh, and data loaders."""
  supplied_model = model
  with maybe_record_goodput(recorder, GoodputEvent.TPU_INIT):
    if runtime is not None:
      runtime.resolve_config()
      mesh_fields = (
          "mesh_axes", "shard_mode", "ici_parallelism", "dcn_parallelism", "num_slices", "elastic_enabled",
          "subslice_shape", "enable_single_controller", "inference_benchmark_test", "allow_split_physical_axes",
          "use_lineage", "custom_mesh", "optimize_mesh_for_tpu_v6e",
      )
      mismatched_fields = [field for field in mesh_fields if getattr(config, field) != getattr(runtime.config, field)]
      if mismatched_fields:
        raise ValueError(f"Training and runtime physical mesh configuration differs: {', '.join(mismatched_fields)}")
      if mesh is not None and mesh is not runtime.mesh:
        raise ValueError("The supplied mesh must be the runtime's mesh.")
      if devices is None:
        devices = runtime.devices
      elif runtime.devices is None or (
          devices is not runtime.devices
          and (len(devices) != len(runtime.devices) or any(
              device is not runtime_device for device, runtime_device in zip(devices, runtime.devices)
          ))
      ):
        raise ValueError("The supplied devices must match the runtime's devices in order and identity.")
    init_rng = jax.random.PRNGKey(config.init_weights_seed)
    if runtime is not None:
      if runtime.mesh is None:
        runtime.initialize(elastic_retry_cache)
      mesh = runtime.mesh
    elif mesh is None:
      mesh = elastic_utils.get_or_build(
          elastic_retry_cache, "mesh", functools.partial(device_mesh.get_mesh_from_config, config, devices)
      )
    context_parallel_size = mesh.shape.get(config.context_sharding, 1)
    if model is None:
      create_model, model = elastic_utils.get_or_build(
          elastic_retry_cache, "abstract_model",
          functools.partial(nnx_abstract_model.create_nnx_abstract_model, config, mesh, devices),
      )
      init_state_args = ()
    else:
      supplied_graphdef, supplied_state = nnx.split(supplied_model)
      init_state_args = (supplied_state,)

      def create_model(model_state=supplied_state):
        return nnx.merge(supplied_graphdef, model_state)

    if learning_rate_schedule is None:
      learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(config)
    if tx is None:
      tx = optimizers.get_optimizer(config, learning_rate_schedule, model, mesh=mesh)

    def create_train_state_fn(*model_args):
      train_model = create_model(*model_args)
      wrt = (
          getattr(nnx, "LoRAParam", nnx.Param)
          if getattr(getattr(config, "lora", None), "enable_lora", False)
          else nnx.Param
      )
      optimizer = nnx.Optimizer(train_model, tx, wrt=wrt)
      return train_state_nnx.TrainStateNNX(train_model, optimizer)

    init_state_fn = create_train_state_fn
    if checkpoint_manager is _UNSET_CHECKPOINT_MANAGER:
      checkpoint_manager = create_checkpoint_manager(config, mesh, init_state_fn)
    if checkpoint_step >= 0 and checkpoint_manager is None:
      raise ValueError(f"Cannot restore checkpoint step {checkpoint_step} without a checkpoint manager.")
    if checkpoint_manager is not None:
      restore_step = (
          selected_source.step if selected_source is not None and selected_source.kind == "manager"
          else None if selected_source is not None
          else checkpointing.latest_step(checkpoint_manager) if checkpoint_step < 0 else checkpoint_step
      )
      if restore_step is not None:
        checkpoint_progress.validate_completed_steps(restore_step + 1, config.steps)

  with maybe_record_goodput(recorder, GoodputEvent.TRAINING_PREPARATION):
    if data_iterator is None:
      data_iterator, created_eval_iterator = create_data_iterator(config, mesh)
      if eval_data_iterator is None:
        eval_data_iterator = created_eval_iterator
    rampup_manager = create_rampup_manager(config, checkpoint_manager)
    context_parallel_strategy = config.context_parallel_strategy.lower()
    if context_parallel_size > 1 and config.packing:
      if context_parallel_strategy not in ("all_gather", "ring", "ulysses", "usp"):
        raise ValueError(
            "Context parallelism with sequence packing supports context_parallel_strategy='all_gather', 'ring', "
            "'ulysses', or 'usp'."
        )
      if (
          config.hardware in ("gpu", "gpu_multiprocess")
          and config.attention == "cudnn_flash_te"
          and not (context_parallel_strategy == "ring" and config.context_parallel_load_balance)
      ):
        raise ValueError("Packing is only supported for load balanced ring attention with context parallelism for GPU.")

    data_iterator_for_loader = data_iterator
    reorder_cp_size = sequence_reorder.reordered_cp_size(config, mesh)
    with jax.set_mesh(mesh):
      if reorder_cp_size > 1:
        if config.context_parallel_reorder_strategy == ReorderStrategy.AUTO:
          reorder_strategy = (
              ReorderStrategy.STRIPED
              if config.packing and context_parallel_strategy == "ring" and config.hardware in ("gpu", "gpu_multiprocess")
              else ReorderStrategy.DUAL_CHUNK_SWAP
          )
        else:
          reorder_strategy = config.context_parallel_reorder_strategy
        reorder_fn = maxtext_utils.get_reorder_callable(
            reorder_cp_size, config.shard_mode, reorder_strategy, config.hardware
        )
        data_iterator_for_loader = _reorder_data_iterator_for_loader(reorder_fn, data_iterator)
        if eval_data_iterator:
          eval_data_iterator = _ReorderedDataIterator(reorder_fn, eval_data_iterator)

    data_loader = create_dataloader(config, mesh, data_iterator_for_loader, recorder, rampup_manager)
    shaped_batch = maxtext_utils.get_shaped_batch(config)
    moe_bootstrap.maybe_bootstrap_te_moe(config, mesh, shaped_batch)
    snapshotter = None if elastic_retry_cache is None else elastic_retry_cache.snapshotter
    state, _, state_mesh_shardings, data_iterator, _ = state_setup.setup_training_state(
        data_iterator, config, mesh, checkpoint_manager, init_state_fn, snapshotter=snapshotter, step=checkpoint_step,
        selected_source=selected_source, init_state_args=init_state_args,
    )
    if getattr(getattr(config, "lora", None), "enable_lora", False) and getattr(config.lora, "lora_restore_path", None):
      target_model_state = (
          state["model"] if isinstance(state, (nnx.State, dict)) and "model" in state else getattr(state, "model", state)
      )
      lora_utils.restore_lora_from_path(target_model_state, config)
      _, _, state_mesh_shardings = state_abstraction.get_abstract_state_nnx(config, mesh, init_state_fn, True)
    with logical_axis_rules(config.logical_axis_rules):
      state_graphdef = nnx.graphdef(nnx.eval_shape(init_state_fn))

    if isinstance(state, diloco.DiLoCoTrainState):
      state_params = state.params
      if hasattr(state_mesh_shardings, "model"):
        _, state_mesh_shardings_params, _ = nnx.split(state_mesh_shardings.model, nnx.Param, ...)
      else:
        state_mesh_shardings_params = state_mesh_shardings.params
    else:
      with logical_axis_rules(config.logical_axis_rules):
        _, state_params, _ = nnx.split(state.model, nnx.Param, ...)
        _, state_mesh_shardings_params, _ = nnx.split(state_mesh_shardings.model, nnx.Param, ...)

    if config.enable_diloco:
      with jax.set_mesh(mesh), logical_axis_rules(config.logical_axis_rules):
        outer_params_sharding = (
            state_mesh_shardings_params.to_pure_dict()
            if isinstance(state_mesh_shardings_params, nnx.State)
            else state_mesh_shardings_params
        )
        if not isinstance(state, diloco.DiLoCoTrainState):
          state, outer_opt_state_sharding = diloco.build_diloco_state(config, lambda: state, mesh=mesh)
        else:
          outer_opt_state_sharding = (optax.TraceState(trace=outer_params_sharding), optax.EmptyState())
        step_mesh = state_mesh_shardings.optimizer.step.mesh
        inner_state_shardings = diloco_sharding.add_diloco_to_sharding(state_mesh_shardings)
        state_mesh_shardings = diloco.DiLoCoTrainState(
            inner_state_shardings, outer_params_sharding, outer_opt_state_sharding,
            jax.sharding.NamedSharding(mesh=step_mesh, spec=jax.sharding.PartitionSpec()),
        )

    if not config.using_pipeline_parallelism and not config.use_multimodal:
      sharding.assert_params_sufficiently_sharded(state_params, mesh, config.sharding_tolerance)
    if config.debug_sharding:
      logical_annotations_params = None
      mesh_report.print_non_trivial_mesh_axis(model.mesh)
      sharding_report.print_shardings_params(state_params, state_mesh_shardings_params, mesh, logical_annotations_params)

  if config.enable_diloco:
    train_state = state
    model = state_graphdef
  else:
    train_state = nnx.merge(state_graphdef, state)
    if supplied_model is not None:
      nnx.update(supplied_model, nnx.state(train_state.model))
      train_state.model = supplied_model
    model = train_state.model

  return (
      init_rng, checkpoint_manager, state_mesh_shardings, model, mesh, learning_rate_schedule,
      data_iterator, data_loader, rampup_manager, eval_data_iterator, train_state,
  )


def validate_train_config(config):
  """Validate configuration for training."""
  if getattr(config, "use_dpo", False):
    raise ValueError("Legacy DPO implementation in train.py is removed. Please use post-training train_dpo.py instead.")
  assert config.run_name, "Erroring out, need a real run_name"
  if config.dataset_path and not config.dataset_path.startswith("gs://"):
    max_logging.log("WARNING: 'dataset_path' might be pointing your local file system")
  if not config.base_output_directory.startswith("gs://"):
    max_logging.log("WARNING: 'base_output_directory' might be pointing your local file" " system")
  assert config.steps > 0, "You must set steps or learning_rate_schedule_steps to a positive integer."
  if config.quantization in ("fp8", "nanoo_fp8"):
    assert config.gradient_accumulation_steps == 1, (
        "fp8 can't be used with gradient_accumulation_steps right now. Please"
        " use other quantization or set gradient_accumulation_steps to 1"
    )
  if config.packing and config.dataset_type == "synthetic":
    max_logging.log(
        "WARNING: Sequence packing is essentially ignored for synthetic data. "
        "Please use a real dataset to use sequence packing."
    )


