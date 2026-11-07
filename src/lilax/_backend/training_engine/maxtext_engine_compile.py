"""Ahead-of-time compilation of native training engine kernels."""

import os
from typing import Any

from flax import nnx
import jax

from lilax._backend.common import common_types, train_state_nnx
from lilax._backend.configs import pyconfig
from lilax._backend.trainers.pre_train import train_compile as pre_train_compile
from lilax._backend.training_engine import kernel_compilation
from lilax._backend.training_engine import maxtext_engine
from lilax._backend.storage import gcs as gcs_utils
from lilax._backend.utils import maxtext_utils
from lilax._backend.observability.diagnostics import system_info
from lilax._backend.models.creation import nnx_abstract_model, scan_layers_sync

KERNEL_NAMES = kernel_compilation.KERNEL_NAMES
HLO_DUMP_DEFAULTS = {
    "dump_hlo_local_module_name": f"jit_({'|'.join(KERNEL_NAMES)})",
    "dump_hlo_module_name": "jit_",
}


def _propagation_mesh(mesh: jax.sharding.Mesh) -> jax.sharding.Mesh:
  axis_types = getattr(mesh, "axis_types", None)
  if axis_types is not None and all(axis_type == jax.sharding.AxisType.Explicit for axis_type in axis_types):
    return mesh
  return jax.sharding.Mesh(mesh.devices, mesh.axis_names, axis_types=(jax.sharding.AxisType.Explicit,) * len(mesh.axis_names))


def _rehome_aval(aval: Any, mesh: jax.sharding.Mesh) -> Any:
  if not hasattr(aval, "shape") or not hasattr(aval, "dtype"):
    return aval
  spec = getattr(getattr(aval, "sharding", None), "spec", None)
  target = jax.sharding.NamedSharding(mesh, spec) if spec is not None else None
  return jax.ShapeDtypeStruct(aval.shape, aval.dtype, sharding=target)


class AbstractMaxTextEngine(maxtext_engine.MaxTextTrainingEngine):
  """A training engine whose state consists of abstract shapes and shardings."""

  def __init__(
      self, training_config: pyconfig.HyperParameters, mesh: jax.sharding.Mesh, *, model: nnx.Module | None = None
  ) -> None:
    if mesh is None:
      raise ValueError("AbstractMaxTextEngine requires a mesh for the target topology.")
    super().__init__(training_config, mesh=mesh, model=model)

  def _build_model(self, wrap_with_tunix_adapter: bool, tokenizer_pad_id: int | None) -> Any:
    del wrap_with_tunix_adapter, tokenizer_pad_id
    _, abstract_model = nnx_abstract_model.create_nnx_abstract_model(
        scan_layers_sync.verify_and_sync_scan_layers(self._config),
        self._mesh,
        model_mode=common_types.MODEL_MODE_TRAIN,
        rng_key=self._init_rng,
    )
    return abstract_model

  def _build_optimizer(self, tx: Any) -> Any:
    self._state = self._trace_train_state(tx)
    self._model = self._state.model
    return self._state.optimizer

  def _trace_train_state(self, tx: Any) -> Any:
    model_graphdef, model_pure = nnx.split(self._model)

    def build(model_state):
      model = nnx.merge(model_graphdef, model_state)
      return train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, tx, wrt=nnx.Param))

    propagation_mesh = _propagation_mesh(self._mesh)
    with self._sharding_ctx():
      state_graphdef, _ = nnx.split(nnx.eval_shape(build, model_pure))
      with jax.set_mesh(propagation_mesh):
        state_pure = jax.eval_shape(
            lambda model_state: nnx.split(build(model_state))[1],
            jax.tree.map(lambda aval: _rehome_aval(aval, propagation_mesh), model_pure),
        )
    return nnx.merge(state_graphdef, jax.tree.map(lambda aval: _rehome_aval(aval, self._mesh), state_pure))

  def _checkpoint_dir(self) -> str:
    return ""

  def _place_leaf(self, leaf: Any, target: jax.sharding.Sharding) -> Any:
    return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=target)

  def _cannot_run(self, operation: str) -> RuntimeError:
    return RuntimeError(
        f"AbstractMaxTextEngine.{operation}() needs real weights, and this engine has only shapes. It "
        "exists to be traced and compiled: call `compile_kernels()`, or build a MaxTextTrainingEngine instead."
    )

  def fwd_bwd(self, payload: Any) -> None:
    raise self._cannot_run("fwd_bwd")

  def update(self) -> int:
    raise self._cannot_run("update")

  def fwd_only(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
    raise self._cannot_run("fwd_only")

  def save_checkpoint(self, metadata: Any = None, *, step: int | None = None, force: bool = False) -> None:
    raise self._cannot_run("save_checkpoint")

  def restore_checkpoint(self, *, step: int | None = None) -> Any:
    raise self._cannot_run("restore_checkpoint")


def get_shaped_micro_batch(config: pyconfig.HyperParameters) -> dict[str, jax.ShapeDtypeStruct]:
  shaped_batch = maxtext_utils.get_shaped_batch(config)
  micro_batch_size = int(config.micro_batch_size_to_train_on)

  def to_micro_batch(aval: jax.ShapeDtypeStruct) -> jax.ShapeDtypeStruct:
    if not aval.shape or aval.shape[0] == micro_batch_size:
      return aval
    return jax.ShapeDtypeStruct((micro_batch_size,) + aval.shape[1:], aval.dtype)

  return {key: to_micro_batch(aval) for key, aval in shaped_batch.items()}


def compile_engine_kernels(
    config: pyconfig.HyperParameters, topology_mesh: jax.sharding.Mesh, *, model: nnx.Module | None = None
) -> dict[str, Any]:
  return AbstractMaxTextEngine(config, topology_mesh, model=model).compile_kernels(get_shaped_micro_batch(config))


def kernel_save_path(compiled_trainstep_file: str, kernel_name: str) -> str:
  stem, extension = os.path.splitext(compiled_trainstep_file)
  return f"{stem}_{kernel_name}{extension}"


def compile_engine(config: pyconfig.HyperParameters, *, model: nnx.Module | None = None) -> dict[str, Any]:
  print("Starting training_engine/maxtext_engine_compile.py...", flush=True)
  pre_train_compile.validate_config(config)
  if config.enable_diloco:
    raise NotImplementedError(
        "enable_diloco is not supported here: MaxTextTrainingEngine has no DiLoCo outer step, so the "
        "numbers reported would describe a different computation."
    )
  topology_mesh = pre_train_compile.get_topology_mesh(config)
  system_info.print_system_information()
  print("Jitting and compiling the engine's kernels...", flush=True)
  compiled = compile_engine_kernels(config, topology_mesh, model=model)
  print("Jitting and compilation complete!", flush=True)
  for name in KERNEL_NAMES:
    print(f"--- {name} ---")
    print(f"Cost analysis: {compiled[name].cost_analysis()}")
    print(f"Memory analysis: {compiled[name].memory_analysis()}")
  if config.compiled_trainstep_file != "":
    for name in KERNEL_NAMES:
      save_path = kernel_save_path(config.compiled_trainstep_file, name)
      pre_train_compile.save_compiled(compiled[name], save_path)
      print(f"Successfully saved compiled {name} kernel as {save_path}")
  print("Finished training_engine/maxtext_engine_compile.py successfully!", flush=True)
  if config.dump_hlo:
    if not os.path.isdir(config.dump_hlo_local_dir):
      raise FileNotFoundError(
          f"dump_hlo is set but XLA wrote nothing to {config.dump_hlo_local_dir}: "
          f"dump_hlo_local_module_name={config.dump_hlo_local_module_name!r} matched none of the engine's "
          "kernels (jit_first_kernel, jit_accum_kernel, jit__update_kernel)."
      )
    gcs_utils.upload_dump(
        config.dump_hlo_local_dir, config.dump_hlo_gcs_dir, module_name=config.dump_hlo_module_name,
        delete_local_after=config.dump_hlo_delete_local_after, all_host_upload=config.dump_hlo_upload_all,
    )
  return compiled
