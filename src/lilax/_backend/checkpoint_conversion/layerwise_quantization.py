"""Quantize and save a MaxText parameter checkpoint."""

from typing import Any

from aqt.jax.v2 import aqt_tensor
from flax import nnx
from flax.core.spmd import logical_axis_rules
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from lilax._backend.common.checkpoints import checkpointing
from lilax._backend.common import common_types
from lilax._backend.layers.quantization.configure import configure_quantization
from lilax._backend.distributed import device_mesh
from lilax._backend.observability import max_logging
from lilax._backend.utils import maxtext_utils_nnx
from lilax._backend.models.creation import model_factory, pretrained

IGNORE = ocp.PLACEHOLDER
PRNGKeyType = Any
DictKey = jax.tree_util.DictKey


def get_original_path_key(aqt_k_tuple: tuple[DictKey, ...]) -> tuple[DictKey, ...] | None:
  path = list(aqt_k_tuple)
  text = jax.tree_util.keystr(aqt_k_tuple)
  if "AqtEinsum_" in text or "AqtDotGeneral_" not in text:
    return None
  for i, key in enumerate(path):
    if isinstance(key, DictKey) and key.key.startswith("AqtDotGeneral_"):
      if (len(path) > i + 2 and isinstance(path[i + 1], DictKey) and path[i + 1].key == "qrhs"
          and isinstance(path[i + 2], DictKey) and path[i + 2].key == "frozen"):
        return tuple(path[:i]) + (DictKey("kernel"),)
      return None
  return None


def get_quantized_param_paths(aqt_params: Any, params: Any) -> set[tuple[DictKey, ...]]:
  aqt_flat, _ = jax.tree_util.tree_flatten_with_path(aqt_params, is_leaf=lambda x: isinstance(x, aqt_tensor.QTensor))
  if not aqt_flat:
    return set()
  param_flat, _ = jax.tree_util.tree_flatten_with_path(params)
  paths = {tuple(path) for path, _ in param_flat}
  removed = set()
  for path, _ in aqt_flat:
    original = get_original_path_key(path)
    if original is None:
      continue
    if original in paths:
      removed.add(original)
      continue
    available = {jax.tree_util.keystr(key) for key in paths}
    raise ValueError(
        f"Mapped AQT path {jax.tree_util.keystr(path)} to {jax.tree_util.keystr(original)},"
        f" but not found in params. Available: {available}"
    )
  return removed


def remove_quantized_params(params: Any, aqt_vars: Any) -> Any:
  paths = get_quantized_param_paths(aqt_vars, params)
  if not paths:
    return params
  return jax.tree_util.tree_map_with_path(lambda path, value: {} if tuple(path) in paths else value, params)


def insert_deepseekmoeblock_scope(aqt_layer_tree: dict[str, Any]) -> dict[str, Any]:
  if not isinstance(aqt_layer_tree, dict):
    return aqt_layer_tree
  tree = dict(aqt_layer_tree)
  einsums = {key: tree.pop(key) for key in list(tree) if key.startswith("AqtEinsum_")}
  if einsums:
    if "DeepSeekMoeBlock_0" not in tree:
      max_logging.log("Error: 'DeepSeekMoeBlock_0' not found in AQT vars for MoE layer.")
      tree["DeepSeekMoeBlock_0"] = {}
    elif not isinstance(tree["DeepSeekMoeBlock_0"], dict):
      max_logging.log(f"Error: 'DeepSeekMoeBlock_0' is not a dict, type: {type(tree['DeepSeekMoeBlock_0'])}")
      tree["DeepSeekMoeBlock_0"] = {}
    tree["DeepSeekMoeBlock_0"].update(einsums)
  return tree


class LayerwiseQuantization:
  def __init__(self, config: Any, rng: PRNGKeyType):
    self.config = config
    self.rng = rng
    devices = device_mesh.create_device_mesh(config=config)
    self._mesh = jax.sharding.Mesh(devices, config.mesh_axes)
    self.quant = configure_quantization(config)
    self.unboxed_abstract_state = None

  def load_and_quantize(self) -> None:
    self._load_and_quantize_nnx()

  def _load_and_quantize_nnx(self) -> None:
    config = self.config
    max_logging.log("Loading full-precision NNX checkpoint in TRAIN mode...")
    with self._mesh:
      train_model = pretrained.from_pretrained(
          config, mesh=self._mesh, model_mode=common_types.MODEL_MODE_TRAIN, quant_mode_str="train"
      )
    max_logging.log("Building CONVERT-mode model (random init) and copying kernels in...")
    rngs = maxtext_utils_nnx.create_nnx_rngs(config, rng_key=self.rng)
    with logical_axis_rules(config.logical_axis_rules):
      convert_model = model_factory.from_config(
          config, mesh=self._mesh, rngs=rngs, model_mode=common_types.MODEL_MODE_TRAIN, quant_mode_str="convert"
      )
    self._copy_kernel_leaves_(convert_model, train_model)
    del train_model
    length = config.max_target_length
    tokens = jnp.zeros((1, length), dtype=jnp.int32)
    positions = jnp.arange(length, dtype=jnp.int32)[None, :]
    segments = jnp.ones((1, length), dtype=jnp.int32)
    max_logging.log("Running CONVERT-mode forward to populate AQT scale factors...")
    with logical_axis_rules(config.logical_axis_rules):
      convert_model(tokens, positions, decoder_segment_ids=segments, enable_dropout=False,
                    model_mode=common_types.MODEL_MODE_TRAIN)
    state = (nnx.state(convert_model)
             .filter(lambda path, var: not isinstance(var, (nnx.RngState, nnx.Cache, nnx.Intermediate, nnx.BatchStat)))
             .to_pure_dict())
    state = self._strip_kernels_at_quantized_paths(state)
    if config.save_quantized_params_path:
      max_logging.log(f"Saving NNX-format quantized checkpoint to {config.save_quantized_params_path}")

      def wrap(node):
        return {key: wrap(value) for key, value in node.items()} if isinstance(node, dict) else {"value": node}

      checkpointer = ocp.PyTreeCheckpointer(
          use_ocdbt=config.checkpoint_storage_use_ocdbt, use_zarr3=config.checkpoint_storage_use_zarr3
      )
      checkpointer.save(config.save_quantized_params_path, wrap(state), force=True)
      max_logging.log(f"Saved NNX-format quantized checkpoint at: {config.save_quantized_params_path}")
    else:
      max_logging.log("Skipping save: save_quantized_params_path is null.")

  @staticmethod
  def _copy_kernel_leaves_(dst_model, src_model):
    src = nnx.state(src_model).to_pure_dict()
    dst_state = nnx.state(dst_model)
    dst = dst_state.to_pure_dict()

    def walk(target, source):
      if not (isinstance(target, dict) and isinstance(source, dict)):
        return
      for key, child in target.items():
        if key not in source:
          continue
        if key in ("kernel", "embedding", "scale", "bias") and not isinstance(child, dict):
          target[key] = source[key]
        elif isinstance(child, dict):
          walk(child, source[key])

    walk(dst, src)
    nnx.replace_by_pure_dict(dst_state, dst)
    nnx.update(dst_model, dst_state)

  @staticmethod
  def _strip_kernels_at_quantized_paths(state_dict):
    if not isinstance(state_dict, dict):
      return state_dict
    has_aqt = "AqtDotGeneral_0" in state_dict
    return {
        key: LayerwiseQuantization._strip_kernels_at_quantized_paths(value) if isinstance(value, dict) else value
        for key, value in state_dict.items() if key != "kernel" or not has_aqt
    }

  def _load_layer(self, layer_name):
    config = self.config
    with logical_axis_rules(config.logical_axis_rules):
      return checkpointing.load_params_from_path(
          config.load_parameters_path, self._create_partial_abstract_params(self.unboxed_abstract_state.params, layer_name),
          config.checkpoint_storage_concurrent_gb, config.checkpoint_storage_use_ocdbt, config.checkpoint_storage_use_zarr3,
      )

  def _create_partial_abstract_params(self, abstract_unboxed_params, layer):
    def map_leaf(path, value):
      if not any(isinstance(key, jax.tree_util.DictKey) and key.key == layer for key in path):
        return IGNORE
      if isinstance(value, jax.ShapeDtypeStruct):
        zeros = jnp.zeros(value.shape, value.dtype)
        if value.sharding is not None:
          try:
            return jax.device_put(zeros, value.sharding)
          except Exception as error:
            max_logging.log(f"Error applying sharding for path {path}: {error}")
        return zeros
      return value
    return jax.tree_util.tree_map_with_path(map_leaf, abstract_unboxed_params)


def validate_config(config):
  assert config.load_full_state_path == "", "Operation on full states not supported! Convert to parameter checkpoint first."


def quantize_checkpoint(config, rng=None):
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  validate_config(config)
  LayerwiseQuantization(config, jax.random.PRNGKey(1234) if rng is None else rng).load_and_quantize()
