"""Bridge checkpoints storing separate MoE input projections and models using a fused projection."""

import jax
import jax.numpy as jnp

from lilax._backend.integration.vllm.convert_utils import _partition_size


def _fuse_moe_weights(ckpt_tree, model_arrays_tree):
  """Fuse and interleave separate MoE input projections for the target sharding."""
  def _is_fusion_site(node):
    return (
        isinstance(node, dict)
        and "wi_0" in node
        and "wi_1" in node
        and not isinstance(node["wi_0"], dict)
        and not isinstance(node["wi_1"], dict)
    )

  def _key_str(key):
    if hasattr(key, "key"):
      return key.key
    if hasattr(key, "attr"):
      return key.attr
    if hasattr(key, "name"):
      return key.name
    return key

  def _lookup_model(path):
    node = model_arrays_tree
    for key in path:
      name = _key_str(key)
      if isinstance(node, dict) and name in node:
        node = node[name]
      else:
        return None
    return node

  def _maybe_fuse(path, ckpt_node):
    if not _is_fusion_site(ckpt_node):
      return ckpt_node
    model_node = _lookup_model(path)
    if not isinstance(model_node, dict) or "wi" not in model_node:
      return ckpt_node

    wi_model = model_node["wi"]
    axis = wi_model.ndim - 1
    n_shards = 1
    wi_sharding = getattr(wi_model, "sharding", None)
    if isinstance(wi_sharding, jax.sharding.NamedSharding):
      spec = wi_sharding.spec
      partition = spec[axis] if axis < len(spec) else None
      n_shards = _partition_size(partition, wi_sharding.mesh)
    target_half_dim = wi_model.shape[-1] // 2

    def _pad_and_chunk(arr, target_total_size):
      shape = arr.shape
      chunk_size = shape[-1] // n_shards
      target_chunk_size = target_total_size // n_shards
      pad_amount = target_chunk_size - chunk_size
      arr_reshaped = arr.reshape(*shape[:-1], n_shards, chunk_size)
      if pad_amount > 0:
        pad_widths = [(0, 0)] * arr_reshaped.ndim
        pad_widths[-1] = (0, pad_amount)
        arr_reshaped = jnp.pad(arr_reshaped, pad_widths)
      return arr_reshaped

    padded_chunked_wi_0 = _pad_and_chunk(ckpt_node["wi_0"], target_half_dim)
    padded_chunked_wi_1 = _pad_and_chunk(ckpt_node["wi_1"], target_half_dim)
    wi_interleaved = jnp.concatenate([padded_chunked_wi_0, padded_chunked_wi_1], axis=-1)
    new_node = {key: value for key, value in ckpt_node.items() if key not in ("wi_0", "wi_1")}
    new_node["wi"] = wi_interleaved.reshape(*wi_model.shape)
    return new_node

  return jax.tree_util.tree_map_with_path(_maybe_fuse, ckpt_tree, is_leaf=_is_fusion_site)


def adjust_target(target, meta_tree, is_nnx, use_value_wrapper=True):
  """Replace fused 'wi' restore targets with 'wi_0'/'wi_1' halves where the checkpoint stores them separately."""
  if not hasattr(target, "items") or not hasattr(meta_tree, "items"):
    return target
  new_target = {}
  for key, value in target.items():
    if key == "wi" and "wi" not in meta_tree and "wi_0" in meta_tree and "wi_1" in meta_tree:
      array = value["value"] if is_nnx and use_value_wrapper else value
      half_dim = array.shape[-1] // 2
      split = jax.ShapeDtypeStruct(shape=array.shape[:-1] + (half_dim,), dtype=array.dtype, sharding=array.sharding)
      new_target["wi_0"] = {"value": split} if is_nnx and use_value_wrapper else split
      new_target["wi_1"] = {"value": split} if is_nnx and use_value_wrapper else split
    else:
      new_target[key] = adjust_target(value, meta_tree.get(key, {}), is_nnx, use_value_wrapper)
  return new_target
