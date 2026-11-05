import jax.numpy as jnp

from lilax._backend.checkpoint_conversion.model_families.llama31 import fused as llama31_fused
from lilax._backend.checkpoint_conversion.projection_primitives import fuse_projections


def convert_native_state(state, config, *, fused_qkv, fused_mlp):
  def convert(node, path):
    if not isinstance(node, dict):
      return node
    result = {name: convert(value, path + (name,)) for name, value in node.items()}
    if not ("vision_encoder" in path and "Llama4VisionEncoder_0" in path
            and path[-1] == "self_attention_vision"):
      return result
    names = ("query", "key", "value")
    if "qkv_proj" in result and not fused_qkv:
      projection = result.pop("qkv_proj")
      if "kernel" not in projection or set(projection) - {"kernel", "bias"}:
        raise NotImplementedError("Fused Llama 4 vision conversion supports unadapted kernels only.")
      for name in names:
        result[name] = {}
      for parameter, value in projection.items():
        axis = 0 if parameter == "bias" else 1
        heads = config.num_attention_heads_for_vit
        if value.shape[axis] != 3 * heads:
          raise ValueError(f"Unexpected fused QKV shape: {value.shape}")
        for name, split in zip(names, jnp.split(value, [heads, 2 * heads], axis=axis)):
          result[name][parameter] = split
    elif fused_qkv and all(name in result and isinstance(result[name], dict) for name in names):
      projections = [result.pop(name) for name in names]
      parameters = set(projections[0])
      if "kernel" not in parameters or parameters - {"kernel", "bias"} or any(
          set(projection) != parameters for projection in projections
      ):
        raise NotImplementedError("Fused Llama 4 vision conversion supports unadapted kernels only.")
      result["qkv_proj"] = {
          parameter: fuse_projections([projection[parameter] for projection in projections],
                                      0 if parameter == "bias" else 1, array_module=jnp)
          for parameter in parameters
      }
    return result

  state = llama31_fused.convert_native_state(state, config, fused_qkv=fused_qkv, fused_mlp=fused_mlp)
  return convert(state, ())


def unfuse_native_state(state, config):
  return convert_native_state(state, config, fused_qkv=False, fused_mlp=False)
