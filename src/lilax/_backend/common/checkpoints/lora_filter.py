"""Filter checkpoint state to LoRA trainable values."""

from flax import nnx


def _filter_lora_trainable_state(state, preserve_optimizer=False):
  def _lora_filter(path, val):
    path_str = "/".join(str(part) for part in path).lower()
    return isinstance(val, nnx.LoRAParam) or "lora" in path_str or "step" in path_str

  if isinstance(state, (nnx.State, nnx.Module)):
    return nnx.state(state, _lora_filter)

  def _filter_dict(val, path=()):
    if path == ("nnx_aux",) or (preserve_optimizer and path == ("opt_state",)):
      return val
    if isinstance(val, dict):
      res = {}
      for key, value in val.items():
        curr_path = path + (str(key),)
        filtered = _filter_dict(value, curr_path)
        if filtered is not None:
          res[key] = filtered
      return res if res else None
    path_str = "/".join(path).lower()
    if "lora" in path_str or "step" in path_str:
      return val
    return None

  return _filter_dict(state)
