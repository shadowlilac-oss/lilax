"""Translate m3 container layer names to and from legacy checkpoint layouts."""

from collections.abc import Mapping


def _is_index(key):
  return isinstance(key, int) or (isinstance(key, str) and key.isdecimal())


def _indexed_layers(subtree):
  if isinstance(subtree, Mapping) and set(subtree) == {"layers"}:
    subtree = subtree["layers"]
  if isinstance(subtree, Mapping) and subtree and all(_is_index(index) for index in subtree):
    return subtree
  return None


def _translate(tree, to_legacy, nested=True):
  if isinstance(tree, Mapping):
    result = {}
    for key, value in tree.items():
      if key == "decoder" and isinstance(value, Mapping):
        decoder = {}
        layers = {}
        for name, subtree in value.items():
          indexed = _indexed_layers(subtree) if to_legacy and name == "layers" else None
          if indexed is not None:
            decoder.update({f"layers_{index}": _translate(layer, to_legacy) for index, layer in indexed.items()})
          elif not to_legacy and _is_legacy_layer_name(name):
            layers[int(name[7:])] = _translate(subtree, to_legacy, nested)
          else:
            decoder[name] = _translate(subtree, to_legacy, nested)
        if layers:
          decoder["layers"] = {"layers": layers} if nested else layers
        result[key] = decoder
      else:
        result[key] = _translate(value, to_legacy, nested)
    return result
  if isinstance(tree, list):
    return [_translate(value, to_legacy, nested) for value in tree]
  if isinstance(tree, tuple):
    values = [_translate(value, to_legacy, nested) for value in tree]
    return type(tree)(*values) if hasattr(tree, "_fields") else tuple(values)
  return tree


def to_legacy_layer_names(tree):
  return _translate(tree, to_legacy=True)


def _is_legacy_layer_name(name):
  return isinstance(name, str) and name.startswith("layers_") and name[7:].isdecimal()


def match_checkpoint_layer_names(target, checkpoint_tree):
  decoder = checkpoint_tree.get("decoder", None) if hasattr(checkpoint_tree, "get") else None
  if hasattr(decoder, "keys") and any(_is_legacy_layer_name(name) for name in decoder.keys()):
    return to_legacy_layer_names(target)
  return target


def from_legacy_layer_names(tree, model_state=None):
  nested = True
  if model_state is not None:
    model = model_state.get("model", model_state)
    decoder = model.get("decoder", {})
    layers = decoder.get("layers", {})
    indexed = _indexed_layers(layers)
    if indexed is None:
      return tree
    nested = indexed is not layers
  return _translate(tree, to_legacy=False, nested=nested)
