"""Decoder layer containers for pipeline stages."""

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from lilax._backend.common.common_types import Config
from lilax._backend.layers.quantization.aqt import AqtQuantization as Quant


class NNXSequentialPipelineStage(nnx.Module):
  """Sequential unscanned series of decoder layers formatted for a single pipeline stage."""

  def __init__(
      self,
      layer_cls,
      num_layers: int,
      config: Config,
      mesh: Mesh,
      quant: Quant,
      model_mode: str,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.scan_layers = config.scan_layers
    self.num_layers = num_layers
    # Dynamically assign layers with explicit string names to ensure correct PyTree paths (layers_0)
    for i in range(num_layers):
      layer = layer_cls(config=config, mesh=mesh, quant=quant, model_mode=model_mode, rngs=rngs)
      setattr(self, f"layers_{i}", layer)

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      **kwargs,
  ):
    for i in range(self.num_layers):
      layer = getattr(self, f"layers_{i}")
      out = layer(
          inputs,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          **kwargs,
      )
      inputs = out[0] if isinstance(out, tuple) else out
    if self.scan_layers:
      return inputs, None
    return inputs


class NNXScannedPipelineStage(nnx.Module):
  """Scanned block of decoder layers formatted for a single pipeline stage."""

  def __init__(
      self,
      layer_cls,
      num_layers: int,
      config: Config,
      mesh: Mesh,
      quant: Quant,
      model_mode: str,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config

    def create_layer_fn(rng):
      return layer_cls(config=config, mesh=mesh, quant=quant, model_mode=model_mode, rngs=rng)

    forked_rngs = rngs.fork(split=num_layers)

    out_axes = nnx.StateAxes({nnx.Param: config.param_scan_axis, ...: 0})
    self.scanned_layers = nnx.vmap(
        create_layer_fn,
        in_axes=0,
        out_axes=out_axes,
        axis_name="layers_per_stage",
        transform_metadata={nnx.PARTITION_NAME: "layers_per_stage"},
    )(forked_rngs)

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      **kwargs,
  ):
    graphdef, params, state = nnx.split(self.scanned_layers, nnx.Param, ...)

    scan_axis = self.config.param_scan_axis
    if scan_axis != 0:
      params = jax.tree.map(lambda x: jnp.moveaxis(x, scan_axis, 0) if x.ndim > scan_axis else x, params)

    def layer_fn(carry, scanned_vars):
      current_params, current_state = scanned_vars
      layer = nnx.merge(graphdef, current_params, current_state)
      layer_out = layer(
          carry,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          **kwargs,
      )
      new_carry = layer_out[0] if isinstance(layer_out, tuple) else layer_out
      # Avoid returning and stacking read-only parameters inside the scan body.
      # This prevents huge unnecessary memory allocation.
      _, _, updated_state = nnx.split(layer, nnx.Param, ...)
      return new_carry, updated_state

    final_carry, scanned_state = jax.lax.scan(layer_fn, inputs, (params, state))

    if scan_axis != 0:
      scanned_params, scanned_other = scanned_state.split(nnx.Param, ...)
      if scanned_params:
        scanned_params = jax.tree.map(
            lambda x: jnp.moveaxis(x, 0, scan_axis) if x.ndim > scan_axis else x, scanned_params
        )
      scanned_state = nnx.State.merge(scanned_params, scanned_other)

    nnx.update(self.scanned_layers, scanned_state)

    if self.config.scan_layers:
      return final_carry, None
    return final_carry
