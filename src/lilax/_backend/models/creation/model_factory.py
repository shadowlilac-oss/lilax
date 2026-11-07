"""Construct Linen or NNX MaxText models from a configuration and mesh."""

from collections.abc import Sequence
from typing import Callable, overload

from flax import nnx
import jax
from jax.sharding import Mesh

from lilax._backend.common.common_types import MODEL_MODE_TRAIN
from lilax._backend.configs import pyconfig
from lilax._backend.experimental.lineage import lineage_adapter
from lilax._backend.layers import nnx_wrappers
from lilax._backend.layers.quantization.configure import configure_quantization
from lilax._backend.layers.quantization.qwix_model import maybe_quantize_model
from lilax._backend.models import models
from lilax._backend.distributed import device_mesh
from lilax._backend.utils import maxtext_utils_nnx


@overload
def from_config(
    config: pyconfig.HyperParameters,
    devices: Sequence[jax.Device] | None = None,
    mesh: Mesh | None = None,
    *,
    model_mode: str = MODEL_MODE_TRAIN,
    rngs: None = None,
    quant_mode_str: str = "train",
) -> nnx_wrappers.ToLinen: ...


@overload
def from_config(
    config: pyconfig.HyperParameters,
    devices: Sequence[jax.Device] | None = None,
    mesh: Mesh | None = None,
    *,
    model_mode: str = MODEL_MODE_TRAIN,
    rngs: nnx.Rngs,
    quant_mode_str: str = "train",
) -> models.Transformer: ...


def from_config(
    config: pyconfig.HyperParameters,
    devices: Sequence[jax.Device] | None = None,
    mesh: Mesh | None = None,
    *,
    model_mode: str = MODEL_MODE_TRAIN,
    rngs: nnx.Rngs | None = None,
    quant_mode_str: str = "train",
) -> nnx_wrappers.ToLinen | models.Transformer:
  """Construct a model using the supplied configuration, mesh, and RNGs."""
  if mesh is None:
    mesh = device_mesh.get_mesh_from_config(config, devices)
  return create_model(config, mesh, model_mode=model_mode, rngs=rngs, quant_mode_str=quant_mode_str)


def get_transformer_model(config, mesh, quant, model_mode: str = MODEL_MODE_TRAIN, rngs: nnx.Rngs | None = None):
  """Returns the transformer model based on the configuration."""
  if rngs is not None and lineage_adapter.is_native(config):
    return lineage_adapter.LineageTransformer(config, mesh, quant=quant, rngs=rngs, model_mode=model_mode)
  if rngs is not None:
    return models.Transformer(config, mesh, quant=quant, rngs=rngs, model_mode=model_mode)
  return models.transformer_as_linen(config, mesh, quant=quant, model_mode=model_mode)


def create_model(
    config, mesh, model_mode: str = MODEL_MODE_TRAIN, rngs: nnx.Rngs | None = None, *, quant_mode_str: str = "train"
):
  """Instantiates and returns the model object, sharded across the mesh."""
  if config.use_m3_model:
    from lilax._backend.m3 import models as m3_models
    from lilax._backend.m3.infra import scan as m3_scan

    model = m3_models.create_model(config, mesh, rngs=rngs, quant_mode_str=quant_mode_str)
    return m3_scan.maybe_apply_scan(model, config)

  quant = configure_quantization(config, quant_mode_str=quant_mode_str)
  model = get_transformer_model(config, mesh, quant, model_mode=model_mode, rngs=rngs)
  return maybe_quantize_model(model, config)


def get_nnx_create_model_fn(
    config, mesh=None, devices=None, model_mode=MODEL_MODE_TRAIN, rng_key=None, *, quant_mode_str: str = "train"
) -> Callable:
  def _create_model():
    rngs = maxtext_utils_nnx.create_nnx_rngs(config, model_mode=model_mode, rng_key=rng_key)
    return from_config(config, devices, mesh, rngs=rngs, model_mode=model_mode, quant_mode_str=quant_mode_str)

  return _create_model
