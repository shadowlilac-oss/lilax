"""Create sharded NNX models and restore their pretrained parameters."""

from flax import nnx

from lilax._backend.common.common_types import MODEL_MODE_TRAIN
from lilax._backend.common.checkpoints.model_path import model_checkpoint_path
from lilax._backend.configs import pyconfig
from lilax._backend.experimental.lineage import lineage_adapter
from lilax._backend.integration.tunix.tunix_adapter import TunixMaxTextAdapter
from lilax._backend.utils import maxtext_utils_nnx
from lilax._backend.models.creation.checkpoint_loading import lineage_checkpoint
from lilax._backend.models.creation.checkpoint_loading.converted_source import use_converted_checkpoint_if_possible
from lilax._backend.models.creation.checkpoint_loading.orbax_restore import restore_orbax_params
from lilax._backend.models.creation.nnx_abstract_model import create_nnx_abstract_model
from lilax._backend.models.creation.scan_layers_sync import verify_and_sync_scan_layers


def from_pretrained(
    config,
    mesh=None,
    devices=None,
    model_mode=MODEL_MODE_TRAIN,
    rng_key=None,
    wrap_with_tunix_adapter=False,
    tokenizer_pad_id=None,
    *,
    quant_mode_str: str = "train",
    return_effective_config=False,
):
  """Create an NNX model with sharded parameters, optionally restoring a checkpoint."""
  original_mesh = mesh
  config = use_converted_checkpoint_if_possible(config)
  if config.load_parameters_path:
    checkpoint_path = str(model_checkpoint_path(config.load_parameters_path))
    if checkpoint_path != config.load_parameters_path:
      pydantic_config = getattr(config, "_pydantic_config", config)
      config = pyconfig.HyperParameters(pydantic_config.model_copy(update={"load_parameters_path": checkpoint_path}))
  config = verify_and_sync_scan_layers(config)
  create_model_fn, abstract_model = create_nnx_abstract_model(
      config, mesh, devices, model_mode, rng_key, quant_mode_str=quant_mode_str
  )
  _, abstract_state = nnx.split(abstract_model)
  specs = nnx.get_partition_spec(abstract_state)
  model = maxtext_utils_nnx.create_nnx_sharded_model(abstract_model, create_model_fn, mesh=mesh)
  sharded_state = nnx.state(model)
  if mesh is None:
    mesh = model.mesh

  with mesh:
    maxtext_lineage_checkpoint = False
    if config.load_parameters_path and lineage_adapter.is_native(config):
      maxtext_lineage_checkpoint = lineage_checkpoint.is_maxtext_layout(config)

    if maxtext_lineage_checkpoint:
      lineage_checkpoint.restore_maxtext_layout(model, sharded_state, config, mesh)
    elif config.load_parameters_path:
      restore_orbax_params(model, sharded_state, specs, config, mesh)

    if wrap_with_tunix_adapter:
      with mesh:
        use_no_op_mappings = "maxtext_config" in config.vllm_additional_config
        model = TunixMaxTextAdapter(base_model=model, use_no_op_mappings=use_no_op_mappings, pad_id=tokenizer_pad_id)
        model.config = None
    result = model if original_mesh else (model, mesh)
    return (result, config) if return_effective_config else result
