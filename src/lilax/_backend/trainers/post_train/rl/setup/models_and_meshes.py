"""Create RL reference and actor models with their trainer and rollout meshes."""

from flax import nnx
from jax.sharding import Mesh

from lilax._backend.integration.tunix.tunix_adapter import TunixMaxTextAdapter
from lilax._backend.distributed import device_mesh
from lilax._backend.observability import max_logging
from lilax._backend.models.creation.pretrained import from_pretrained


def create_models_and_meshes(trainer_config, sampler_config, trainer_devices, sampler_devices, tokenizer_pad_id=None):
  """Create reference and actor models and their respective meshes."""
  max_logging.log("Creating reference model and also meshes for reference and rollout")
  reference_model, reference_mesh = from_pretrained(
      trainer_config, devices=trainer_devices, wrap_with_tunix_adapter=True, tokenizer_pad_id=tokenizer_pad_id
  )
  devices_array = device_mesh.create_device_mesh(sampler_config, sampler_devices)
  rollout_mesh = Mesh(devices_array, sampler_config.mesh_axes)

  if trainer_config.load_checkpoint_only_once:
    max_logging.log("Creating policy model by copying reference model instead of restoring from checkpoint again.")
    with reference_mesh:
      actor_base_model = nnx.clone(reference_model.base)
      use_no_op_mappings = "maxtext_config" in trainer_config.vllm_additional_config
      actor_model = TunixMaxTextAdapter(
          base_model=actor_base_model, use_no_op_mappings=use_no_op_mappings, pad_id=tokenizer_pad_id
      )
      actor_model.config = None
    actor_mesh = reference_mesh
  else:
    max_logging.log("Creating policy model with same config as reference model on trainer mesh")
    actor_model, actor_mesh = from_pretrained(
        trainer_config, devices=trainer_devices, wrap_with_tunix_adapter=True, tokenizer_pad_id=tokenizer_pad_id
    )

  return reference_model, reference_mesh, actor_model, actor_mesh, rollout_mesh
