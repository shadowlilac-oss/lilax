"""Native Wan models, schedulers, and configuration."""

from lilax._backend.m3.configs.config import load_yaml_config, process_config
from lilax._backend.m3.configs.diffusers_config import DiffusionConfig
from lilax._backend.m3.models.diffuser_schedulers.scheduling_flow_match import (
    FlaxFlowMatchScheduler,
    FlaxFlowMatchSchedulerOutput,
    FlowMatchSchedulerState,
)
from lilax._backend.m3.models.diffuser_schedulers.scheduling_unipc_multistep import (
    FlaxUniPCMultistepScheduler,
    FlaxUniPCMultistepSchedulerOutput,
    UniPCMultistepSchedulerState,
)
from lilax._backend.m3.models.wan import (
    AutoencoderKLWan,
    AutoencoderKLWanCache,
    UMT5EncoderModel,
    VAE_MESH_AXES,
    WAN_MESH_AXES,
    WanModel,
    WanSharding,
)
from lilax._backend.m3.models.wan.utils_wan import (
    WAN21_CONFIG_PATH,
    create_wan_mesh,
    get_flash_block_sizes,
    get_precision,
    yml_rules,
)


def load_wan_config() -> dict:
  """Load the bundled Wan 2.1 configuration without initializing the runtime."""
  config = load_yaml_config(WAN21_CONFIG_PATH)
  if config.get("transformer_class") == "maxtext.m3.models.wan.transformer_wan.WanModel":
    config["transformer_class"] = f"{WanModel.__module__}.{WanModel.__name__}"
  if config.get("vae_class") == "maxtext.m3.models.wan.autoencoder_kl_wan.AutoencoderKLWan":
    config["vae_class"] = f"{AutoencoderKLWan.__module__}.{AutoencoderKLWan.__name__}"
  return config
