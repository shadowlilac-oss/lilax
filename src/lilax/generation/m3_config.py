"""Inert M3 configuration and explicit runtime processing."""

from lilax._backend.m3.configs.config import (
    BaseConfig, CheckpointConfig, DataConfig, DeviceConfig, DiffusionConfig, OptimizerConfig, TrainerConfig,
    TransformerConfig, load_yaml_config, process_config,
)
from lilax._backend.m3.configs.glm5_3_flash_config import (
    Glm5Config, Glm5Sharding, MTP_LAYER_INDEX, validate_no_packing,
)
