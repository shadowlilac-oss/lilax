"""Modern MaxText Models configuration and explicit runtime processing."""

import datetime
import os
import re
from typing import Any, Optional, Sequence

import jax
import jax.numpy as jnp
from lilax._backend.observability import max_logging
from pydantic import BaseModel, ConfigDict, Field
import yaml


class DeviceConfig(BaseModel):
  model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
  hardware: str = "tpu"
  skip_jax_distributed_system: bool = False
  jax_cache_dir: str = ""
  mesh_axes: list[str] = Field(default_factory=lambda: ["data", "fsdp", "context", "tensor"])
  logical_axis_rules: tuple = ()
  vae_logical_axis_rules: tuple = ()
  data_sharding: tuple = ()
  dcn_data_parallelism: int = 1
  dcn_fsdp_parallelism: int = 1
  dcn_context_parallelism: int = -1
  dcn_tensor_parallelism: int = 1
  ici_data_parallelism: int = 1
  ici_fsdp_parallelism: int = 1
  ici_context_parallelism: int = -1
  ici_tensor_parallelism: int = 1
  allow_split_physical_axes: bool = False
  compile_topology_num_slices: int = -1
  num_slices: int = 1
  quantization_local_shard_count: int = 1


class CheckpointConfig(BaseModel):
  model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
  run_name: str = ""
  base_output_directory: str = ""
  checkpoint_directory: str = ""
  output_dir: str = ""
  checkpoint_every: int = -1
  checkpoint_dir: str = ""
  tensorboard_dir: str = ""
  metrics_dir: str = ""
  save_final_checkpoint: bool = False
  save_config_to_gcs: bool = False
  enable_single_replica_ckpt_restoring: bool = False
  aot_cache_dir: str = ""
  unet_checkpoint: str = ""
  revision: str = ""
  lightning_from_pt: bool = True
  lightning_repo: str = ""
  lightning_ckpt: str = ""


class OptimizerConfig(BaseModel):
  model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
  adam_b1: float = 0.9
  adam_b2: float = 0.999
  adam_eps: float = 1e-8
  adam_weight_decay: float = 0.0
  opt_enable_grad_clipping: bool = False
  max_grad_value: float = 1.0
  opt_enable_grad_global_norm_clipping: bool = False
  max_grad_norm: float = 1.0
  warmup_steps_fraction: float = 0.1
  learning_rate_schedule_steps: int = -1
  save_optimizer: bool = False


class TrainerConfig(BaseModel):
  model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
  learning_rate: float = 1e-5
  scale_lr: bool = False
  max_train_samples: int = -1
  max_train_steps: int = 1500
  num_train_epochs: int = 1
  seed: int = 0
  per_device_batch_size: float = 1.0
  global_batch_size: int = 0
  disable_training_weights: bool = False
  train_text_encoder: bool = False
  text_encoder_learning_rate: float = 4.25e-6
  snr_gamma: float = -1.0
  remat_policy: str = "NONE"
  names_which_can_be_saved: list[str] = Field(default_factory=list)
  names_which_can_be_offloaded: list[str] = Field(default_factory=list)
  enable_tile_search: bool = False
  tile_search_mode: str = "smart"
  tile_search_iters: int = 10
  tile_search_out: str = ""
  optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
  eval_every: int = -1
  eval_data_dir: str = ""
  enable_generate_video_for_eval: bool = False
  eval_max_number_of_samples_in_bucket: int = 60
  enable_eval_timesteps: bool = False
  timesteps_list: list[int] = Field(default_factory=lambda: [125, 250, 375, 500, 625, 750, 875])
  num_eval_samples: int = 420
  metrics_file: str = ""
  write_metrics: bool = True
  timing_metrics_file: str = ""
  write_timing_metrics: bool = True
  gcs_metrics: bool = False
  log_period: int = 100
  enable_mllog: bool = False
  enable_ssim: bool = False
  enable_ml_diagnostics: bool = False


class DataConfig(BaseModel):
  model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
  dataset_name: str = ""
  dataset_path: str = ""
  train_data_dir: str = ""
  eval_data_dir: str = ""
  per_device_batch_size: float = 1.0
  max_sequence_length: int = 512


class TransformerConfig(BaseModel):
  model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
  dim: Optional[int] = None
  base_emb_dim: Optional[int] = None
  num_layers: Optional[int] = None
  base_num_decoder_layers: Optional[int] = None
  num_heads: Optional[int] = None
  base_num_query_heads: Optional[int] = None
  num_kv_heads: Optional[int] = None
  base_num_kv_heads: Optional[int] = None
  head_dim: Optional[int] = None
  mlp_dim: Optional[int] = None
  base_mlp_dim: Optional[int] = None
  weights_dtype: Any = "bfloat16"
  activations_dtype: Any = "bfloat16"
  attention: str = "flash"
  precision: str = "DEFAULT"
  scan_layers: bool = False
  dropout: float = 0.0
  split_head_dim: bool = True
  norm_num_groups: int = 32
  use_base2_exp: bool = True


class BaseConfig(BaseModel):
  model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
  device: DeviceConfig = Field(default_factory=DeviceConfig)
  checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
  trainer: TrainerConfig = Field(default_factory=TrainerConfig)
  data: DataConfig = Field(default_factory=DataConfig)
  transformer: Optional[TransformerConfig] = None

  def _get_subconfigs(self) -> list[BaseModel]:
    subs = []
    for attr in ("device", "checkpoint", "trainer", "data", "transformer", "autoencoder", "scheduler", "text_encoder", "pipeline"):
      sub = self.__dict__.get(attr) if attr in type(self).model_fields else None
      if isinstance(sub, BaseModel):
        subs.append(sub)
    return subs

  def __getattr__(self, name: str) -> Any:
    if self.__pydantic_extra__ and name in self.__pydantic_extra__:
      return self.__pydantic_extra__[name]
    for sub in self._get_subconfigs():
      if hasattr(sub, name):
        return getattr(sub, name)
      if sub.__pydantic_extra__ and name in sub.__pydantic_extra__:
        return sub.__pydantic_extra__[name]
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

  def __getitem__(self, key: str) -> Any:
    try:
      return getattr(self, key)
    except AttributeError as exc:
      raise KeyError(key) from exc

  def __setitem__(self, key: str, value: Any) -> None:
    setattr(self, key, value)

  def __setattr__(self, name: str, value: Any) -> None:
    if name in type(self).model_fields:
      super().__setattr__(name, value)
      return
    updated = False
    if self.__pydantic_extra__ and name in self.__pydantic_extra__:
      self.__pydantic_extra__[name] = value
      updated = True
    for sub in self._get_subconfigs():
      if hasattr(sub, name) or (sub.__pydantic_extra__ and name in sub.__pydantic_extra__):
        setattr(sub, name, value)
        updated = True
    if updated:
      return
    if self.__pydantic_extra__ is None:
      object.__setattr__(self, "__pydantic_extra__", {})
    self.__pydantic_extra__[name] = value

  def get(self, key: str, default: Any = None) -> Any:
    try:
      return getattr(self, key)
    except AttributeError:
      return default

  def __contains__(self, key: str) -> bool:
    try:
      getattr(self, key)
      return True
    except AttributeError:
      return False

  def keys(self):
    return self.get_keys().keys()

  def values(self):
    return self.get_keys().values()

  def items(self):
    return self.get_keys().items()

  def __iter__(self):
    return iter(self.get_keys())

  def get_keys(self) -> dict[str, Any]:
    result = {}
    for sub in self._get_subconfigs():
      result.update(sub.model_dump())
      if sub.__pydantic_extra__:
        result.update(sub.__pydantic_extra__)
    result.update(self.model_dump())
    if self.__pydantic_extra__:
      result.update(self.__pydantic_extra__)
    return result

  @classmethod
  def from_dict(cls, raw: dict[str, Any]) -> "BaseConfig":
    values = dict(raw)
    for name, config_type in (("device", DeviceConfig), ("checkpoint", CheckpointConfig), ("trainer", TrainerConfig),
                              ("data", DataConfig), ("transformer", TransformerConfig)):
      values[name] = config_type.model_validate(raw.get(name) or raw)
    return cls(**values)


class AutoencoderConfig(BaseModel):
  model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
  vae_type: str = "autoencoder_kl"
  vae_weights_dtype: Any = "float32"
  vae_dtype: Any = "float32"
  replicate_vae: bool = False
  vae_decode_chunk: int = 1
  vae_encode_chunk: int = 4
  vae_spatial: int = -1
  vae_logical_axis_rules: tuple = ()


class SchedulerConfig(BaseModel):
  model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
  scheduler_type: str = "unipc_multistep"
  scheduler_dtype: Any = "float32"
  use_experimental_scheduler: bool = True
  flow_shift: float = 5.0
  guidance_scale: float = 5.0
  guidance_rescale: float = 0.0
  num_inference_steps: int = 50
  diffusion_scheduler_config: dict = Field(default_factory=dict)
  timestep_bias: dict = Field(default_factory=dict)


class TextEncoderConfig(BaseModel):
  model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
  text_encoder_type: str = "umt5"
  text_encoder_dtype: Any = "float32"
  max_sequence_length: int = 512
  text_encoder_logical_axis_rules: tuple = ()


class DiffusionTransformerConfig(TransformerConfig):
  model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
  patch_size: tuple[int, int, int] = (1, 2, 2)
  in_channels: int = 16
  out_channels: int = 16
  text_dim: int = 4096
  freq_dim: int = 256
  qk_norm: str = "rms_norm"
  cross_attn_norm: bool = True
  flash_block_sizes: Optional[Any] = None
  flash_min_seq_length: int = 4096
  ulysses_shards: int = -1
  ulysses_attention_chunks: int = 1
  remat_policy: str = "NONE"
  names_which_can_be_saved: list[str] = Field(default_factory=list)
  names_which_can_be_offloaded: list[str] = Field(default_factory=list)
  enable_jax_named_scopes: bool = False


class DiffusionPipelineConfig(BaseModel):
  model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
  pretrained_model_name_or_path: str = ""
  model_name: str = ""
  model_type: str = "T2V"
  prompt: str = ""
  prompt_2: str = ""
  negative_prompt: str = ""
  prompt_file: str = ""
  do_classifier_free_guidance: bool = True
  height: int = 720
  width: int = 1280
  num_frames: int = 81
  fps: int = 16
  seed: int = 0
  use_cfg_cache: bool = False
  use_batched_text_encoder: bool = False
  use_kv_cache: bool = False
  use_magcache: bool = False
  magcache_thresh: float = 0.12
  magcache_K: int = 2
  retention_ratio: float = 0.2
  mag_ratios_base: list = Field(default_factory=list)
  enable_lora: bool = False
  lora_config: dict = Field(default_factory=dict)
  controlnet_model_name_or_path: str = ""
  controlnet_from_pt: bool = True
  controlnet_conditioning_scale: float = 0.5
  controlnet_image: str = ""
  quantization: str = ""
  use_qwix_quantization: bool = False
  qwix_module_path: str = ""
  weight_quantization_calibration_method: str = "absmax"
  act_quantization_calibration_method: str = "absmax"
  bwd_quantization_calibration_method: str = "absmax"


class DiffusionConfig(BaseConfig):
  model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
  autoencoder: AutoencoderConfig = Field(default_factory=AutoencoderConfig)
  scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
  text_encoder: TextEncoderConfig = Field(default_factory=TextEncoderConfig)
  transformer: DiffusionTransformerConfig = Field(default_factory=DiffusionTransformerConfig)
  pipeline: DiffusionPipelineConfig = Field(default_factory=DiffusionPipelineConfig)

  @classmethod
  def from_dict(cls, raw: dict[str, Any]) -> "DiffusionConfig":
    values = dict(raw)
    for name, config_type in (("device", DeviceConfig), ("checkpoint", CheckpointConfig), ("trainer", TrainerConfig),
                              ("data", DataConfig), ("autoencoder", AutoencoderConfig), ("scheduler", SchedulerConfig),
                              ("text_encoder", TextEncoderConfig), ("transformer", DiffusionTransformerConfig),
                              ("pipeline", DiffusionPipelineConfig)):
      values[name] = config_type.model_validate(raw.get(name) or raw)
    return cls(**values)


def _lists_to_tuples(value: Any) -> Any:
  if isinstance(value, (list, tuple)):
    return tuple(_lists_to_tuples(item) for item in value)
  return value


def get_global_batch_size(per_device_batch_size: float) -> int:
  num_devices = len(jax.devices())
  return num_devices if per_device_batch_size < 1 else int(num_devices * per_device_batch_size)


def maybe_initialize_jax_distributed_system(raw_keys: dict[str, Any]) -> None:
  if raw_keys.get("skip_jax_distributed_system", False):
    max_logging.log("Skipping jax distributed system due to skip_jax_distributed_system=True flag.")
    return
  if jax.distributed.is_initialized():
    max_logging.log("Jax distributed system is already initialized.")
    return
  if raw_keys.get("enable_single_controller", False):
    max_logging.log("Skipping jax distributed system since its not needed for single controller.")
    return
  if raw_keys.get("inference_benchmark_test", False) or raw_keys.get("compile_topology", ""):
    max_logging.log("Skipping jax distributed system initialization.")
    return
  try:
    jax.distributed.initialize(initialization_timeout=raw_keys.get("jax_distributed_initialization_timeout", 300))
    max_logging.log("Jax distributed system initialized!")
  except Exception as exc:  # pylint: disable=broad-exception-caught
    max_logging.log(f"Warning: jax.distributed.initialize() skipped or failed: {exc}")


def _resolve_interpolations(values: dict[str, Any], root: dict[str, Any] | None = None) -> dict[str, Any]:
  if root is None:
    root = values
  pattern = re.compile(r"\$\{([a-zA-Z0-9_\.]+)\}")

  def lookup(path: str) -> Any:
    current = root
    for part in path.split("."):
      if isinstance(current, dict) and part in current:
        current = current[part]
      else:
        return None
    return current

  for key, value in list(values.items()):
    if isinstance(value, str) and "${" in value:
      match = pattern.fullmatch(value.strip())
      if match:
        resolved = lookup(match.group(1))
        if resolved is not None:
          values[key] = resolved
      else:
        values[key] = pattern.sub(lambda match: str(lookup(match.group(1)) or match.group(0)), value)
    elif isinstance(value, dict):
      _resolve_interpolations(value, root)
  return values


def _clean_default_entry(entry: Any) -> tuple[str, Optional[str]]:
  if isinstance(entry, str):
    return entry.strip().removeprefix("/").split("@")[0], None
  if isinstance(entry, dict):
    for key, value in entry.items():
      return key.strip().removeprefix("/").split("@")[0], str(value).strip()
  return "", None


def _find_yaml_file(rel_path: str, search_paths: Sequence[str]) -> Optional[str]:
  for search_path in search_paths:
    for extension in ("", ".yml", ".yaml"):
      candidate = os.path.join(search_path, rel_path + extension)
      if os.path.isfile(candidate):
        return candidate
  return None


def load_yaml_config(file_path: str, search_paths: Sequence[str] | None = None) -> dict[str, Any]:
  """Load a YAML configuration and merge its modular defaults."""
  file_path = os.path.abspath(os.path.expanduser(file_path))
  if not os.path.isfile(file_path):
    raise FileNotFoundError(f"Configuration file not found: {file_path}")
  base_dir = os.path.dirname(file_path)
  if search_paths is None:
    search_paths = [base_dir, os.path.abspath(os.path.join(base_dir, "..")),
                    os.path.abspath(os.path.join(base_dir, "..", "..")),
                    os.path.abspath(os.path.join(base_dir, "models")),
                    os.path.abspath(os.path.join(base_dir, "..", "models"))]
  with open(file_path, "r", encoding="utf-8") as file:
    data = yaml.safe_load(file) or {}
  merged = {}
  for entry in data.get("defaults", []):
    if entry == "_self_":
      continue
    prefix, name = _clean_default_entry(entry)
    if not prefix:
      continue
    target = f"{prefix}/{name}" if name else prefix
    found = _find_yaml_file(target, search_paths)
    if not found:
      found = _find_yaml_file(prefix, search_paths)
    if found:
      merged.update(load_yaml_config(found, search_paths))
    else:
      max_logging.log(f"Warning: could not resolve default config entry: {target}")
  for key, value in data.items():
    if key not in ("defaults", "hydra"):
      merged[key] = value
  return merged


def process_config(raw: dict[str, Any] | BaseConfig, overrides: Optional[dict[str, Any]] = None, **kwargs) -> BaseConfig:
  """Resolve a configuration and explicitly initialize its JAX runtime."""
  if isinstance(raw, BaseConfig):
    raw_dict = raw.get_keys()
    for key in raw_dict.keys() - type(raw).model_fields.keys():
      raw_dict[key] = getattr(raw, key)
    updated_fields = set()
  else:
    raw_dict = dict(raw)
    updated_fields = set(raw_dict)
  if overrides:
    raw_dict.update(overrides)
    updated_fields.update(overrides)
  raw_dict.update(kwargs)
  updated_fields.update(kwargs)
  updated_fields.update(key for key, value in raw_dict.items() if isinstance(value, str) and "${" in value)
  _resolve_interpolations(raw_dict)
  for field in ("weights_dtype", "activations_dtype", "vae_weights_dtype", "vae_dtype", "scheduler_dtype", "text_encoder_dtype"):
    if field in raw_dict and isinstance(raw_dict[field], str):
      try:
        raw_dict[field] = jnp.dtype(raw_dict[field])
        updated_fields.add(field)
      except Exception:  # pylint: disable=broad-exception-caught
        pass
  run_name = raw_dict.get("run_name", "") or os.environ.get("JOBSET_NAME", "")
  run_name = run_name or datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
  raw_dict["run_name"] = run_name
  updated_fields.add("run_name")
  output = raw_dict.get("base_output_directory", "") or raw_dict.get("checkpoint_directory", "") or raw_dict.get("output_dir", "")
  if output:
    raw_dict["base_output_directory"] = output
    updated_fields.add("base_output_directory")
  for field, directory in (("tensorboard_dir", "tensorboard"), ("checkpoint_dir", "checkpoints"), ("metrics_dir", "metrics")):
    if not raw_dict.get(field):
      raw_dict[field] = os.path.join(output, run_name, directory, "") if output else ""
      updated_fields.add(field)
  for field in ("logical_axis_rules", "vae_logical_axis_rules", "text_encoder_logical_axis_rules", "data_sharding"):
    if field in raw_dict and raw_dict[field] is not None:
      raw_dict[field] = _lists_to_tuples(raw_dict[field])
      updated_fields.add(field)
  if not kwargs.get("unittest", False):
    maybe_initialize_jax_distributed_system(raw_dict)
  batch_size = raw_dict.get("per_device_batch_size", 1.0)
  raw_dict["total_train_batch_size"] = get_global_batch_size(batch_size)
  num_devices = len(jax.devices())
  raw_dict["global_batch_size_to_load"] = num_devices if batch_size < 1 else int(num_devices * batch_size)
  raw_dict["global_batch_size_to_train_on"] = int(num_devices * batch_size)
  updated_fields.update(("total_train_batch_size", "global_batch_size_to_load", "global_batch_size_to_train_on"))
  if raw_dict.get("jax_cache_dir"):
    jax.config.update("jax_compilation_cache_dir", raw_dict["jax_cache_dir"])
  is_diffusion = any(key in raw_dict for key in ("pretrained_model_name_or_path", "model_type", "vae_dtype", "vae_decode_chunk", "flow_shift", "num_frames", "num_inference_steps"))
  config = DiffusionConfig.from_dict(raw_dict) if is_diffusion else BaseConfig.from_dict(raw_dict)
  for field in updated_fields - type(config).model_fields.keys():
    if field in raw_dict:
      setattr(config, field, raw_dict[field])
  for key in sorted(raw_dict):
    max_logging.log(f"Config param {key}: {raw_dict[key]}")
  return config
