"""CPU Hugging Face checkpoint import and its worker settings."""

from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from pydantic import BaseModel, ConfigDict

from lilax._backend.checkpoint_conversion.cpu_bootstrap import convert_on_cpu
from lilax.configuration.loading import AutoConfig
from lilax.configuration.model_architecture import ArchitectureConfig
from lilax.configuration.model_mapping import model_config_from_source, project_model_config
from lilax.modeling.hub import get_model_capabilities, resolve_model_snapshot


class CheckpointConversionConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  conversion_output_dir: str | Path | None = None
  lazy_load_tensors: bool = True
  eager_load_method: str = "safetensors"
  save_dtype: str = "bfloat16"
  simulated_cpu_devices_count: int = 16


def import_huggingface_checkpoint(source, destination, options, *, fused_qkv, fused_mlp):
  snapshot = resolve_model_snapshot(source)
  public_config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
  if not get_model_capabilities(public_config.model_name).hf_import:
    raise ValueError(f"Hugging Face import is not supported for {public_config.model_name!r}")
  architecture = public_config.architecture_config or ArchitectureConfig()
  public_config.architecture_config = architecture.model_copy(update={
      name: value for name, value in (("fused_qkv", fused_qkv), ("fused_mlp", fused_mlp)) if value is not None
  })
  source_config = project_model_config(public_config)
  destination.parent.mkdir(parents=True, exist_ok=True)
  staging_parent = Path(options.conversion_output_dir).expanduser() if options.conversion_output_dir else destination.parent
  staging_parent.mkdir(parents=True, exist_ok=True)
  with TemporaryDirectory(prefix="lilax-conversion-", dir=staging_parent) as staging:
    checkpoint = convert_on_cpu(
        source_config, snapshot, staging, lazy_load_tensors=options.lazy_load_tensors,
        eager_load_method=options.eager_load_method, save_dtype=options.save_dtype,
        simulated_cpu_devices_count=options.simulated_cpu_devices_count, conversion_mode="explicit",
    )
    shutil.copytree(checkpoint, destination)
  saved_source = source_config.model_copy(update={"weight_dtype": options.save_dtype})
  return saved_source, model_config_from_source(saved_source, metadata_config=public_config)
