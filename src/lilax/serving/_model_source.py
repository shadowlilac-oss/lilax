"""Bind serving settings to model weights without reusing a training mesh."""

from pathlib import Path
from tempfile import TemporaryDirectory

from lilax._backend.configs.pyconfig import HyperParameters
from lilax._backend.configs.runtime import resolve_runtime_config
from lilax.configuration.loading import AutoConfig
from lilax.configuration.model_mapping import project_model_config
from lilax.modeling.sources import resolve_checkpoint_path
from lilax.runtime import Runtime


def serving_model_source(model, *, overrides=None):
  overrides = dict(overrides or {})
  resource = None
  source = model.runtime.public_config
  if model.is_materialized:
    root = Path.home() / "tmp" / "lilax" / "serving"
    root.mkdir(parents=True, exist_ok=True)
    resource = TemporaryDirectory(dir=root)
    checkpoint_path = str(Path(resource.name) / "model")
    try:
      model.save_pretrained(checkpoint_path)
      source = project_model_config(AutoConfig.from_pretrained(checkpoint_path, local_files_only=True))
    except BaseException:
      resource.cleanup()
      raise
  else:
    if model._checkpoint is None:
      raise ValueError("Serving requires pretrained or materialized model weights.")
    runtime = Runtime(model.runtime.public_config)
    runtime.config = resolve_runtime_config(runtime.public_config, overrides=runtime._overrides | overrides)
    checkpoint_path = resolve_checkpoint_path(model._checkpoint, runtime)
  try:
    config = resolve_runtime_config(source, overrides=model.runtime._overrides | overrides)
    config = HyperParameters(config._pydantic_config.model_copy(update={
        "load_parameters_path": checkpoint_path, "load_full_state_path": "", "convert_checkpoint_if_possible": False,
    }))
  except BaseException:
    if resource is not None:
      resource.cleanup()
    raise
  return config, resource


def serving_overrides(options):
  values = options.model_dump(exclude={"parallelism_config", "cache_config"}, exclude_none=True)
  parallelism = options.parallelism_config
  if parallelism is not None:
    for domain in (parallelism.mesh, parallelism.sharding):
      if domain is not None:
        values.update(domain.model_dump(exclude_unset=True))
  cache = getattr(options, "cache_config", None)
  if cache is not None:
    values.update(cache.model_dump())
  if "hf_overrides" in values:
    values["vllm_hf_overrides"] = values.pop("hf_overrides")
  return values
