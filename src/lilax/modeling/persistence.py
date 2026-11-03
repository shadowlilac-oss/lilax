"""Native model checkpoint persistence and Hugging Face export."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from etils import epath
from flax import nnx
import orbax.checkpoint as ocp

from lilax._backend.checkpoint_conversion.to_huggingface import convert_to_huggingface
from lilax.configuration import QwixQATConfig
from lilax.modeling.hub import get_model_capabilities, resolve_model_name, resolve_model_snapshot


def checkpoint_qat_config(path):
  checkpoint_dir = epath.Path(path)
  metadata_path = checkpoint_dir / "maxtext_easy_config_metadata.json"
  config_path = checkpoint_dir / "config.json"
  if metadata_path.exists() and config_path.exists():
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if "qwix_qat_config" in metadata["explicit_keys"]:
      saved_qat_config = json.loads(config_path.read_text(encoding="utf-8"))["qwix_qat_config"]
      if saved_qat_config is not None:
        return QwixQATConfig.model_validate(saved_qat_config)
  return None


def restore_qat_checkpoint(model, runtime, config, checkpoint_path):
  state = nnx.state(model).filter(
      lambda path, var: not isinstance(var, (nnx.RngState, nnx.Cache, nnx.Intermediate, nnx.BatchStat))
  ).to_pure_dict()

  def wrap_value(node):
    if isinstance(node, dict):
      return {key: wrap_value(value) for key, value in node.items()}
    return {"value": node}

  def unwrap_value(node):
    if isinstance(node, dict):
      if len(node) == 1 and "value" in node and not isinstance(node["value"], dict):
        return node["value"]
      return {key: unwrap_value(value) for key, value in node.items()}
    return node

  target = wrap_value(state)
  checkpointer = ocp.PyTreeCheckpointer(
      use_ocdbt=config.checkpoint_storage_use_ocdbt, use_zarr3=config.checkpoint_storage_use_zarr3,
  )
  with runtime.mesh:
    restored = checkpointer.restore(
        checkpoint_path, item=target, restore_args=ocp.checkpoint_utils.construct_restore_args(target)
    )
    nnx.update(model, unwrap_value(restored))


def save_pretrained(
    model, save_directory, *, format="maxtext", hf_model_path=None, revision=None, token=None, cache_dir=None,
    local_files_only=False, override_model_architecture=False,
):
  model.materialize(runtime=model.runtime)
  if format == "huggingface":
    if model.existing_graph_qat_config is not None:
      raise NotImplementedError("Hugging Face export of Qwix QAT models is not supported.")
    capabilities = get_model_capabilities(model.runtime.config.model_name)
    if not capabilities.hf_export:
      raise ValueError(f"Hugging Face export is not supported for {capabilities.model_name!r}")
    if hf_model_path is None and capabilities.hf_repo_id is None:
      raise ValueError("Hugging Face export requires hf_model_path for this model.")
    snapshot = resolve_model_snapshot(
        hf_model_path or capabilities.hf_repo_id, revision=revision, token=token, cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    resolve_model_name(hf_model_path or capabilities.hf_repo_id, model_name=capabilities.model_name, snapshot=snapshot)
    temporary_root = Path.home() / "tmp" / "lilax"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=temporary_root) as directory:
      checkpoint_path = str(Path(directory) / "checkpoint")
      model.save_pretrained(checkpoint_path)
      lora = model.construction_config._pydantic_config.lora.model_copy(update={"lora_restore_path": ""})
      config = model.construction_config._pydantic_config.model_copy(
          update={"load_parameters_path": checkpoint_path, "load_full_state_path": "",
                  "base_output_directory": str(save_directory), "lora": lora}
      )
      return convert_to_huggingface(
          config, override_model_architecture=override_model_architecture, hf_model_path=str(snapshot)
      )
  if format != "maxtext":
    raise ValueError(f"Unsupported checkpoint format: {format!r}")
  public_config = model.runtime.public_config
  updates = public_config.get_source_values()
  updates["scan_layers"] = model.construction_config.scan_layers
  updates["qwix_qat_config"] = model.existing_graph_qat_config
  saved_config = public_config.model_copy(update=updates)
  config_json = saved_config.model_dump_json(exclude={"hf_access_token"}, indent=2)
  state = (
      nnx.state(model.native_model)
      .filter(lambda path, var: not isinstance(var, (nnx.RngState, nnx.Cache, nnx.Intermediate, nnx.BatchStat)))
      .to_pure_dict()
  )

  def wrap_value(node):
    if isinstance(node, dict):
      return {key: wrap_value(value) for key, value in node.items()}
    return {"value": node}

  checkpointer = ocp.PyTreeCheckpointer(
      use_ocdbt=model.runtime.config.checkpoint_storage_use_ocdbt,
      use_zarr3=model.runtime.config.checkpoint_storage_use_zarr3,
  )
  checkpointer.save(save_directory, wrap_value(state), force=True)
  (epath.Path(save_directory) / "config.json").write_text(config_json, encoding="utf-8")
  explicit_keys = public_config._explicit_keys | {"scan_layers"}
  if model.existing_graph_qat_config is not None:
    explicit_keys |= {"qwix_qat_config"}
  metadata = {"explicit_keys": sorted(explicit_keys - {"hf_access_token"})}
  (epath.Path(save_directory) / "maxtext_easy_config_metadata.json").write_text(
      json.dumps(metadata), encoding="utf-8"
  )
