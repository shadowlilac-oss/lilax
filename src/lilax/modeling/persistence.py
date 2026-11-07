"""Native model checkpoint persistence and Hugging Face export."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from etils import epath
from flax import nnx
import jax
import orbax.checkpoint as ocp

from lilax._backend.checkpoint_conversion.model_families.gpt_oss.fused import split_moe_weights
from lilax._backend.checkpoint_conversion.model_families.qwen3_5 import fused as fused_qwen3_5
from lilax._backend.checkpoint_conversion.to_huggingface import convert_to_huggingface
from lilax._backend.checkpoint_conversion.projection_layout import (
    is_hybrid_qwen_adapter, projection_layout_adapter, supports_mlp_layout, validate_huggingface_projection_layout,
    validate_mlp_only_layout,
)
from lilax._backend.integration.vllm.convert_utils import _partition_size
from lilax.configuration import QwixQATConfig
from lilax.configuration.model_mapping import model_config_from_effective
from lilax.modeling.hub import get_model_capabilities, resolve_model_name, resolve_model_snapshot


def checkpoint_qat_config(path):
  checkpoint_dir = epath.Path(path)
  metadata_path = checkpoint_dir / "maxtext_easy_config_metadata.json"
  config_path = checkpoint_dir / "config.json"
  if config_path.exists():
    saved_recipe = json.loads(config_path.read_text(encoding="utf-8")).get("quantization_config")
    if isinstance(saved_recipe, dict) and saved_recipe.get("quant_method") == "QwixQATConfig":
      return QwixQATConfig.model_validate({name: value for name, value in saved_recipe.items() if name != "quant_method"})
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
    config = model.construction_config
    if (getattr(config.decoder_block, "value", config.decoder_block) == "gpt_oss"
        and config.prefuse_moe_weights and config.padded_base_moe_mlp_dim is not None
        and config.padded_base_moe_mlp_dim != config.mlp_dim):
      raise NotImplementedError("Hugging Face export of padded GPT-OSS expert output weights is not supported.")
    validate_huggingface_projection_layout(model.construction_config)
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
  updates = {}
  updates["scan_layers"] = model.construction_config.scan_layers
  updates["hardware"] = model.runtime.config.hardware
  updates["qwix_qat_config"] = model.existing_graph_qat_config
  state = (
      nnx.state(model.native_model)
      .filter(lambda path, var: not isinstance(var, (nnx.RngState, nnx.Cache, nnx.Intermediate, nnx.BatchStat)))
      .to_pure_dict()
  )
  config = model.construction_config
  decoder = getattr(config.decoder_block, "value", config.decoder_block)
  if decoder == "gpt_oss" and config.prefuse_moe_weights:
    if model.existing_graph_qat_config is not None:
      raise NotImplementedError("Portable GPT-OSS prefused export does not support Qwix QAT.")
    width = config.mlp_dim
    padded_width = config.padded_base_moe_mlp_dim if config.padded_base_moe_mlp_dim is not None else width
    if not isinstance(width, int) or not isinstance(padded_width, int) or width <= 0 or padded_width < width:
      raise ValueError("Cannot determine GPT-OSS prefused weight widths from the model configuration.")
    if padded_width != width:
      raise NotImplementedError("Portable GPT-OSS prefused export does not support padded expert weights.")
    def split_live_moe(node):
      if not isinstance(node, dict):
        return 0
      count = 0
      mlp = node.get("GptOssMlp")
      if isinstance(mlp, dict) and "wi" in mlp:
        wi = mlp.pop("wi")
        if mlp.get("wi_scale") is not None:
          raise NotImplementedError("Portable GPT-OSS prefused FP8 expert weights are not supported.")
        sharding = getattr(wi, "sharding", None)
        if not isinstance(sharding, jax.sharding.NamedSharding):
          raise ValueError("Cannot determine GPT-OSS prefused source shard count from wi sharding.")
        axis = wi.ndim - 1
        partition = sharding.spec[axis] if axis < len(sharding.spec) else None
        shards = _partition_size(partition, sharding.mesh)
        if wi.shape[axis] != 2 * padded_width or width % shards or padded_width % shards:
          raise ValueError("GPT-OSS prefused weights have incompatible widths or uneven shard chunks.")
        weights = split_moe_weights(wi, axis=axis, num_shards=shards)
        mlp["wi_0"], mlp["wi_1"] = weights
        count += 1
      for value in node.values():
        count += split_live_moe(value)
      return count

    if not split_live_moe(state):
      raise ValueError("GPT-OSS prefused model does not contain the expected expert wi weights.")
    updates["prefuse_moe_weights"] = False
  validate_mlp_only_layout(config)
  layout_adapter = projection_layout_adapter(config)
  unfuse_weights = layout_adapter is not None and (
      config.fused_qkv or (supports_mlp_layout(layout_adapter) and config.fused_mlp)
  )
  if unfuse_weights:
    if model.existing_graph_qat_config is not None:
      raise NotImplementedError("Portable fused projection export does not support Qwix QAT.")
    target_mlp = False if supports_mlp_layout(layout_adapter) else config.fused_mlp
    if is_hybrid_qwen_adapter(layout_adapter) and layout_adapter is not fused_qwen3_5:
      converted = layout_adapter.convert_native_state({"decoder": state["decoder"]}, config,
                                                       fused_qkv=False, fused_mlp=target_mlp)
      state["decoder"] = converted["decoder"]
    else:
      state = layout_adapter.convert_native_state(state, config, fused_qkv=False, fused_mlp=target_mlp)
    updates.update({"fused_qkv": False, "fused_mlp": target_mlp})
  saved_config = model_config_from_effective(
      public_config, model.construction_config, updates=updates, metadata_config=model.config,
  )
  config_json = saved_config.to_json_string(use_diff=False)

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
  if decoder == "gpt_oss" and config.prefuse_moe_weights:
    explicit_keys |= {"prefuse_moe_weights"}
  if unfuse_weights:
    explicit_keys |= {"fused_qkv", "fused_mlp"}
  if model.existing_graph_qat_config is not None:
    explicit_keys |= {"qwix_qat_config"}
  metadata = {
      "explicit_keys": sorted(explicit_keys - {"hf_access_token"}),
      "checkpoint_layout": {"scan_layers": config.scan_layers, "param_scan_axis": config.param_scan_axis},
  }
  (epath.Path(save_directory) / "maxtext_easy_config_metadata.json").write_text(
      json.dumps(metadata), encoding="utf-8"
  )
