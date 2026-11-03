"""Checkpoint selection and QAT restore preparation for shared trainers."""

from lilax._backend.trainers.post_train.checkpoint_selection import select_training_checkpoint
from lilax.configuration import QwixQATConfig
from lilax.peft import PeftModel


def prepare_training_checkpoint(checkpoint_step, manager, args, config, model):
  manager.save_only_lora_params = getattr(args, "save_only_lora_params", None)
  manager.restore_only_lora_params = getattr(args, "restore_only_lora_params", None)
  manager, checkpoint_step, marker = select_training_checkpoint(
      config.checkpoint_dir, None, step=checkpoint_step, manager=manager,
  )
  if marker is not None:
    if (not isinstance(marker, dict) or set(marker) != {"version", "kind", "qat_config"}
        or type(marker["version"]) is not int or marker["version"] != 1
        or marker["kind"] not in ("qat", "fp")
        or (marker["kind"] == "fp" and marker["qat_config"] is not None)
        or (marker["kind"] == "qat" and not isinstance(marker["qat_config"], dict))):
      raise ValueError("Invalid full-state training graph metadata.")
    if marker["kind"] == "qat":
      if config.lora.lora_restore_path:
        raise ValueError("Cannot combine a QAT full-state restore with lora_restore_path.")
      model.prepare_for_full_state_restore(
          QwixQATConfig.model_validate(marker["qat_config"]), runtime=model.runtime,
      )
    elif model.pending_qat_config is not None:
      raise ValueError("The selected full-state checkpoint has an FP graph, not the requested QAT graph.")
  elif checkpoint_step is not None and model.pending_qat_config is not None:
    raise ValueError("Cannot restore QAT full state without training graph metadata.")
  if (model.pending_qat_config is not None
      and (isinstance(model, PeftModel) or config.lora.enable_lora or config.lora.lora_restore_path)):
    if manager.save_only_lora_params is not False or manager.restore_only_lora_params is not False:
      raise NotImplementedError("QAT with LoRA requires full-model save and restore to preserve QAT statistics.")
    if config.lora.lora_restore_path or getattr(config, "learn_to_init_mode", False):
      raise NotImplementedError("QAT with LoRA requires full-model checkpointing with optimizer state.")
  return manager, checkpoint_step
