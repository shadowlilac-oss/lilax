"""Checkpoint selection and QAT restore preparation for shared trainers."""

from lilax._backend.trainers.post_train.checkpoint_selection import select_training_checkpoint
from lilax.configuration import QwixQATConfig
from lilax.peft import PeftModel


def prepare_training_checkpoint(checkpoint_step, manager, args, config, model, *, resume=True):
  policy = args.checkpoint_config
  manager.save_only_lora_params = None if policy is None else policy.save_only_lora_params
  manager.restore_only_lora_params = None if policy is None else policy.restore_only_lora_params
  marker = None
  if resume:
    manager, checkpoint_step, marker = select_training_checkpoint(
        config.checkpoint_dir, None, step=checkpoint_step, manager=manager,
    )
    if checkpoint_step is None:
      raise ValueError("No training checkpoint is available to resume.")
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
    if manager.save_only_lora_params is True or manager.restore_only_lora_params is True:
      raise NotImplementedError("QAT with LoRA requires full-model checkpointing; adapter-only selection omits QAT state.")
    manager.save_only_lora_params = False
    manager.restore_only_lora_params = False
    if config.lora.lora_restore_path or getattr(config, "learn_to_init_mode", False):
      raise NotImplementedError("QAT with LoRA requires full-model checkpointing with optimizer state.")
  return manager, checkpoint_step
