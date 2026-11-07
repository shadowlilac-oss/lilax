"""Batch size and sequence length resolution per model mode."""

from lilax._backend.common.common_types import MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_TRAIN


@staticmethod
def get_batch_seq_len_for_mode(config, model_mode):
  """Resolves the batch size and sequence length based on the model's operational mode."""
  if model_mode == MODEL_MODE_PREFILL:
    batch_size = 1
    seq_len = config.max_prefill_predict_length
  elif model_mode == MODEL_MODE_AUTOREGRESSIVE:
    batch_size = config.micro_batch_size_to_train_on
    seq_len = 1
  elif model_mode == MODEL_MODE_TRAIN:
    batch_size = config.micro_batch_size_to_train_on
    seq_len = config.max_target_length
  else:
    raise ValueError(f"Unknown model_mode: {model_mode}")
  return batch_size, seq_len
