from lilax.configuration.domain_types import DType
from lilax.configuration.domains.execution_mesh import ShardMode


def validate_weight_layout_options(self):
  if self.lm_head_weight_grad_in_kernel_order is None:
    self.lm_head_weight_grad_in_kernel_order = self.shard_mode == ShardMode.EXPLICIT and not self.logits_via_embedding
  elif self.lm_head_weight_grad_in_kernel_order and self.logits_via_embedding:
    raise ValueError(
        "lm_head_weight_grad_in_kernel_order only applies to the untied LM head, but logits_via_embedding is True."
    )
  if self.dense_weight_grad_in_kernel_order is None:
    self.dense_weight_grad_in_kernel_order = self.shard_mode == ShardMode.EXPLICIT
  if self.save_fsdp_gathered_weights and self.quantization:
    raise ValueError("save_fsdp_gathered_weights casts kernels to config.dtype, which is not valid when quantizing.")
  if self.save_fsdp_gathered_weights and self.weight_dtype in (DType.FLOAT8_E4M3FN, DType.FLOAT8_E5M2):
    raise ValueError(
        "save_fsdp_gathered_weights casts kernels to config.dtype, which discards the fp8 encoding "
        f"of weight_dtype={self.weight_dtype}."
    )
  if self.save_fsdp_gathered_weights and self.parameter_memory_host_offload:
    raise ValueError(
        "save_fsdp_gathered_weights keeps gathered kernels in HBM, which defeats parameter_memory_host_offload."
    )
  return self
