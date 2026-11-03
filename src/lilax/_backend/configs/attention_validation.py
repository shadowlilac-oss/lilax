"""Validation of native attention configuration."""

from lilax._backend.common.common_types import AttentionType, DecoderBlockType
from lilax._backend.utils import max_logging


def validate_attention_backend_support(config):
  if config.moba and config.attention not in ("dot_product"):
    raise ValueError("MoBA is only supported with dot_product attention.")
  if config.decoder_block == DecoderBlockType.DEEPSEEK4:
    match (config.attention, config.use_tokamax_splash):
      case ("dot_product", _):
        pass
      case ("flash", True):
        pass
      case _:
        raise ValueError(
            "DeepSeek4 is only supported with `dot_product` attention or `flash` attention "
            "with `use_tokamax_splash=True`."
        )


def validate_attention_type(self):
  if self.attention_type == AttentionType.CHUNK.value and (
      not isinstance(self.chunk_attn_window_size, int) or self.chunk_attn_window_size <= 0
  ):
    raise ValueError("`chunk_attn_window_size` must be an integer > 0 for 'chunk' attention.")
  if self.attention_type == AttentionType.LOCAL_SLIDING.value and (
      not isinstance(self.sliding_window_size, int) or self.sliding_window_size <= 0
  ):
    raise ValueError("`sliding_window_size` must be an integer > 0 for 'local_sliding' attention.")
  if self.attention_type == AttentionType.BLOCK_DIFFUSION.value:
    if self.packing:
      # Document-local block origins inside a packed sequence are not tracked
      # in attention metadata; packing without realignment would cause
      # cross-document block-attention leakage.
      raise ValueError("Block-diffusion attention does not support packing; set `packing=False`.")
    if self.attention not in ("autoselected", "dot_product", "flash"):
      raise ValueError("Block-diffusion attention is supported only by dot_product attention and TPU Splash attention.")
    if self.attention in ("autoselected", "flash") and self.hardware != "tpu":
      raise ValueError(
          "Block-diffusion attention with attention='autoselected' or attention='flash' requires hardware='tpu'; "
          "use attention='dot_product' on other hardware."
      )


def validate_splash_diag_skip(self):
  """Validates Tokamax Splash diagonal sub-tile skipping and warns when it cannot take effect."""
  if self.sa_qk_diag_grid < 2 or (self.sa_qk_diag_grid & (self.sa_qk_diag_grid - 1)) != 0:
    raise ValueError(f"sa_qk_diag_grid must be a power of 2 and >= 2, got {self.sa_qk_diag_grid}.")
  if not (self.use_tokamax_splash and (self.sa_qk_diag_skip or self.sa_sv_diag_skip)):
    return
  # Tokamax checks square training blocks when the kernel is built at startup, but `eval_sa_*` blocks are
  # only used under custom eval rules and first traced at the first eval step, so check them here.
  uses_eval_sa_blocks = bool(self.logical_axis_rules_for_eval) and (
      self.logical_axis_rules_for_eval != self.logical_axis_rules
  )
  if uses_eval_sa_blocks and not self.eval_sa_block_q == self.eval_sa_block_kv == self.eval_sa_block_kv_compute:
    raise ValueError(
        "sa_qk_diag_skip/sa_sv_diag_skip with custom eval logical axis rules require square eval blocks "
        "(eval_sa_block_q == eval_sa_block_kv == eval_sa_block_kv_compute); got "
        f"{self.eval_sa_block_q}/{self.eval_sa_block_kv}/{self.eval_sa_block_kv_compute}."
    )
  # The skip only runs when the Splash kernel gets a pure `CausalMask`; otherwise it is silently disabled.
  cp_size = self.ici_context_parallelism * self.dcn_context_parallelism
  strategy = self.context_parallel_strategy.lower()
  if self.attention_type not in (AttentionType.GLOBAL.value, AttentionType.MLA.value):
    reason = f"attention_type={self.attention_type!r} (only 'global' and 'mla' are supported)"
  elif cp_size > 1 and strategy in ("ring", "usp"):
    reason = f"context_parallel_strategy={strategy!r} (ring/USP attention does not use the flags)"
  elif cp_size > 1 and strategy == "all_gather" and self.context_parallel_load_balance:
    reason = "load-balanced all_gather context parallelism (it uses LoadBalancedCausalMask)"
  else:
    return
  max_logging.warning(
      f"sa_qk_diag_skip={self.sa_qk_diag_skip}, sa_sv_diag_skip={self.sa_sv_diag_skip} have no effect with "
      f"{reason}. Diagonal sub-tile skipping requires a pure causal mask."
  )
