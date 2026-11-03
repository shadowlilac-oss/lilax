"""Splash attention configuration."""

from pydantic import BaseModel, Field, NonNegativeInt


class SplashAttention(BaseModel):
  """Tunable block sizes for Splash Attention kernels."""

  sa_block_q: int = Field(512, description="Block size for Q in splash attention.")
  sa_block_kv: int = Field(512, description="Block size for KV in splash attention.")
  sa_block_kv_compute: int = Field(512, description="Block size for KV compute in splash attention.")
  eval_sa_block_q: int = Field(512, description="Block size for Q in splash attention during evaluation.")
  eval_sa_block_kv: int = Field(512, description="Block size for KV in splash attention during evaluation.")
  eval_sa_block_kv_compute: int = Field(512, description="Block size for KV compute in splash attention during evaluation.")
  sa_block_q_dkv: int = Field(512, description="Block size for Q_dkv in splash attention.")
  sa_block_kv_dkv: int = Field(512, description="Block size for KV_dkv in splash attention.")
  sa_block_kv_dkv_compute: int = Field(512, description="Block size for KV_dkv compute in splash attention.")
  sa_block_q_dq: int = Field(512, description="Block size for Q_dq in splash attention.")
  sa_block_kv_dq: int = Field(512, description="Block size for KV_dq in splash attention.")
  sa_use_fused_bwd_kernel: bool = Field(False, description="Use fused backward kernel in splash attention.")
  sa_bwd_dkv_megacore: bool = Field(
      False, description="Megacore-parallel kv-head groups in the static dkv grid. Needs >1 KV head; useful at local batch 1."
  )
  sa_q_layout: str = Field("HEAD_DIM_MINOR", description="Layout for Q in splash attention.")
  sa_k_layout: str = Field("HEAD_DIM_MINOR", description="Layout for K in splash attention.")
  sa_v_layout: str = Field("HEAD_DIM_MINOR", description="Layout for V in splash attention.")
  eval_sa_q_layout: str = Field("HEAD_DIM_MINOR", description="Layout for Q in splash attention during evaluation.")
  eval_sa_k_layout: str = Field("HEAD_DIM_MINOR", description="Layout for K in splash attention during evaluation.")
  eval_sa_v_layout: str = Field("HEAD_DIM_MINOR", description="Layout for V in splash attention during evaluation.")
  use_splash_scheduler: bool = Field(False, description="Use experimental splash attention scheduler.")
  ring_scan_unroll: NonNegativeInt = Field(1, description=(
      "Unroll factor for the Tokamax ring attention scan. 0 fully unrolls; values at or above the ring size are "
      "equivalent to full unrolling. Program size and compile time grow with the factor."
  ))
  sa_fuse_reciprocal: bool = Field(True, description="Maps to fuse_reciprocal in SplashConfig.")
  sa_use_base2_exp: bool = Field(True, description="Maps to use_base2_exp in SplashConfig.")
  sa_qk_diag_skip: bool = Field(False, description="Maps to qk_diag_skip in SplashConfig.")
  sa_sv_diag_skip: bool = Field(False, description="Maps to sv_diag_skip in SplashConfig.")
  sa_qk_diag_grid: int = Field(2, description="Maps to qk_diag_grid in SplashConfig.")
  local_sa_block_q: int | None = Field(None, description="Block size for Q in local splash attention.")
  local_sa_block_kv: int | None = Field(None, description="Block size for KV in local splash attention.")
  local_sa_block_kv_compute: int | None = Field(None, description="Block size for KV compute in local splash attention.")
  eval_local_sa_block_q: int | None = Field(None, description="Block size for Q in local splash attention during evaluation.")
  eval_local_sa_block_kv: int | None = Field(None, description="Block size for KV in local splash attention during evaluation.")
  eval_local_sa_block_kv_compute: int | None = Field(None, description="Block size for KV compute in local splash attention during evaluation.")
  local_sa_block_q_dkv: int | None = Field(None, description="Block size for Q_dkv in local splash attention.")
  local_sa_block_kv_dkv: int | None = Field(None, description="Block size for KV_dkv in local splash attention.")
  local_sa_block_kv_dkv_compute: int | None = Field(None, description="Block size for KV_dkv compute in local splash attention.")
  local_sa_block_q_dq: int | None = Field(None, description="Block size for Q_dq in local splash attention.")
  local_sa_block_kv_dq: int | None = Field(None, description="Block size for KV_dq in local splash attention.")
  local_sa_use_fused_bwd_kernel: bool | None = Field(None, description="Use fused backward kernel in local splash attention.")
  local_sa_q_layout: str | None = Field(None, description="Layout for Q in local splash attention.")
  local_sa_k_layout: str | None = Field(None, description="Layout for K in local splash attention.")
  local_sa_v_layout: str | None = Field(None, description="Layout for V in local splash attention.")
  eval_local_sa_q_layout: str | None = Field(None, description="Layout for Q in local splash attention during evaluation.")
  eval_local_sa_k_layout: str | None = Field(None, description="Layout for K in local splash attention during evaluation.")
  eval_local_sa_v_layout: str | None = Field(None, description="Layout for V in local splash attention during evaluation.")
  local_use_splash_scheduler: bool | None = Field(None, description="Use experimental local splash attention scheduler.")
  local_sa_fuse_reciprocal: bool | None = Field(None, description="Maps to local fuse_reciprocal in SplashConfig.")
  local_sa_use_base2_exp: bool | None = Field(None, description="Maps to local use_base2_exp in SplashConfig.")
  experimental_sa_quant_q_fp8: bool = Field(False, description=(
      "Experimental flag: If enabled, the Q tensor in splash attention is"
      " quantized to jnp.float8_e4m3fn, without scaling factors."
  ))
  experimental_sa_quant_k_fp8: bool = Field(False, description=(
      "Experimental flag: If enabled, the K tensor in splash attention is"
      " quantized to jnp.float8_e4m3fn, without scaling factors."
  ))
  sa_dkv_reuse_scaled_q: bool = Field(False, description=(
      "Experimental flag (maps to SplashConfig.dkv_reuse_scaled_q). With sa_use_base2_exp=True, the"
      " fused dkv backward kernel normally keeps two copies of q live (LOG2E-scaled q for logits and"
      " unscaled q for dK). When True, dkv reuses the LOG2E-scaled q for the dK matmul and divides dK"
      " by LOG2E, keeping only one live q copy. Works with bf16 or fp8 q across all attention types."
      " Subsumed by sa_q_prescaled_log2e; prefer sa_dkv_reuse_scaled_q when q is bf16 or on paths where"
      " sa_q_prescaled_log2e is not supported (non-MLA or use_mla_absorbed_mqa=True)."
  ))
  sa_q_prescaled_log2e: bool = Field(False, description=(
      "Experimental flag (maps to SplashConfig.q_prescaled_log2e). With sa_use_base2_exp=True and"
      " experimental_sa_quant_q_fp8=True, in-kernel LOG2E scaling rounds q to fp8 a second time after"
      " MaxText's softmax_scale cast. When True, attention_mla.py folds LOG2E into the f32"
      " softmax_scale multiply before the fp8 cast (single fp8 rounding), and the kernel skips internal"
      " q scaling in fwd/dq/dkv and divides dq and dk by LOG2E. Strictly subsumes sa_dkv_reuse_scaled_q"
      " (sa_q_prescaled_log2e > sa_dkv_reuse_scaled_q > both False for non-absorbed MLA with fp8 q)."
  ))
  sa_ignore_segment_ids: bool = Field(False, description=(
      "Do not pass decoder segment ids to splash attention, which skips the"
      " per-block segment mask. Only valid when every row is a single"
      " segment: dataset_type='c4_mlperf' with stream chunking for both"
      " train and eval. Requires attention='flash' and attention_type in"
      " ('global', 'mla', 'local_sliding', 'chunk')."
  ))
  use_max_logit_estimate: int = Field(-1, description="-1 means no estimate, any > 0 value will be used as max logit estimate")
  cost_estimate_flops_fwd: int = Field(-1, description=(
      "-1 means using splash default cost estimation, any >= 0 value will be used as cost estimation for splash "
      "to overlap for communication (forward)"
  ))
  cost_estimate_flops_bwd: int = Field(-1, description=(
      "-1 means using splash default cost estimation, any >= 0 value will be used as cost estimation for splash "
      "to overlap for communication (backward)"
  ))
  dq_reduction_steps: int = Field(0, description="the number of reduction steps. For now, only 3 or all the kv steps are supported.")
