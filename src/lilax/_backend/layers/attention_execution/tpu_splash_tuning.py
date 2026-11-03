"""Splash Attention block sizes, layouts and kernel options of an attention layer, passed to the flash kernels."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class SplashTuning:
  """Training and evaluation Splash settings read from the attention layer."""

  block_q: int
  block_kv: int
  block_kv_compute: int
  eval_block_q: int
  eval_block_kv: int
  eval_block_kv_compute: int
  block_q_dkv: int
  block_kv_dkv: int
  block_kv_dkv_compute: int
  block_q_dq: int
  block_kv_dq: int
  use_fused_bwd_kernel: bool
  q_layout: str
  k_layout: str
  v_layout: str
  eval_q_layout: str
  eval_k_layout: str
  eval_v_layout: str
  use_splash_scheduler: bool
  fuse_reciprocal: bool
  use_base2_exp: bool
  qk_diag_skip: bool
  sv_diag_skip: bool


@dataclasses.dataclass(frozen=True)
class SplashBlockSizes:
  """Splash block sizes and layouts selected for the current train or eval step."""

  block_q: int
  block_kv: int
  block_kv_compute: int
  block_q_dkv: int
  block_kv_dkv: int
  block_kv_dkv_compute: int
  block_q_dq: int
  block_kv_dq: int
  q_layout: str
  k_layout: str
  v_layout: str


def select_block_sizes(tuning: SplashTuning, is_eval: bool) -> SplashBlockSizes:
  """Eval reuses its forward blocks for the backward blocks; training uses the dedicated ones."""
  if is_eval:
    return SplashBlockSizes(
        block_q=tuning.eval_block_q,
        block_kv=tuning.eval_block_kv,
        block_kv_compute=tuning.eval_block_kv_compute,
        block_q_dkv=tuning.eval_block_q,
        block_kv_dkv=tuning.eval_block_kv,
        block_kv_dkv_compute=tuning.eval_block_kv_compute,
        block_q_dq=tuning.eval_block_q,
        block_kv_dq=tuning.eval_block_kv,
        q_layout=tuning.eval_q_layout,
        k_layout=tuning.eval_k_layout,
        v_layout=tuning.eval_v_layout,
    )
  return SplashBlockSizes(
      block_q=tuning.block_q,
      block_kv=tuning.block_kv,
      block_kv_compute=tuning.block_kv_compute,
      block_q_dkv=tuning.block_q_dkv,
      block_kv_dkv=tuning.block_kv_dkv,
      block_kv_dkv_compute=tuning.block_kv_dkv_compute,
      block_q_dq=tuning.block_q_dq,
      block_kv_dq=tuning.block_kv_dq,
      q_layout=tuning.q_layout,
      k_layout=tuning.k_layout,
      v_layout=tuning.v_layout,
  )
