"""Derivation of local Splash Attention settings."""


def resolve_local_splash_defaults(config):
  if config.local_sa_block_q is None:
    config.local_sa_block_q = config.sa_block_q
  if config.local_sa_block_kv is None:
    config.local_sa_block_kv = config.sa_block_kv
  if config.local_sa_block_kv_compute is None:
    config.local_sa_block_kv_compute = config.sa_block_kv_compute
  if config.eval_local_sa_block_q is None:
    config.eval_local_sa_block_q = config.eval_sa_block_q
  if config.eval_local_sa_block_kv is None:
    config.eval_local_sa_block_kv = config.eval_sa_block_kv
  if config.eval_local_sa_block_kv_compute is None:
    config.eval_local_sa_block_kv_compute = config.eval_sa_block_kv_compute
  if config.local_sa_block_q_dkv is None:
    config.local_sa_block_q_dkv = config.sa_block_q_dkv
  if config.local_sa_block_kv_dkv is None:
    config.local_sa_block_kv_dkv = config.sa_block_kv_dkv
  if config.local_sa_block_kv_dkv_compute is None:
    config.local_sa_block_kv_dkv_compute = config.sa_block_kv_dkv_compute
  if config.local_sa_block_q_dq is None:
    config.local_sa_block_q_dq = config.sa_block_q_dq
  if config.local_sa_block_kv_dq is None:
    config.local_sa_block_kv_dq = config.sa_block_kv_dq
  if config.local_sa_use_fused_bwd_kernel is None:
    config.local_sa_use_fused_bwd_kernel = config.sa_use_fused_bwd_kernel
  if config.local_sa_q_layout is None:
    config.local_sa_q_layout = config.sa_q_layout
  if config.local_sa_k_layout is None:
    config.local_sa_k_layout = config.sa_k_layout
  if config.local_sa_v_layout is None:
    config.local_sa_v_layout = config.sa_v_layout
  if config.eval_local_sa_q_layout is None:
    config.eval_local_sa_q_layout = config.eval_sa_q_layout
  if config.eval_local_sa_k_layout is None:
    config.eval_local_sa_k_layout = config.eval_sa_k_layout
  if config.eval_local_sa_v_layout is None:
    config.eval_local_sa_v_layout = config.eval_sa_v_layout
  if config.local_use_splash_scheduler is None:
    config.local_use_splash_scheduler = config.use_splash_scheduler
  if config.local_sa_fuse_reciprocal is None:
    config.local_sa_fuse_reciprocal = config.sa_fuse_reciprocal
  if config.local_sa_use_base2_exp is None:
    config.local_sa_use_base2_exp = config.sa_use_base2_exp
