"""Grouped matmul tile sizes for the DeepSeek FP8 batch-split routed experts."""

from lilax._backend.kernels.tiling import eval_tiles


def gmm_tile_sizes(config):
  """Returns the (wi, wo) megablox tile sizes, using the eval forward tiles in eval mode."""
  if eval_tiles.is_eval(config):
    wi_tile_fwd_batch_seq = config.eval_wi_tile_fwd_batch_seq
    wi_tile_fwd_embed_dim = config.eval_wi_tile_fwd_embed_dim
    wi_tile_fwd_mlp_dim = config.eval_wi_tile_fwd_mlp_dim
    wo_tile_fwd_batch_seq = config.eval_wo_tile_fwd_batch_seq
    wo_tile_fwd_embed_dim = config.eval_wo_tile_fwd_embed_dim
    wo_tile_fwd_mlp_dim = config.eval_wo_tile_fwd_mlp_dim
  else:
    wi_tile_fwd_batch_seq = config.wi_tile_fwd_batch_seq
    wi_tile_fwd_embed_dim = config.wi_tile_fwd_embed_dim
    wi_tile_fwd_mlp_dim = config.wi_tile_fwd_mlp_dim
    wo_tile_fwd_batch_seq = config.wo_tile_fwd_batch_seq
    wo_tile_fwd_embed_dim = config.wo_tile_fwd_embed_dim
    wo_tile_fwd_mlp_dim = config.wo_tile_fwd_mlp_dim

  wi_tile_size = (
      wi_tile_fwd_batch_seq,  # m (LHS batch)
      wi_tile_fwd_embed_dim,  # k  (contracting)
      wi_tile_fwd_mlp_dim,  # n (RHS batch)
      config.wi_tile_dlhs_batch_seq,  # m (LHS batch)
      config.wi_tile_dlhs_mlp_dim,  # k (contracting)
      config.wi_tile_dlhs_embed_dim,  # n (RHS batch)
      config.wi_tile_drhs_batch_seq,  # Called m in megablox, but this is contracting
      config.wi_tile_drhs_embed_dim,  # Called k in megablox, but this is LHS batch dim
      config.wi_tile_drhs_mlp_dim,  # Called n in megablox, and indeed is the RHS batch dim
  )

  wo_tile_size = (
      wo_tile_fwd_batch_seq,  # m (LHS batch)
      wo_tile_fwd_mlp_dim,  # k (contracting)
      wo_tile_fwd_embed_dim,  # n (RHS batch)
      config.wo_tile_dlhs_batch_seq,  # m (LHS batch)
      config.wo_tile_dlhs_embed_dim,  # k (contracting)
      config.wo_tile_dlhs_mlp_dim,  # n (RHS)
      config.wo_tile_drhs_batch_seq,  # Called m in megablox, but this is contracting
      config.wo_tile_drhs_mlp_dim,  # Called k in megablox, but this is LHS batch dim
      config.wo_tile_drhs_embed_dim,  # Called n in megablox, and indeed is the RHS batch dim
  )
  return wi_tile_size, wo_tile_size
