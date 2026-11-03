"""Validation and derivation of native DiLoCo configuration."""

from math import prod


def validate_streaming_diloco(config):
  if config.enable_streaming_diloco:
    if not config.scan_layers:
      raise ValueError("enable_streaming_diloco=True requires scan_layers=True.")
    if config.num_diloco_fragments is not None and config.num_diloco_fragments > 1:
      num_transformer_fragments = config.num_diloco_fragments - 1
      if config.num_decoder_layers % num_transformer_fragments != 0:
        raise ValueError(
            f"The number of decoder layers ({config.num_decoder_layers}) must be divisible by "
            f"(num_diloco_fragments - 1) ({num_transformer_fragments}) when enable_streaming_diloco is True."
        )
    if config.diloco_bucketize_non_scanned and config.shard_optimizer_over_data:
      # train_compile.py derives the outer param shardings after the Zero-1 overlay, the runtime path before it,
      # so the bucket split axes could differ between AOT and training.
      raise ValueError("diloco_bucketize_non_scanned does not support shard_optimizer_over_data (Zero-1).")


def resolve_diloco_replicas(config):
  # Resolve dcn_diloco_parallelism=-1 if left unspecified, using the same convention as dcn_data_parallelism.
  # num_diloco_replicas must be computed after this resolution, so we resolve it here rather than
  # relying on fill_unspecified_mesh_axes (which runs later during mesh creation).
  if config.dcn_diloco_parallelism == -1:
    other_dcn_product = prod(v for v in config.dcn_parallelism if v != -1)
    assert other_dcn_product > 0 and config.num_slices % other_dcn_product == 0, (
        f"Cannot resolve dcn_diloco_parallelism=-1: num_slices={config.num_slices} is not divisible "
        f"by the product of other DCN parallelism values ({other_dcn_product})."
    )
    config.dcn_diloco_parallelism = config.num_slices // other_dcn_product
    # Keep dcn_parallelism list consistent with the resolved value.
    diloco_idx = config.dcn_parallelism.index(-1)
    config.dcn_parallelism[diloco_idx] = config.dcn_diloco_parallelism
  config.num_diloco_replicas = int(config.ici_diloco_parallelism * config.dcn_diloco_parallelism)


def validate_diloco_grouped_matmul(config):
  # (b/496973624) use_tokamax_gmm is incompatible with enable_diloco: drjax.map_fn wraps
  # the train step in jax.vmap over the diloco axis, which causes JAX to batch through
  # lax.scan (layer scan).
  # Tokamax's vmap_rule then tries to reconstruct GroupSizes with a batched 2-D value, but
  # GroupSizes.__post_init__ requires exactly a 1-D shape.
  if config.enable_diloco and config.use_tokamax_gmm:
    raise ValueError(
        "use_tokamax_gmm=True is not compatible with enable_diloco=True due to a known "
        "incompatibility between tokamax's GroupSizes vmap_rule and JAX's scan batching. "
        "Please set use_tokamax_gmm=False."
    )
