# Hyperconnection settings

Return to the [configuration reference](configuration.md).

Import these configurations from `lilax.configuration`. Keep the settings loaded with the matching model family. See [model configuration](../../guides/models/configuration.md) for loading and updating a focused domain.

These are constructor defaults. A loaded checkpoint can supply different values. Changing architecture dimensions or routing does not convert existing weights.

## HyperconnectionConfig

Import from `lilax.configuration`. Pydantic model with keyword fields.

Configures manifold-constrained hyperconnections. Pallas mHC requires `enable_mhc_lite=True`. Transformer Engine mHC requires expansion rate four, lite disabled, and Pallas disabled. Invalid combinations raise `ValueError`.

<details>
<summary>Parameters</summary>

- `mhc_expansion_rate` (`PositiveInt`, optional, defaults to `1`). Number of parallel streams.
- `sinkhorn_iterations` (`PositiveInt`, optional, defaults to `20`). Sinkhorn-Knopp iteration count.
- `enable_mhc_lite` (`bool`, optional, defaults to `False`). Stream mixing through a convex combination of permutations instead of Sinkhorn iterations.
- `use_mhc_pallas_kernel` (`bool`, optional, defaults to `False`). TPU Pallas mHC-lite kernel. Requires lite mode.
- `mhc_pallas_kernel_fwd_block_size` (`int`, optional, defaults to `256`). Forward block size.
- `mhc_pallas_kernel_bwd_block_size` (`int`, optional, defaults to `128`). Backward row block size.
- `mhc_pallas_kernel_bwd_feature_block_size` (`int`, optional, defaults to `1024`). Backward feature block size.
- `te_mhc` (`bool`, optional, defaults to `False`). Transformer Engine fused GPU mHC with the restrictions above.

</details>

