# Wan components

Import from `lilax.generation.wan`. See [component guide](../../guides/components/wan-components.md).

## Configuration and sharding

`load_yaml_config(file_path, search_paths=None)` reads mappings, `DiffusionConfig.from_dict(raw)` creates typed configuration, and `process_config(raw, overrides=None, **kwargs)` initializes JAX and resolves dimensions and directories. Their full entries are in [configuration construction](m3-components.md#load_yaml_config) and [from_dict](m3-components.md#from_dict).

`WAN_MESH_AXES` is `("data", "fsdp", "context", "tensor")`. `VAE_MESH_AXES` is `("redundant", "vae_spatial")`. `WAN21_CONFIG_PATH` identifies the bundled configuration.

### load_wan_config

```python
load_wan_config() -> dict
```

Returns bundled Wan 2.1 14B settings without runtime initialization.

Nested configuration defaults are listed in [diffusion settings](diffusion-settings.md).

### WanSharding

Import from `lilax.generation.wan`. Frozen dataclass.

```python
WanSharding(rules: tuple[tuple[str, MeshAxes], ...] = (),
	mesh_axes: tuple[str, ...] = WAN_MESH_AXES, context_axis: str = "context")
```

Maps logical axes. Unknown mesh axes or duplicate assignment of one mesh axis within a tensor raise `ValueError`.

<details>
<summary>Parameters</summary>

- `rules` (`tuple[tuple[str, MeshAxes], ...]`, optional, defaults to `()`). Mapping of logical names to a mesh name, tuple of names, or `None`.
- `mesh_axes` (`tuple[str, ...]`, optional, defaults to `WAN_MESH_AXES`). Valid physical axis names.
- `context_axis` (`str`, optional, defaults to `"context"`). Context axis name.

</details>

### from_rules

```python
WanSharding.from_rules(rules: Sequence[Sequence[Any]], mesh_axes: Sequence[str] = WAN_MESH_AXES, **kwargs)
```

Returns sharding from normalized rule pairs. Repeated logical names raise `ValueError`.

<details>
<summary>Parameters</summary>

- `rules` (`Sequence[Sequence[Any]]`). Logical-to-mesh pairs.
- `mesh_axes` (`Sequence[str]`, optional, defaults to `WAN_MESH_AXES`). Physical axis names.
- `**kwargs` (`dict[str, Any]`). Other `WanSharding` fields.

</details>

### Sharding call

```python
sharding(tensor_name: str, axes: Sequence[str | None], tensor_type: TensorType = TensorType.WEIGHT) -> PartitionSpec
```

Returns mapped partition information. The default call maps weights. `sharding.constrain` maps activations.

<details>
<summary>Parameters</summary>

- `tensor_name` (`str`). Tensor name for diagnostics.
- `axes` (`Sequence[str | None]`). Logical dimension names, with `None` for replication.
- `tensor_type` (`TensorType`, optional, defaults to `TensorType.WEIGHT`). Weight or activation mapping.

</details>

### constrain

```python
sharding.constrain(x: jax.Array, tensor_name: str, axes: Sequence[str | None]) -> jax.Array
```

Returns the constrained array using activation mapping.

<details>
<summary>Parameters</summary>

- `x` (`jax.Array`). The array.
- `tensor_name` (`str`). Tensor name.
- `axes` (`Sequence[str | None]`). Logical dimension names.

</details>

### map_axis

```python
sharding.map_axis(axis: str, tensor_name: str, tensor_type: TensorType) -> MeshAxes
```

Returns a mesh-axis mapping. Missing rules raise `KeyError`.
For weight mapping without a tensor-category argument, use `sharding(tensor_name, axes)`. For activation constraints, use `sharding.constrain(x, tensor_name, axes)`.

<details>
<summary>Parameters</summary>

- `axis` (`str`). Logical axis name.
- `tensor_name` (`str`). Tensor name.
- `tensor_type` (`TensorType`). The tensor category.

</details>

### create_wan_mesh

```python
create_wan_mesh(config) -> jax.sharding.Mesh
```

Returns a JAX mesh. Split physical axes raise `NotImplementedError`.

<details>
<summary>Parameters</summary>

- `config` (`DiffusionConfig`). Mesh axes and ICI/DCN parallelism settings.

</details>

### get_precision

```python
get_precision(config)
```

Returns the named JAX precision, or `None` for an absent or empty setting. Invalid names fail lookup.

<details>
<summary>Parameters</summary>

- `config` (`DiffusionConfig`). Configuration with an uppercase `precision` name.

</details>

### get_flash_block_sizes

```python
get_flash_block_sizes(config)
```

Returns optional flash-attention block settings.

<details>
<summary>Parameters</summary>

- `config` (`DiffusionConfig`). Configuration with `flash_block_sizes` as a mapping or JSON string, or an empty value for `None`.

</details>

### yml_rules

```python
yml_rules(key: str, path: str = WAN21_CONFIG_PATH) -> tuple
```

Reads YAML and returns rule tuples.

<details>
<summary>Parameters</summary>

- `key` (`str`). The rule entry.
- `path` (`str`, optional, defaults to `WAN21_CONFIG_PATH`). The YAML file.

</details>

## WanModel

Import from `lilax.generation.wan`. Flax NNX module.

```python
WanModel(
	rngs: nnx.Rngs, model_type="t2v", patch_size: Tuple[int] = (1, 2, 2),
	num_attention_heads: int = 40, attention_head_dim: int = 128, in_channels: int = 16,
	out_channels: int = 16, text_dim: int = 4096, freq_dim: int = 256, ffn_dim: int = 13824,
	num_layers: int = 40, dropout: float = 0.0, cross_attn_norm: bool = True,
	qk_norm: Optional[str] = "rms_norm_across_heads", eps: float = 1e-6, image_dim: Optional[int] = None,
	added_kv_proj_dim: Optional[int] = None, rope_max_seq_len: int = 1024, pos_embed_seq_len: Optional[int] = None,
	image_seq_len: Optional[int] = None, flash_min_seq_length: int = 4096, flash_block_sizes: BlockSizes = None,
	sharding: Sharding | None = None, dtype: jnp.dtype = jnp.float32, weights_dtype: jnp.dtype = jnp.float32,
	precision: jax.lax.Precision = None, attention: str = "dot_product", mask_padding_tokens: bool = True,
	scan_layers: bool = True, enable_jax_named_scopes: bool = False, attention_config: Optional[dict] = None,
)
```

Constructs the NNX diffusion transformer. Hidden states have shape `(batch, channels, frames, height, width)`.

<details>
<summary>Parameters</summary>

- `rngs` (`nnx.Rngs`). Construction RNGs.
- `model_type` (`object`, optional, defaults to `"t2v"`). Unused model-type value, with no type constraint.
- `patch_size` (`Tuple[int]`, optional, defaults to `(1, 2, 2)`). Temporal and spatial patch dimensions.
- `num_attention_heads` (`int`, optional, defaults to `40`). Attention heads.
- `attention_head_dim` (`int`, optional, defaults to `128`). Head width.
- `in_channels` (`int`, optional, defaults to `16`). Latent input channels.
- `out_channels` (`int`, optional, defaults to `16`). Output channels.
- `text_dim` (`int`, optional, defaults to `4096`). Conditioning width.
- `freq_dim` (`int`, optional, defaults to `256`). Timestep frequency width.
- `ffn_dim` (`int`, optional, defaults to `13824`). Feed-forward width.
- `num_layers` (`int`, optional, defaults to `40`). Transformer layers.
- `dropout` (`float`, optional, defaults to `0.0`). Dropout probability.
- `cross_attn_norm` (`bool`, optional, defaults to `True`). Flag for cross-attention input normalization.
- `qk_norm` (`Optional[str]`, optional, defaults to `"rms_norm_across_heads"`). Query/key normalization.
- `eps` (`float`, optional, defaults to `1e-6`). Normalization epsilon.
- `image_dim` (`Optional[int]`, optional, defaults to `None`). Image conditioning width.
- `added_kv_proj_dim` (`Optional[int]`, optional, defaults to `None`). Added key/value projection width.
- `rope_max_seq_len` (`int`, optional, defaults to `1024`). Limit on rotary positions.
- `pos_embed_seq_len` (`Optional[int]`, optional, defaults to `None`). Positional embedding length.
- `image_seq_len` (`Optional[int]`, optional, defaults to `None`). Image sequence length.
- `flash_min_seq_length` (`int`, optional, defaults to `4096`). The flash-attention threshold.
- `flash_block_sizes` (`BlockSizes`, optional, defaults to `None`). Flash tiling configuration.
- `sharding` (`Sharding | None`, optional, defaults to `None`). Logical-axis mapping.
- `dtype` (`jnp.dtype`, optional, defaults to `jnp.float32`). Activation dtype.
- `weights_dtype` (`jnp.dtype`, optional, defaults to `jnp.float32`). Weight dtype.
- `precision` (`jax.lax.Precision`, optional, defaults to `None`). Matrix multiplication precision.
- `attention` (`str`, optional, defaults to `"dot_product"`). Attention computation.
- `mask_padding_tokens` (`bool`, optional, defaults to `True`). Flag for padding masks.
- `scan_layers` (`bool`, optional, defaults to `True`). Flag for transformer layer scanning.
- `enable_jax_named_scopes` (`bool`, optional, defaults to `False`). Flag for named scopes.
- `attention_config` (`Optional[dict]`, optional, defaults to `None`). Attention setting overrides.

</details>

### Model call

```python
model(hidden_states: jax.Array, timestep: jax.Array, encoder_hidden_states: jax.Array,
	encoder_hidden_states_image: Optional[jax.Array] = None, return_dict: bool = True,
	attention_kwargs: Optional[Dict[str, Any]] = None, deterministic: bool = True,
	rngs: Optional[nnx.Rngs] = None, skip_blocks: bool = False, cached_residual: Optional[jax.Array] = None,
	return_residual: bool = False, kv_cache: Optional[Dict[str, Tuple[jax.Array, jax.Array]]] = None,
	rotary_emb: Optional[jax.Array] = None, encoder_attention_mask: Optional[jax.Array] = None,
)
```

Returns the denoised array, or `(array, residual)` when `return_residual=True`.
`return_dict` does not change the result.
Skipping blocks without `cached_residual` raises `ValueError`.

<details>
<summary>Parameters</summary>

- `hidden_states` (`jax.Array`). `(batch, channels, frames, height, width)` latents.
- `timestep` (`jax.Array`). Diffusion timesteps.
- `encoder_hidden_states` (`jax.Array`). Text conditioning.
- `encoder_hidden_states_image` (`Optional[jax.Array]`, optional, defaults to `None`). Image conditioning.
- `return_dict` (`bool`, optional, defaults to `True`). Unused output-format flag.
- `attention_kwargs` (`Optional[Dict[str, Any]]`, optional, defaults to `None`). Unused attention arguments.
- `deterministic` (`bool`, optional, defaults to `True`). Flag for deterministic execution without dropout.
- `rngs` (`Optional[nnx.Rngs]`, optional, defaults to `None`). Execution RNGs.
- `skip_blocks` (`bool`, optional, defaults to `False`). Flag for reuse of cached block residuals.
- `cached_residual` (`Optional[jax.Array]`, optional, defaults to `None`). Cached block residual for reuse when `skip_blocks=True`.
- `return_residual` (`bool`, optional, defaults to `False`). Flag for inclusion of the block residual in the result.
- `kv_cache` (`Optional[Dict[str, Tuple[jax.Array, jax.Array]]]`, optional, defaults to `None`). Conditioning cache.
- `rotary_emb` (`Optional[jax.Array]`, optional, defaults to `None`). Precomputed rotary embeddings.
- `encoder_attention_mask` (`Optional[jax.Array]`, optional, defaults to `None`). Conditioning visibility.

</details>

### compute_kv_cache

```python
model.compute_kv_cache(encoder_hidden_states: jax.Array,
	encoder_hidden_states_image: Optional[jax.Array] = None, timestep: Optional[jax.Array] = None,
) -> Tuple[Dict[str, Tuple[jax.Array, jax.Array]], Optional[jax.Array]]
```

Returns cache and an optional conditioning mask.

<details>
<summary>Parameters</summary>

- `encoder_hidden_states` (`jax.Array`). Text conditioning.
- `encoder_hidden_states_image` (`Optional[jax.Array]`, optional, defaults to `None`). Image conditioning.
- `timestep` (`Optional[jax.Array]`, optional, defaults to `None`). Timestep conditioning. Omission uses zeros.

</details>

### text_embedding_shape

```python
WanModel.text_embedding_shape(transformer_config: dict, batch: int, max_sequence_length: int) -> tuple
```

Returns the text-embedding shape.

<details>
<summary>Parameters</summary>

- `transformer_config` (`dict`). Transformer configuration containing `text_dim`.
- `batch` (`int`). Batch size.
- `max_sequence_length` (`int`). Sequence length.

</details>

### train_tflops_per_sample

```python
WanModel.train_tflops_per_sample(transformer_config: dict, latent_shape: tuple, text_len: int) -> float
```

Returns estimated training work in TFLOPs per sample, not measured speed.

<details>
<summary>Parameters</summary>

- `transformer_config` (`dict`). Transformer dimensions and layer count.
- `latent_shape` (`tuple`). Latent dimensions.
- `text_len` (`int`). Text conditioning length.

</details>

### conditional_named_scope

```python
model.conditional_named_scope(name: str)
```

Returns a named-scope context when enabled, otherwise an empty context.

<details>
<summary>Parameters</summary>

- `name` (`str`). Scope name.

</details>

## AutoencoderKLWan

Import from `lilax.generation.wan`. Flax NNX module.

```python
AutoencoderKLWan(rngs: nnx.Rngs, base_dim: int = 96, z_dim: int = 16,
	dim_mult: Optional[List[int]] = None, num_res_blocks: int = 2, attn_scales: Optional[List[float]] = None,
	temperal_downsample: Optional[List[bool]] = None, dropout: float = 0.0,
	latents_mean: Optional[List[float]] = None, latents_std: Optional[List[float]] = None,
	sharding: Sharding | None = None, dtype: jnp.dtype = jnp.float32, weights_dtype: jnp.dtype = jnp.float32,
	precision: jax.lax.Precision = None, vae_decode_chunk: int = 1, vae_encode_chunk: int = 4)
```

Constructs the video autoencoder.

The parameter spelling `temperal_downsample` is part of the API. Chunks must be positive or `-1`.
Finite encoding chunks must match temporal downsampling multiples.

<details>
<summary>Parameters</summary>

- `rngs` (`nnx.Rngs`). Construction RNGs.
- `base_dim` (`int`, optional, defaults to `96`). Base channel width.
- `z_dim` (`int`, optional, defaults to `16`). Latent channels.
- `dim_mult` (`Optional[List[int]]`, optional, defaults to `None`). Channel multipliers. Omission uses `[1, 2, 4, 4]`.
- `num_res_blocks` (`int`, optional, defaults to `2`). Residual blocks.
- `attn_scales` (`Optional[List[float]]`, optional, defaults to `None`). Attention scales. Omission uses `[]`.
- `temperal_downsample` (`Optional[List[bool]]`, optional, defaults to `None`). Temporal downsampling. Omission uses `[False, True, True]`.
- `dropout` (`float`, optional, defaults to `0.0`). Dropout probability.
- `latents_mean` (`Optional[List[float]]`, optional, defaults to `None`). Latent means used instead of bundled values.
- `latents_std` (`Optional[List[float]]`, optional, defaults to `None`). Latent standard deviations used instead of bundled values.
- `sharding` (`Sharding | None`, optional, defaults to `None`). Logical-axis mapping.
- `dtype` (`jnp.dtype`, optional, defaults to `jnp.float32`). Activation dtype.
- `weights_dtype` (`jnp.dtype`, optional, defaults to `jnp.float32`). Weight dtype.
- `precision` (`jax.lax.Precision`, optional, defaults to `None`). Multiplication precision.
- `vae_decode_chunk` (`int`, optional, defaults to `1`). Latent frames per decode chunk or `-1` for unchunked decoding.
- `vae_encode_chunk` (`int`, optional, defaults to `4`). Frames per encode chunk or `-1` for unchunked encoding.

</details>

### latent_shape

```python
AutoencoderKLWan.latent_shape(vae_config: dict, batch: int, num_frames: int, height: int, width: int) -> tuple
```

Returns latent dimensions.

<details>
<summary>Parameters</summary>

- `vae_config` (`dict`). Latent channels and temporal downsampling.
- `batch` (`int`). Batch size.
- `num_frames` (`int`). Input frames.
- `height` (`int`). Input pixel height.
- `width` (`int`). Input pixel width.

</details>

### encode

```python
autoencoder.encode(x: jax.Array, feat_cache: AutoencoderKLWanCache,
	return_dict: bool = True) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]
```

Returns an object with `latent_dist`, or a tuple when `return_dict=False`.

<details>
<summary>Parameters</summary>

- `x` (`jax.Array`). Input video samples.
- `feat_cache` (`AutoencoderKLWanCache`). Mutable feature cache.
- `return_dict` (`bool`, optional, defaults to `True`). Flag for an output record instead of a tuple.

</details>

### decode

```python
autoencoder.decode(z: jax.Array, feat_cache: AutoencoderKLWanCache, return_dict: bool = True)
```

Returns an object with `sample`, or a one-element tuple when `return_dict=False`.

<details>
<summary>Parameters</summary>

- `z` (`jax.Array`). Latent samples.
- `feat_cache` (`AutoencoderKLWanCache`). Mutable feature cache.
- `return_dict` (`bool`, optional, defaults to `True`). Flag for an output record instead of a tuple.

</details>

### AutoencoderKLWanCache

Import from `lilax.generation.wan`. Class.

```python
AutoencoderKLWanCache(module)
```

Creates and initializes VAE feature cache state.

<details>
<summary>Parameters</summary>

- `module` (`AutoencoderKLWan`). The matching Wan autoencoder.

</details>

### init_cache

```python
cache.init_cache()
```

Resets the public VAE cache.

Returns `None`.

## UMT5EncoderModel

Import from `lilax.generation.wan`. Flax NNX module.

```python
UMT5EncoderModel(vocab_size: int = 256384, d_model: int = 4096, d_kv: int = 64, d_ff: int = 10240,
	num_layers: int = 24, num_heads: int = 64, relative_attention_num_buckets: int = 32,
	relative_attention_max_distance: int = 128, layer_norm_epsilon: float = 1e-6,
	dense_act_fn: str = "gelu_new", is_gated_act: bool = True, dtype: jnp.dtype = jnp.float32,
	weights_dtype: jnp.dtype = jnp.float32, precision: Optional[jax.lax.Precision] = None, *,
	rngs: nnx.Rngs, sharding: Sharding | None = None)
```

Constructs the text encoder. Shared module operations follow [Flax NNX](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/module.html).

<details>
<summary>Parameters</summary>

- `vocab_size` (`int`, optional, defaults to `256384`). Vocabulary size.
- `d_model` (`int`, optional, defaults to `4096`). Embedding width.
- `d_kv` (`int`, optional, defaults to `64`). Key/value head width.
- `d_ff` (`int`, optional, defaults to `10240`). Feed-forward width.
- `num_layers` (`int`, optional, defaults to `24`). Encoder layers.
- `num_heads` (`int`, optional, defaults to `64`). Attention heads.
- `relative_attention_num_buckets` (`int`, optional, defaults to `32`). Relative-position buckets.
- `relative_attention_max_distance` (`int`, optional, defaults to `128`). Relative-position maximum distance.
- `layer_norm_epsilon` (`float`, optional, defaults to `1e-6`). Normalization epsilon.
- `dense_act_fn` (`str`, optional, defaults to `"gelu_new"`). Dense activation.
- `is_gated_act` (`bool`, optional, defaults to `True`). Flag for gated activation.
- `dtype` (`jnp.dtype`, optional, defaults to `jnp.float32`). Activation dtype.
- `weights_dtype` (`jnp.dtype`, optional, defaults to `jnp.float32`). Parameter dtype.
- `precision` (`Optional[jax.lax.Precision]`, optional, defaults to `None`). Multiplication precision.
- `rngs` (`nnx.Rngs`). Required keyword-only RNGs.
- `sharding` (`Sharding | None`, optional, defaults to `None`). Axis mapping.

</details>

### Encoder call

```python
text_encoder(input_ids: jax.Array, attention_mask: Optional[jax.Array] = None) -> jax.Array
```

Returns text embeddings.

<details>
<summary>Parameters</summary>

- `input_ids` (`jax.Array`). Token IDs.
- `attention_mask` (`Optional[jax.Array]`, optional, defaults to `None`). Visible tokens. Omission uses an all-visible mask.

</details>
