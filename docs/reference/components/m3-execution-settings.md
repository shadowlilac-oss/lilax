# M3 execution and data settings

Import `DeviceConfig`, `CheckpointConfig`, `DataConfig`, and `TransformerConfig` from `lilax.generation.m3_config`. These are component configurations. They accept extra keys and do not establish that every consumer supports every combination.

## DeviceConfig

Import from `lilax.generation.m3_config`. Pydantic model.

<details>
<summary>Parameters</summary>

- `hardware` (`str`, optional, defaults to `"tpu"`). The accelerator for mesh construction.
- `skip_jax_distributed_system` (`bool`, optional, defaults to `False`). Flag to skip explicit distributed initialization during processing.
- `jax_cache_dir` (`str`, optional, defaults to `""`). The JAX compilation cache directory.
- `mesh_axes` (`list[str]`, optional, defaults to `["data", "fsdp", "context", "tensor"]`). Physical axis names. Each instance receives a fresh value.
- `logical_axis_rules` (`tuple`, optional, defaults to `()`). Mapping of logical tensor axes to mesh axes.
- `vae_logical_axis_rules` (`tuple`, optional, defaults to `()`). Mapping of VAE axes.
- `data_sharding` (`tuple`, optional, defaults to `()`). Input batch partitions.
- `dcn_data_parallelism` (`int`, optional, defaults to `1`). Data parallelism across slices.
- `dcn_fsdp_parallelism` (`int`, optional, defaults to `1`). Parameter sharding across slices.
- `dcn_context_parallelism` (`int`, optional, defaults to `-1`). Context parallelism across slices.
- `dcn_tensor_parallelism` (`int`, optional, defaults to `1`). Tensor parallelism across slices.
- `ici_data_parallelism` (`int`, optional, defaults to `1`). Data parallelism within slices.
- `ici_fsdp_parallelism` (`int`, optional, defaults to `1`). Parameter sharding within slices.
- `ici_context_parallelism` (`int`, optional, defaults to `-1`). Context parallelism within slices.
- `ici_tensor_parallelism` (`int`, optional, defaults to `1`). Tensor parallelism within slices.
- `allow_split_physical_axes` (`bool`, optional, defaults to `False`). Flag for splitting physical axes during mesh construction.
- `compile_topology_num_slices` (`int`, optional, defaults to `-1`). Retained compilation topology slice count.
- `num_slices` (`int`, optional, defaults to `1`). The accelerator slice count.
- `quantization_local_shard_count` (`int`, optional, defaults to `1`). The local quantization shard count.

</details>

## CheckpointConfig

Import from `lilax.generation.m3_config`. Pydantic model.

<details>
<summary>Parameters</summary>

- `run_name` (`str`, optional, defaults to `""`). Run directory name. Processing fills an empty name from `JOBSET_NAME` or the current time.
- `base_output_directory` (`str`, optional, defaults to `""`). The output root.
- `checkpoint_directory` (`str`, optional, defaults to `""`). The fallback output root.
- `output_dir` (`str`, optional, defaults to `""`). The fallback after both other output roots.
- `checkpoint_every` (`int`, optional, defaults to `-1`). Retained component checkpoint interval.
- `checkpoint_dir` (`str`, optional, defaults to `""`). Checkpoint output. Processing derives an empty value under the run directory.
- `tensorboard_dir` (`str`, optional, defaults to `""`). TensorBoard output. Processing derives an empty value under the run directory.
- `metrics_dir` (`str`, optional, defaults to `""`). Metrics output. Processing derives an empty value under the run directory.
- `save_final_checkpoint` (`bool`, optional, defaults to `False`). Retained final checkpoint preference.
- `save_config_to_gcs` (`bool`, optional, defaults to `False`). Retained GCS configuration-writing preference.
- `enable_single_replica_ckpt_restoring` (`bool`, optional, defaults to `False`). Single-replica restoration where supported.
- `aot_cache_dir` (`str`, optional, defaults to `""`). Compiled artifact storage.
- `unet_checkpoint` (`str`, optional, defaults to `""`). Retained UNet checkpoint location.
- `revision` (`str`, optional, defaults to `""`). The repository revision.
- `lightning_from_pt` (`bool`, optional, defaults to `True`). PyTorch Lightning weights in retained metadata.
- `lightning_repo` (`str`, optional, defaults to `""`). Retained Lightning repository.
- `lightning_ckpt` (`str`, optional, defaults to `""`). Retained Lightning filename.

</details>

## DataConfig

Import from `lilax.generation.m3_config`. Pydantic model.

<details>
<summary>Parameters</summary>

- `dataset_name` (`str`, optional, defaults to `""`). Retained dataset identity.
- `dataset_path` (`str`, optional, defaults to `""`). Retained dataset location.
- `train_data_dir` (`str`, optional, defaults to `""`). Retained training input directory.
- `eval_data_dir` (`str`, optional, defaults to `""`). Retained evaluation input directory.
- `per_device_batch_size` (`float`, optional, defaults to `1.0`). Examples per device. Processing derives global load and training sizes from the device count.
- `max_sequence_length` (`int`, optional, defaults to `512`). Limit on the token sequence length.

</details>

## TransformerConfig

Import from `lilax.generation.m3_config`. Pydantic model.

`BaseConfig` composes device, checkpoint, trainer, and data defaults. Its direct constructor defaults `transformer=None`. `BaseConfig.from_dict` constructs a transformer domain from the mapping as well. The [M3 construction reference](m3-components.md) describes mapping access and explicit runtime processing.

See [configure components](../../guides/components/m3-settings.md).

<details>
<summary>Parameters</summary>

- `dim` (`Optional[int]`, optional, defaults to `None`). Hidden width.
- `base_emb_dim` (`Optional[int]`, optional, defaults to `None`). Base embedding width.
- `num_layers` (`Optional[int]`, optional, defaults to `None`). Layer count.
- `base_num_decoder_layers` (`Optional[int]`, optional, defaults to `None`). Base decoder layer count.
- `num_heads` (`Optional[int]`, optional, defaults to `None`). Query-head count.
- `base_num_query_heads` (`Optional[int]`, optional, defaults to `None`). Base query-head count.
- `num_kv_heads` (`Optional[int]`, optional, defaults to `None`). Key and value head count.
- `base_num_kv_heads` (`Optional[int]`, optional, defaults to `None`). Base key and value head count.
- `head_dim` (`Optional[int]`, optional, defaults to `None`). Head width.
- `mlp_dim` (`Optional[int]`, optional, defaults to `None`). Feed-forward width.
- `base_mlp_dim` (`Optional[int]`, optional, defaults to `None`). Base feed-forward width.
- `weights_dtype` (`Any`, optional, defaults to `"bfloat16"`). Stored weight dtype.
- `activations_dtype` (`Any`, optional, defaults to `"bfloat16"`). Activation dtype.
- `attention` (`str`, optional, defaults to `"flash"`). Attention computation.
- `precision` (`str`, optional, defaults to `"DEFAULT"`). Matrix multiplication precision.
- `scan_layers` (`bool`, optional, defaults to `False`). Flag for repeated-layer scanning.
- `dropout` (`float`, optional, defaults to `0.0`). Dropout probability.
- `split_head_dim` (`bool`, optional, defaults to `True`). Flag for separate head-count and head-width dimensions.
- `norm_num_groups` (`int`, optional, defaults to `32`). Normalization group count.
- `use_base2_exp` (`bool`, optional, defaults to `True`). Base-two exponentials in compatible kernels.

</details>
