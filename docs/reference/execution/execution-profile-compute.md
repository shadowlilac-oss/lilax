# Execution and memory profile fields

These optional YAML sections belong to an [execution profile](execution-profile.md). Only explicitly supplied keys override the workflow.

## `execution`

Controls queued work and preparation before timing. Colocated Python input requires single-controller Pathways.

<details>
<summary>Parameters</summary>

- `use_tunix_gradient_accumulation` (`bool`, optional, defaults to `false`). Use Tunix accumulation in supported training paths.
- `max_inflight_computations` (`int`, optional, defaults to `2`). Maximum queued device computations.
- `block_state_before_run_start` (`bool`, optional, defaults to `false`). Wait for state, compile the learning-rate schedule's eager operations, and synchronize hosts before timing. Does not access data.
- `warm_input_reshard_before_run_start` (`bool`, optional, defaults to `false`). Compile training input resharding with an all-zero synthetic batch before timing. Does not read data.
- `warm_eval_input_reshard_before_run_start` (`bool`, optional, defaults to `false`). Compile evaluation input resharding with a synthetic batch before timing. Does not read evaluation data.
- `colocated_python_data_input` (`bool`, optional, defaults to `false`). Place Python data loading on remote TPU VMs.

</details>

## `precision`

The dtype enum accepts `bfloat16`, `float32`, `float16`, `float8_e4m3fn`, and `float8_e5m2`. Acceptance does not make every dtype appropriate for every kernel.

<details>
<summary>Parameters</summary>

- `dtype` (`DType`, optional, defaults to `"bfloat16"`). Activation and compute dtype.
- `grad_dtype` (`DType`, optional, defaults to `"float32"`). Gradient dtype.
- `weight_dtype` (`DType`, optional, defaults to `"float32"`). Stored weight dtype.
- `matmul_precision` (`MatmulPrecision`, optional, defaults to `"default"`). `default`, `high`, `highest`, `bfloat16`, or `float32`.
- `activations_in_float32` (`bool`, optional, defaults to `false`). Cast activations to float32 before the nonlinearity.
- `dtype_mm` (`str`, optional, defaults to `"float32"`). Vision-encoder dtype.

</details>

## `transport`

Enabled TE overlap requires `hardware=gpu_multiprocess`, one local device per process, initialized distributed execution, tensor-sequence parallelism greater than one, tensor parallelism equal to one, and a `te_` quantization recipe. The shipped profile launcher accepts only `cpu`, `gpu`, and `tpu`, so it cannot select that hardware mode. The accepted profile key does not make enabled TE overlap usable through that launcher.

<details>
<summary>Parameters</summary>

- `replicate_quant_scale` (`bool`, optional, defaults to `false`). Replicate quantization scales instead of sharding.
- `te_comm_gemm_overlap` (`str`, optional, defaults to `"disabled"`). `disabled` turns overlap off. `mlp` overlaps MLP communication. `full` also includes supported attention communication on a compatible TE path.

</details>

## `compilation`

Controls ahead-of-time compilation inputs and estimator output.

<details>
<summary>Parameters</summary>

- `compiled_trainstep_file` (`str`, optional, defaults to `""`). Serialized compiled training-step path.
- `compile_topology` (`str`, optional, defaults to `""`). Target topology name.
- `compile_topology_num_slices` (`int`, optional, defaults to `-1`). Target slice count. `-1` is unset.
- `write_estimator_result` (`bool`, optional, defaults to `false`). Write a separate result file when running the estimator.

</details>

## `remat`

Rematerialization recomputes intermediates during the backward pass to reduce retained device memory. Location fields accept `remat`, `device`, or `offload`. These respectively recompute the named value, retain it on the device, or offload it to host memory. A custom policy selects locations individually.

Named policies are `full`, `minimal`, `minimal_with_context`, `minimal_with_quantization`, `minimal_with_context_and_quantization`, `save_dot_with_context_except_mlp`, `save_dot_except_mlpwi`, `save_dot_except_mlp`, `save_qkv_proj`, `qkv_proj_offloaded`, `custom`, `minimal_offloaded`, and `save_out_proj`. `minimal_flash` is a deprecated alias of `minimal_with_context`. `null`, an empty string, or `"none"` leaves no selective save policy.

`full` recomputes the layer. `minimal` retains named attention projections and MLP intermediates. The context and quantization variants also retain those named values. `save_dot_with_context_except_mlp` retains attention projections and context. `save_dot_except_mlpwi` retains attention projections and MLP output. `save_dot_except_mlp` retains only attention projections. `save_qkv_proj` retains query, key, value, unified KV, and fused QKV projections. `save_out_proj` retains only the attention output projection. `qkv_proj_offloaded` offloads separate Q, K, V, and unified KV projections to pinned host memory. `minimal_offloaded` offloads the minimal policy's projections and MLP intermediates. `custom` uses the per-value location choices.

Optimizer host offload is unsupported on RL. Retaining gathered FSDP weights conflicts with parameter host offload. The custom-policy resolver does not include `engram` in its saved or offloaded tensor lists.

<details>
<summary>Parameters</summary>

- `remat_policy` (`str | None`, optional, defaults to `"full"`). Decoder policy.
- `remat_policy_for_vit` (`str | None`, optional, defaults to `"minimal"`). Vision-encoder policy.
- `decoder_layer_input` (`RematLocation`, optional, defaults to `"device"`). Decoder input residual.
- `indexer_cutoff_threshold` (`RematLocation`, optional, defaults to `"remat"`). Indexer cutoff.
- `context` (`RematLocation`, optional, defaults to `"remat"`). Attention context.
- `mlpwi` (`RematLocation`, optional, defaults to `"remat"`). First MLP intermediate.
- `mlpwi_0` (`RematLocation`, optional, defaults to `"remat"`). First gated MLP branch.
- `mlpwi_1` (`RematLocation`, optional, defaults to `"remat"`). Second gated MLP branch.
- `mlpwo` (`RematLocation`, optional, defaults to `"remat"`). Second MLP output.
- `moe_x_sorted` (`RematLocation`, optional, defaults to `"remat"`). Expert-sorted input and routing metadata. Retention avoids repeated dispatch and sorting.
- `moe_routing_maps` (`RematLocation`, optional, defaults to `"remat"`). Expert IDs, permutations, token indices, and group sizes.
- `moe_tc_routing_checkpoint` (`RematLocation`, optional, defaults to `"remat"`). TC row routing weights. Retention avoids the backward weight gather.
- `te_moe_dispatch_remat_location` (`RematLocation`, optional, defaults to `"remat"`). TE expert-dispatch outputs.
- `moe_mlpwi_0` (`RematLocation`, optional, defaults to `"remat"`). First gated MoE intermediate.
- `moe_mlpwi_1` (`RematLocation`, optional, defaults to `"remat"`). Second gated MoE intermediate.
- `moe_mlpwo` (`RematLocation`, optional, defaults to `"remat"`). MoE output.
- `query_proj` (`RematLocation`, optional, defaults to `"remat"`). Separate Q projection.
- `key_proj` (`RematLocation`, optional, defaults to `"remat"`). Separate K projection.
- `value_proj` (`RematLocation`, optional, defaults to `"remat"`). Separate V projection.
- `kv_proj` (`RematLocation`, optional, defaults to `"remat"`). Unified KV projection.
- `qkv_proj` (`RematLocation`, optional, defaults to `"remat"`). Fused QKV projection.
- `query_wa_proj` (`RematLocation`, optional, defaults to `"remat"`). MLA query weighted-attention projection.
- `kv_wa_proj` (`RematLocation`, optional, defaults to `"remat"`). MLA KV weighted-attention projection.
- `out_proj` (`RematLocation`, optional, defaults to `"remat"`). Attention projection.
- `attention_out` (`RematLocation`, optional, defaults to `"remat"`). Attention output.
- `mla_q` (`RematLocation`, optional, defaults to `"remat"`). MLA query intermediate.
- `mla_kv` (`RematLocation`, optional, defaults to `"remat"`). MLA KV intermediate.
- `engram` (`RematLocation`, optional, defaults to `"remat"`). Accepted Engram location field with the resolver limitation above.
- `optimizer_memory_host_offload` (`bool`, optional, defaults to `false`). Keep optimizer state on the host.
- `parameter_memory_host_offload` (`bool`, optional, defaults to `false`). Keep parameters on the host.

</details>

## `diloco`

When the sync period is not divisible by the fragment count, streaming uses `max(1, round(period / fragments))` steps between fragment syncs and emits a warning. Traffic shaping changes the selected network interface and requires deployment privileges for that operation.

Streaming requires scanned layers and a decoder-layer count divisible by `num_diloco_fragments - 1`. Non-scanned bucketization does not support optimizer sharding over data.

Streaming requires DiLoCo and at least two fragments. Non-scanned bucketization requires streaming and at least three fragments.

<details>
<summary>Parameters</summary>

- `enable_diloco` (`bool`, optional, defaults to `false`). Enable outer synchronization.
- `diloco_sync_period` (`int`, optional, defaults to `36`). Local steps between outer syncs.
- `diloco_outer_lr` (`float`, optional, defaults to `0.3`). Outer learning rate.
- `diloco_outer_momentum` (`float`, optional, defaults to `0.9`). Outer momentum.
- `dcn_bandwidth_limit` (`str`, optional, defaults to `""`). DCN egress limit per VM. Empty applies no limit.
- `dcn_bandwidth_burst` (`str`, optional, defaults to `"10mb"`). Traffic-shaping token-bucket burst.
- `dcn_bandwidth_latency` (`str`, optional, defaults to `"50ms"`). Traffic-shaping latency.
- `dcn_bandwidth_interface` (`str`, optional, defaults to `"eth0"`). Affected network interface.
- `enable_streaming_diloco` (`bool`, optional, defaults to `false`). Pipeline fragment sync across local computation.
- `num_diloco_fragments` (`int | None`, optional, defaults to `null`). Total fragments including one non-scanned fragment.
- `use_sequential_layers` (`bool`, optional, defaults to `false`). Sync sequential instead of interleaved layers.
- `diloco_bucketize_non_scanned` (`bool`, optional, defaults to `false`). Split eligible matrices among fragments to spread communication.
- `num_communication_overlapping_steps` (`NonNegativeInt`, optional, defaults to `0`). Communication steps overlapped with computation.
- `communication_overlapping_alpha` (`float`, optional, defaults to `0.0`). Interpolate fragment parameters in `[0, 1]`. `1` retains local parameters, `0` discards inner updates during overlap, `0.5` averages.

</details>
