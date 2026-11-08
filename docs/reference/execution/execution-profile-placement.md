# Placement and sharding profile fields

These optional YAML sections belong to an [execution profile](execution-profile.md). Section defaults below describe declarations, not overrides added for omitted keys.

## `mesh`

The default `mesh_axes` is `[diloco, data, stage, fsdp, fsdp_transpose, context, context_usp_ulysses, context_autoregressive, tensor, tensor_sequence, expert, autoregressive]`.

Mesh-and-rule presets are `""`, `pure-fsdp`, `cp-as-ep`, `ep-as-cp`, `pipeline-large-moe`, `2d-fsdp`, `ep-as-dp`, `shard-exp-on-fsdp`, `expert-on-fsdp-ep`, `fsdp-as-dp-for-attn`, and `fsdp-as-dp-for-attn-rf`. These replace placement rules for the named parallelism arrangement. They must match the model and available mesh.

Without an evaluation preset, evaluation inherits training rules. With an evaluation preset, physical axes absent from the training mesh are ignored and their evaluation targets become replicated. Explicit sharding rejects multimodal models. For Qwen3.5 and Qwen3-Next it also rejects context parallelism and requires sparse matmul when using experts.

Hardware must match `process.hardware`, which accepts `cpu`, `gpu`, or `tpu`. The domain's `gpu_multiprocess` value cannot pass that profile constraint. PRNG and unit-axis settings change process-wide JAX configuration.

<details>
<summary>Parameters</summary>

- `hardware` (`Literal["tpu", "gpu", "gpu_multiprocess", "cpu"]`, optional, defaults to `"gpu"`). Hardware kind subject to the process constraint above.
- `num_slices` (`int`, optional, defaults to `-1`). Slice count. `-1` derives it.
- `mesh_axes` (`list[str]`, optional, defaults to `[diloco, data, stage, fsdp, fsdp_transpose, context, context_usp_ulysses, context_autoregressive, tensor, tensor_sequence, expert, autoregressive]`). Physical dimension names and order. A factory creates a new list.
- `shard_mode` (`ShardMode`, optional, defaults to `"auto"`). `auto` infers sharding. `explicit` uses explicit output shardings in supported operations.
- `inhomogeneous_layer_cycle_interval` (`int`, optional, defaults to `1`). Heterogeneous layer-pattern repetition interval.
- `scan_layers` (`bool`, optional, defaults to `true`). Stacked-layer scanning. Checkpoint metadata selects this on resume.
- `param_scan_axis` (`int`, optional, defaults to `1`). Stacked parameter axis.
- `context_parallel_load_balance` (`bool`, optional, defaults to `true`). Context-work balancing.
- `context_parallel_strategy` (`str`, optional, defaults to `"all_gather"`). `all_gather`, `ring`, `ulysses`, or `usp`.
- `context_parallel_reorder_strategy` (`ReorderStrategy`, optional, defaults to `"auto"`). `auto`, `dual_chunk_swap`, or `striped`.
- `custom_mesh` (`str`, optional, defaults to `""`). `hybrid_ring_64x4` or `hybrid_ring_32x8` when selected.
- `custom_mesh_and_rule` (`CustomRule`, optional, defaults to `""`). Training preset from the list above.
- `custom_mesh_and_rule_for_eval` (`CustomRule`, optional, defaults to `""`). Evaluation preset from the same list.
- `allow_split_physical_axes` (`bool`, optional, defaults to `false`). Physical-axis splitting.
- `optimize_mesh_for_tpu_v6e` (`bool`, optional, defaults to `false`). TPU v6e mesh transformations.
- `shardy` (`bool`, optional, defaults to `true`). Shardy XLA partitioner selection.
- `prng_impl` (`Literal["unsafe_rbg", "rbg", "threefry2x32"]`, optional, defaults to `"unsafe_rbg"`). Training PRNG.
- `remove_size_one_mesh_axis_from_type` (`bool`, optional, defaults to `true`). Unit-axis removal from types.

</details>

## `ici` and `dcn`

Each key below is an integer degree. ICI places work within a slice. DCN places work across slices. Degrees must be positive or `-1`, and at most one explicitly selected degree per section may be `-1`. That degree consumes the remaining devices or slices after the other axes. The product must fit the available mesh.

<details>
<summary>Parameters</summary>

- `ici_diloco_parallelism` (`int`, optional, defaults to `1`). Within-slice independent DiLoCo groups.
- `dcn_diloco_parallelism` (`int`, optional, defaults to `1`). Across-slice independent DiLoCo groups.
- `ici_data_parallelism` (`int`, optional, defaults to `1`). Within-slice replicated data parallelism.
- `dcn_data_parallelism` (`int`, optional, defaults to `-1`). Across-slice replicated data parallelism.
- `ici_fsdp_parallelism` (`int`, optional, defaults to `-1`). Within-slice fully sharded parameters.
- `dcn_fsdp_parallelism` (`int`, optional, defaults to `1`). Across-slice fully sharded parameters.
- `ici_fsdp_transpose_parallelism` (`int`, optional, defaults to `1`). Within-slice transposed FSDP dimension.
- `dcn_fsdp_transpose_parallelism` (`int`, optional, defaults to `1`). Across-slice transposed FSDP dimension.
- `ici_sequence_parallelism` (`int`, optional, defaults to `1`). Within-slice sequence parallelism.
- `dcn_sequence_parallelism` (`int`, optional, defaults to `1`). Across-slice sequence parallelism.
- `ici_context_parallelism` (`int`, optional, defaults to `1`). Within-slice attention context partitions.
- `dcn_context_parallelism` (`int`, optional, defaults to `1`). Across-slice attention context partitions.
- `ici_context_usp_ulysses_parallelism` (`int`, optional, defaults to `1`). Within-slice USP all-to-all head exchange.
- `dcn_context_usp_ulysses_parallelism` (`int`, optional, defaults to `1`). Across-slice USP all-to-all head exchange.
- `ici_context_autoregressive_parallelism` (`int`, optional, defaults to `1`). Within-slice autoregressive context dimension.
- `dcn_context_autoregressive_parallelism` (`int`, optional, defaults to `1`). Across-slice autoregressive context dimension.
- `ici_tensor_parallelism` (`int`, optional, defaults to `1`). Within-slice tensor partitions.
- `dcn_tensor_parallelism` (`int`, optional, defaults to `1`). Across-slice tensor partitions.
- `ici_tensor_sequence_parallelism` (`int`, optional, defaults to `1`). Within-slice tensor sequence dimension.
- `dcn_tensor_sequence_parallelism` (`int`, optional, defaults to `1`). Across-slice tensor sequence dimension.
- `ici_pipeline_parallelism` (`int`, optional, defaults to `1`). Within-slice pipeline stages.
- `dcn_pipeline_parallelism` (`int`, optional, defaults to `1`). Across-slice pipeline stages.
- `ici_expert_parallelism` (`int`, optional, defaults to `1`). Within-slice MoE expert partitions.
- `dcn_expert_parallelism` (`int`, optional, defaults to `1`). Across-slice MoE expert partitions.
- `ici_autoregressive_parallelism` (`int`, optional, defaults to `1`). Within-slice autoregressive execution partitions.
- `dcn_autoregressive_parallelism` (`int`, optional, defaults to `1`). Across-slice autoregressive execution partitions.

</details>

## `pipeline`

Controls pipeline scheduling. Per-repeat phase splitting changes summation order. Storing prefetch buffers in compute dtype changes numerics.

<details>
<summary>Parameters</summary>

- `pipeline_fsdp_ag_per_repeat` (`bool`, optional, defaults to `false`). Circular-repeat weight prefetch.
- `pipeline_bsw_no_carry` (`bool`, optional, defaults to `true`). Previous-repeat weight regathering instead of buffer retention with per-repeat prefetch. Reduces retained weights and adds collectives.
- `pipeline_bsw_phase_split` (`bool`, optional, defaults to `false`). Buffer-slot selection avoidance for current-repeat weights with per-repeat prefetch.
- `pipeline_bsw_in_compute_dtype` (`bool`, optional, defaults to `false`). Weight and gradient-accumulation storage in `dtype` instead of `weight_dtype` with per-repeat prefetch.
- `num_layers_per_pipeline_stage` (`int`, optional, defaults to `1`). Layers per stage.
- `num_pipeline_repeats` (`int`, optional, defaults to `-1`). Repeat count. `-1` derives it.
- `pipeline_parallel_layers` (`int`, optional, defaults to `-1`). Layers to pipeline. `-1` selects all decoder layers.
- `num_pipeline_microbatches` (`int`, optional, defaults to `-1`). Microbatches. `-1` uses the stage count.
- `pipeline_delay_activation_forwarding` (`bool`, optional, defaults to `false`). Activation-forwarding delay for compiler scheduling.
- `pipeline_fsdp_ag_once` (`bool`, optional, defaults to `false`). Once-per-repeat FSDP weight gathering.
- `scan_pipeline_iterations` (`bool`, optional, defaults to `true`). Pipeline iteration scanning.
- `scan_pipeline_repeats` (`bool`, optional, defaults to `false`). Pipeline repeat scanning.
- `scan_layers_per_stage` (`bool`, optional, defaults to `false`). Within-stage layer scanning.
- `set_remat_policy_on_pipeline_iterations` (`bool`, optional, defaults to `true`). Rematerialization policy on the iteration scan.
- `set_remat_policy_on_layers_per_stage` (`bool`, optional, defaults to `false`). Rematerialization policy on the layer scan.

</details>

## `sharding`

Rule fields accept YAML sequences of `[logical_axis, physical_axes]` pairs. A physical target can be a string or a list. An empty list means replication. Rule ordering matters when a logical axis has alternatives.

Variation checks require auto sharding, Megablox, and expert or FSDP parallelism. FSDP prefetch requires scanned layers and has no effect without a multi-device FSDP axis. Retained kernels and deferred gradients increase device-memory use. Internal compilation requires its corresponding compilation environment.

<details>
<summary>Parameters</summary>

- `logical_axis_rules` (`list | tuple`, optional, defaults to the built-in [rule list](execution-profile-sharding-rules.md)). Ordered logical-axis and mesh-axis pairs with axis-name strings or sequences as targets. Model preparation and presets can change these defaults.
- `logical_axis_rules_for_eval` (`Any`, optional, defaults to `[]`). Evaluation rule sequence.
- `data_sharding` (`Any`, optional, defaults to `[[data, stage, fsdp, fsdp_transpose, context, context_usp_ulysses, context_autoregressive, tensor, tensor_sequence, expert, autoregressive]]`). Physical input-data sharding sequence.
- `context_sharding` (`str`, optional, defaults to `"context"`). Context axis, including USP's ring dimension.
- `ulysses_context_sharding` (`str`, optional, defaults to `"context_usp_ulysses"`). USP all-to-all axis.
- `input_data_sharding_logical_axes` (`list[str]`, optional, defaults to `[activation_embed_and_logits_batch, activation_norm_length]`). Input logical axes.
- `sharding_tolerance` (`float`, optional, defaults to `0.02`). Allowed unsharded parameter fraction in `[0, 1]`.
- `check_vma` (`bool`, optional, defaults to `false`). Variation checks in shard maps.
- `shard_optimizer_over_data` (`bool`, optional, defaults to `false`). ZeRO-1 optimizer sharding over data.
- `dense_fsdp_use_two_stage_all_gather` (`bool`, optional, defaults to `false`). Separate dense MLP weight gathers over FSDP and FSDP-transpose.
- `dense_weight_grad_in_kernel_order` (`bool | None`, optional, defaults to `null`). Projection-gradient transpose folding under explicit sharding. Null enables where applicable. No effect under auto sharding or for quantized layers.
- `save_fsdp_gathered_weights` (`bool`, optional, defaults to `false`). Gathered-kernel retention for backward computation.
- `internal_compile` (`bool`, optional, defaults to `false`). Open-source topology mapping bypass for internal compilation.
- `internal_compile_num_devices` (`int`, optional, defaults to `-1`). Internal compile device count.
- `compile_xla_flags` (`str`, optional, defaults to `""`). Compilation-only XLA options.
- `prefetch_fsdp_weights` (`bool`, optional, defaults to `false`). Next-layer weight gathering during current-layer work.
- `prefetch_fsdp_defer_grad_reduce_scatter` (`bool`, optional, defaults to `false`). Gradient reduce-scatter deferral to the next backward layer using XLA FSDP scheduling with prefetch and explicit sharding. Retains an extra layer of unreduced gradients.

</details>
