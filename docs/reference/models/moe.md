# Expert routing and kernels

Return to the [configuration reference](configuration.md).

Import these configurations from `lilax.configuration`. Keep settings loaded with the matching model family.
See [model configuration](../../guides/models/configuration.md) for loading and updating a focused domain.
The keyword fields below show constructor defaults. Loaded checkpoints can supply different values.
Changing architecture dimensions or routing does not convert existing weights.

## MoEParameters

Import from `lilax.configuration`. Pydantic model with keyword fields.

Configures expert routing and execution. Effective expert MLP dimensions belong to the model configuration.
The inherited `base_moe_mlp_dim` class variable is excluded from constructor fields.

**Dropless fallback prerequisites.**

<a id="dropless-fallback-prerequisites"></a>

For more than one expert, both fallback modes require `ragged_buffer_factor > 0`, `use_ragged_sort=True`, and
`KernelConfig.moe.num_moe_emb_chunks=0`. `step` additionally requires ring of experts, DiLoCo disabled,
no precompiled train-step file, and no optimizer or parameter host offload. `layer` also supports ragged
all-to-all without ring of experts, but rejects random routing and deferred routed-bias reductions.

Positive ring-path ragged-buffer sizing requires ragged sort. TE MoE requires a training buffer factor of at least
one or a nonpositive value, and no positive evaluation factor. Layer fallback uses the non-TE sparse path.
Token chunking above one requires ring of experts. Only one of expert-on-FSDP, two-dimensional FSDP, or
embedding-on-FSDP sharding may be enabled. SparseCore FSDP pinning cannot combine with two-stage all-gather.
Grouped GEMM kernel compatibility and quantization constraints depend on the selected hardware.

Chunk weight-gradient accumulation requires Tokamax GMM v2, no embedding chunks, and unfused expert weights.
Input-gradient accumulation also requires Tokamax GMM v2 and unfused expert weights. Flat FSDP weights require
fixed-scale quantization, embedding sharding, and GMM v2. Full-state resumes require matching flat checkpoints.
Backward prequantization requires fixed weight, activation, and backward calibration. Direct QArray combine
requires that prequantization and matching scales across shards, and rejects combine-backward quantization.
Token all-gather quantization requires sparse ring execution, GMM v2, fixed activation scales, and no embedding chunks.

TensorCore ragged sorting requires ring execution, ragged sort, and a positive training buffer factor.
Activation-side routing weights and window routing require TensorCore sorting. Three-dimensional GMM needs
TensorCore sorting and aligned embedding tiles. Three-dimensional dispatch additionally requires embedding width
divisible by 128. Float32 router computation cannot combine with router-projection quantization.

<details>
<summary>Parameters</summary>

- `num_experts` (`PositiveInt`, optional, defaults to `1`). Experts per layer.
- `num_experts_per_tok` (`PositiveInt`, optional, defaults to `1`). Experts selected per token.
- `capacity_factor` (`float`, optional, defaults to `-1.0`). Expert capacity multiplier. Negative disables capacity-based dropping.
- `ragged_buffer_factor` (`float`, optional, defaults to `-1.0`). Training buffer multiplier. Negative allocates worst-case capacity. Positive buffers can overflow independently of `capacity_factor`.
- `eval_ragged_buffer_factor` (`float`, optional, defaults to `-1.0`). Evaluation buffer multiplier under evaluation axis rules. Negative uses worst-case capacity.
- `moe_dropless_fallback` (`Literal["step", "layer"] | None`, optional, defaults to `None`). On overflow, `None` drops tokens, `step` rolls back and replays the whole step dropless, and `layer` uses a dropless branch only for that layer. Requires the prerequisites above.
- `retry_dropless_first_steps` (`int`, optional, defaults to `0`). Initial steps after loop start run directly dropless with step fallback. Zero disables warmup.
- `first_phase_ragged_buffer_factor` (`float`, optional, defaults to `0.0`). Buffer factor for the first N retry steps. Zero disables it. Positive values must be at least the normal factor. Overflowing steps replay dropless.
- `log_required_ragged_buffer_factor` (`bool`, optional, defaults to `False`). Minimum drop-free buffer factor logging per layer and step. Adds cross-shard reductions and a host fetch.
- `warmup_programs_in_init` (`bool`, optional, defaults to `False`). Precompiled training and evaluation warmup on synthetic batches before run timing, with discarded results. Skipped without precompiled programs.
- `warmup_skip_train_dropless` (`bool`, optional, defaults to `False`). Dropless training warmup omission, avoiding its extra state copy. Requires warmup enabled. First dropless execution then pays startup cost.
- `num_moe_token_chunks` (`PositiveInt`, optional, defaults to `1`). Ring pipeline token chunks. One disables chunking.
- `moe_accumulate_chunk_wgrad` (`bool`, optional, defaults to `False`). Gathered weight reuse and weight-gradient accumulation across token chunks. Requires Tokamax GMM v2, no embedding chunks, and unfused expert weights.
- `moe_accumulate_wi_dlhs` (`bool`, optional, defaults to `False`). In-place input-projection gradient accumulation. Requires Tokamax GMM v2 and unfused expert weights.
- `moe_flat_fsdp_weights` (`bool`, optional, defaults to `False`). `[experts * rows, columns]` weight storage over expert and FSDP axes. Requires fixed-scale quantization, embedding sharding, and Tokamax GMM v2. Full-state resumes require flat-layout checkpoints.
- `moe_gmm_v2_dlhs_transpose_rhs` (`bool`, optional, defaults to `False`). Native `transpose_rhs` for backward DLHS instead of explicit RHS transpose.
- `moe_bwd_prequant_before_unsort` (`bool`, optional, defaults to `False`). Output gradient quantization before expansion into expert order. Quantized execution requires fixed weight, activation, and backward calibration.
- `moe_combine_bwd_direct_qarray` (`bool`, optional, defaults to `False`). Local output gradient quantization before expert all-gather. Requires prequantization before unsort and matching fixed scales across shards. Incompatible with combine-backward quantization.
- `moe_3d_dlhs_partial_sum` (`bool`, optional, defaults to `False`). Omitted zero-initialization of 3D GMM v2 input gradients during accumulation onto a partial gradient sum.
- `moe_chunk_barrier` (`bool`, optional, defaults to `False`). Diagnostic serialization of ring token chunks. Effective only with ring of experts and more than one chunk.
- `moe_log_max_load_ratio` (`bool`, optional, defaults to `False`). Maximum and mean expert-shard load ratio logging. Requires sparse matmul, ring of experts, ragged sort, and TE MoE disabled.
- `moe_expert_input_dim` (`int`, optional, defaults to `-1`). Expert input width. Negative uses embedding width.
- `padded_base_moe_mlp_dim` (`int | None`, optional, defaults to `None`). Padded expert intermediate width for GMM v2.
- `load_balance_loss_weight` (`NonNegativeFloat`, optional, defaults to `0.0`). Load-balancing auxiliary loss weight.
- `moe_use_megatron_seq_aux_loss` (`bool`, optional, defaults to `False`). Sequence-wise balancing for non-TE sigmoid or sqrt-softplus routing. `False` uses layer-averaged Switch-style loss.
- `use_custom_sort_vjp` (`bool`, optional, defaults to `True`). Custom sparse-matmul sort VJP.
- `router_topk_matmul_vjp` (`bool`, optional, defaults to `False`). Dense one-hot routing-weight gradient reduction instead of scatter accumulation. Forward routing is unchanged.
- `moe_topk_before_ep_all_gather` (`bool`, optional, defaults to `False`). Local top-k selection before routing all-gather. Requires sparse ring execution. Ignored for forced, hash, or random routing.
- `use_ring_of_experts` (`bool`, optional, defaults to `False`). Ring-of-experts sparse expert parallelism.
- `te_moe_block` (`bool`, optional, defaults to `False`). Transformer Engine fused routing, dispatch, grouped GEMM, and combine.
- `te_moe_disable_cudnn_gmm_fusion` (`bool`, optional, defaults to `False`). Unfused cuDNN grouped matmuls. `False` retains the TE default.
- `te_moe_quantize_before_fsdp_all_gather` (`bool`, optional, defaults to `False`). Weight shard quantization before FSDP gathering instead of after.
- `te_ep_overflow_check_every_n_steps` (`PositiveInt`, optional, defaults to `20`). Buffered steps between host TE receive-capacity overflow checks. Overflowing steps still skip optimizer updates immediately on device.
- `te_gmm_quantization` (`TEGroupedGemmQuantizationType | None`, optional, defaults to `TEGroupedGemmQuantizationType.EMPTY`). TE grouped GEMM quantization.
- `moe_quantize_token_all_gather` (`bool`, optional, defaults to `False`). Token quantization before expert gathering. Requires sparse ring execution, GMM v2, fixed activation calibration, and no embedding chunks.
- `moe_dispatch_no_expert_sharding` (`bool`, optional, defaults to `False`). Dispatch and MLP batch sharding without `expert`, keeping GEMM expert-parallel through AllToAll. `False` keeps `expert` in `activation_batch_moe`.
- `use_ragged_sort` (`bool`, optional, defaults to `False`). Ragged sorting kernel for expert parallelism.
- `moe_quantize_combine_bwd_method` (`str | None`, optional, defaults to `""`). Combine-gradient format. Empty uses BF16, `rowwise` uses token-wise FP8 E5M2, and `fixed,<bound>` uses a fixed FP8 bound.
- `ragged_sort_use_single_sparsecore` (`bool`, optional, defaults to `False`). Single SparseCore execution instead of all SparseCores.
- `moe_tc_ragged_sort` (`bool`, optional, defaults to `False`). TensorCore Pallas sorting instead of SparseCore. Requires ring of experts, ragged sort, and positive training buffer factor.
- `moe_tc_ragged_gather_block_size` (`int`, optional, defaults to `1024`). Gather rows per grid step.
- `moe_tc_ragged_reduce_block_size` (`int`, optional, defaults to `896`). Gather-reduce VMEM rows per block.
- `moe_tc_ragged_mask_padding` (`bool`, optional, defaults to `True`). Zero-filled buffer rows past the valid count, otherwise uninitialized by the kernel.
- `moe_tc_ragged_weights_on_activation` (`bool`, optional, defaults to `False`). Routing weights applied before output projection instead of to expert output. Requires TensorCore ragged sort.
- `moe_tc_ragged_flatten_block_size` (`int`, optional, defaults to `0`). Pallas layout-conversion rows per block around TensorCore kernels. Zero uses XLA reshapes.
- `moe_tc_ragged_3d_gmm` (`bool`, optional, defaults to `False`). `[buffer, embedding // 128, 128]` layout for Tokamax GMM v2. Requires TensorCore sort and aligned embedding tiles.
- `moe_tc_ragged_3d_dispatch` (`bool`, optional, defaults to `False`). 3D dispatch and combine layout. Requires 3D GMM, TensorCore sort, and embedding width divisible by 128.
- `moe_tc_window_routing` (`bool`, optional, defaults to `False`). Local expert shard's routing window only, reducing combine metadata. Requires TensorCore sort.
- `moe_use_direct_token_gather` (`bool`, optional, defaults to `False`). Expert-order gathering without materializing top-k copies.
- `use_gather_mosaic_kernel` (`bool`, optional, defaults to `False`). Custom Mosaic token-gather kernel.
- `ragged_gather_fallback` (`bool`, optional, defaults to `False`). Unconditional JAX reference gather when true. `False` uses SparseCore gather.
- `ragged_gather_reduce_fallback` (`bool`, optional, defaults to `False`). Unconditional JAX reference gather-reduce when true. `False` uses SparseCore gather-reduce.
- `ragged_gather_cost_estimate_flops` (`int`, optional, defaults to `-1`). Gather flop estimate. `-1` computes automatically. Positive values override it.
- `ragged_gather_reduce_cost_estimate_flops` (`int`, optional, defaults to `-1`). Gather-reduce flop estimate with the same override rules.
- `ragged_gather_cost_estimate_bytes_accessed` (`int`, optional, defaults to `-1`). Gather byte-access estimate. `-1` computes automatically. Positive values override it.
- `ragged_gather_reduce_cost_estimate_bytes_accessed` (`int`, optional, defaults to `-1`). Gather-reduce byte-access estimate with the same override rules.
- `moe_pin_sparse_core_all_gathers` (`bool`, optional, defaults to `False`). Dedicated SparseCore placement for MoE FSDP and EP all-gathers through `compute_on`.
- `moe_pin_sparse_core_ep_all_gathers` (`bool`, optional, defaults to `False`). Dedicated SparseCore placement for ring EP dispatch and routing all-gathers only. OR-ed with the all-gathers flag.
- `moe_pin_sparse_core_fsdp_all_gathers` (`bool`, optional, defaults to `False`). Dedicated SparseCore placement for FSDP weight all-gathers only. OR-ed with the all-gathers flag.
- `moe_pin_sparse_core_fsdp_all_gathers_fwd_only` (`bool`, optional, defaults to `False`). Forward-only placement when FSDP gathering is pinned. Weight-gradient resharding stays outside `compute_on`.
- `moe_fsdp_all_gather_sparse_core_id` (`int`, optional, defaults to `0`). FSDP pinning SparseCore ID.
- `moe_ep_all_gather_sparse_core_id` (`int`, optional, defaults to `1`). EP pinning SparseCore ID.
- `use_random_routing` (`bool`, optional, defaults to `False`). Random routing for debugging.
- `interleave_moe_layer_step` (`int`, optional, defaults to `1`). MoE layer frequency. Two means every second layer.
- `moe_fsdp_use_two_stage_all_gather` (`bool`, optional, defaults to `False`). Separate gathers for FSDP and FSDP-transpose sharded weights.
- `shard_exp_on_fsdp` (`bool`, optional, defaults to `False`). Expert weight dimension sharding over FSDP. Mutually exclusive with embedding and 2D FSDP sharding.
- `shard_embed_moe_on_fsdp` (`bool`, optional, defaults to `False`). Expert embedding width sharding for manual quantized gathering. Requires fixed weight calibration.
- `use_2d_fsdp_sharding` (`bool`, optional, defaults to `False`). FSDP and FSDP-transpose sharding axes.
- `norm_topk_prob` (`bool`, optional, defaults to `False`). Selected routing weight normalization to sum to one.
- `return_routed_experts` (`bool`, optional, defaults to `False`). Selected expert indices in decoder intermediates for capture and replay.
- `float32_weight_sum` (`bool`, optional, defaults to `True`). Float32 weighted expert output accumulation followed by a cast to model dtype.
- `float32_gate_logits` (`bool`, optional, defaults to `False`). Float32 router projections and scores with unchanged stored weight dtype. Cannot combine with router-projection quantization.
- `prefuse_moe_weights` (`bool`, optional, defaults to `False`). Initialization-time fusion of w0 and w1 for one FFN1 grouped GEMM in sparse paths and direct fused weights in `vllm_rpa`.
- `fuse_expert_scales` (`bool`, optional, defaults to `False`). Expert scaling factors folded into weights.
- `act_quantization_calibration_method` (`str`, optional, defaults to `"absmax"`). Expert activation calibration.

</details>

## MoEKernelsConfig

Import from `lilax.configuration`. Pydantic model with keyword fields.

Configures expert kernels. TE grouped GEMM requires one local device per process and is incompatible with
Megablox, Tokamax GMM, and quantization.

<details>
<summary>Parameters</summary>

- `megablox` (`bool`, optional, defaults to `True`). Megablox kernels. Automatically disabled on CPU.
- `sparse_matmul` (`bool`, optional, defaults to `True`). Sparse expert matmul.
- `wi_tile_fwd_batch_seq` (`int`, optional, defaults to `512`). Input projection forward batch/sequence tile.
- `wi_tile_fwd_embed_dim` (`int`, optional, defaults to `1024`). Input projection forward embedding tile.
- `wi_tile_fwd_mlp_dim` (`int`, optional, defaults to `1024`). Input projection forward MLP tile.
- `eval_wi_tile_fwd_batch_seq` (`int`, optional, defaults to `512`). Evaluation input projection batch/sequence tile.
- `eval_wi_tile_fwd_embed_dim` (`int`, optional, defaults to `1024`). Evaluation input projection embedding tile.
- `eval_wi_tile_fwd_mlp_dim` (`int`, optional, defaults to `1024`). Evaluation input projection MLP tile.
- `wi_tile_dlhs_batch_seq` (`int`, optional, defaults to `512`). Input projection backward input-gradient batch/sequence tile.
- `wi_tile_dlhs_embed_dim` (`int`, optional, defaults to `1024`). Input projection backward input-gradient embedding tile.
- `wi_tile_dlhs_mlp_dim` (`int`, optional, defaults to `1024`). Input projection backward input-gradient MLP tile.
- `wi_tile_drhs_batch_seq` (`int`, optional, defaults to `512`). Input projection weight-gradient batch/sequence tile.
- `wi_tile_drhs_embed_dim` (`int`, optional, defaults to `1024`). Input projection weight-gradient embedding tile.
- `wi_tile_drhs_mlp_dim` (`int`, optional, defaults to `1024`). Input projection weight-gradient MLP tile.
- `wo_tile_fwd_batch_seq` (`int`, optional, defaults to `512`). Output projection forward batch/sequence tile.
- `wo_tile_fwd_embed_dim` (`int`, optional, defaults to `1024`). Output projection forward embedding tile.
- `wo_tile_fwd_mlp_dim` (`int`, optional, defaults to `1024`). Output projection forward MLP tile.
- `eval_wo_tile_fwd_batch_seq` (`int`, optional, defaults to `512`). Evaluation output projection batch/sequence tile.
- `eval_wo_tile_fwd_embed_dim` (`int`, optional, defaults to `1024`). Evaluation output projection embedding tile.
- `eval_wo_tile_fwd_mlp_dim` (`int`, optional, defaults to `1024`). Evaluation output projection MLP tile.
- `wo_tile_dlhs_batch_seq` (`int`, optional, defaults to `512`). Output projection input-gradient batch/sequence tile.
- `wo_tile_dlhs_embed_dim` (`int`, optional, defaults to `1024`). Output projection input-gradient embedding tile.
- `wo_tile_dlhs_mlp_dim` (`int`, optional, defaults to `1024`). Output projection input-gradient MLP tile.
- `wo_tile_drhs_batch_seq` (`int`, optional, defaults to `512`). Output projection weight-gradient batch/sequence tile.
- `wo_tile_drhs_embed_dim` (`int`, optional, defaults to `1024`). Output projection weight-gradient embedding tile.
- `wo_tile_drhs_mlp_dim` (`int`, optional, defaults to `1024`). Output projection weight-gradient MLP tile.
- `merge_gating_gmm` (`bool`, optional, defaults to `False`). Merged gating GMM kernels.
- `num_moe_emb_chunks` (`int`, optional, defaults to `0`). Embedding chunks for overlapping token gathering and GMM.
- `use_tokamax_gmm` (`bool`, optional, defaults to `False`). Tokamax GMM implementation.
- `use_gmm_v2` (`bool`, optional, defaults to `False`). Tokamax GMM v2.
- `use_te_grouped_gemm` (`bool`, optional, defaults to `False`). TE GPU grouped matmul under the restrictions above.
- `use_gmm_v2_heuristic_tiling` (`bool`, optional, defaults to `False`). Heuristic tiling with GMM v2 enabled.

</details>

## DeepSeekMoEConfig

Import from `lilax.configuration`. Pydantic model with keyword fields.

Configures DeepSeek routing and execution. Positive routed-bias update rates are unsupported with TE MoE.
Deferred reductions require sparse ring routing and reject layer dropless fallback.

<details>
<summary>Parameters</summary>

- `first_num_dense_layers` (`NonNegativeInt`, optional, defaults to `0`). Initial dense layer count.
- `shared_experts` (`NonNegativeInt`, optional, defaults to `0`). Shared expert count.
- `routed_scaling_factor` (`float`, optional, defaults to `1.0`). Routing score multiplier.
- `routed_score_func` (`str`, optional, defaults to `""`). Routing score function, such as softmax or sigmoid.
- `routed_bias` (`bool`, optional, defaults to `False`). Routing bias.
- `routed_bias_update_rate` (`float`, optional, defaults to `0.0`). Training router-bias update rate.
- `defer_small_all_reduces` (`bool`, optional, defaults to `False`). Combined routed-bias count reductions across layers.
- `log_moe_bias_norms` (`bool`, optional, defaults to `False`). Router-bias norm logging.
- `log_step_diagnostics` (`bool`, optional, defaults to `False`). Logging of gradient and parameter norms, maximum gradient, routed-bias checksum and update count, and overflow status.
- `mlp_bias` (`bool`, optional, defaults to `False`). Expert feed-forward projection biases.
- `n_routing_groups` (`int`, optional, defaults to `-1`). Routing group count. Disabled by default.
- `first_num_hash_layers` (`int`, optional, defaults to `0`). DeepSeek V4 hash-routing layer count. Zero disables it.
- `topk_routing_group` (`int`, optional, defaults to `-1`). Selected routing group count.
- `use_batch_split_schedule` (`bool`, optional, defaults to `False`). Microbatch splitting for overlapping communication and computation.
- `batch_split_factor` (`int`, optional, defaults to `1`). Batch split count when the schedule is enabled.
- `use_lineage` (`bool`, optional, defaults to `False`). Lineage DeepSeek-V3 execution.
- `lineage_quantization` (`Literal["none", "fp8_full", "fp8_full_splash_qk", "fp8_full_splash_qk_e3", "fp8_full_combine_per_hop", "fp8_full_splash_qk_e3_combine_per_hop"]`, optional, defaults to `"none"`). Lineage quantization mode. `None` normalizes to `"none"`.
- `lineage_expert_permutation` (`Literal["lpt"] | None`, optional, defaults to `None`). `None` retains expert order. `lpt` uses preceding-step token counts. The first step retains identity order.
- `lineage_diag_band_skip` (`bool`, optional, defaults to `True`). Row-band diagonal skip in Lineage Splash attention.
- `lineage_dq_reduction_steps` (`int`, optional, defaults to `1`). Query-gradient reduction. One uses VMEM reduction. Zero disables it.

</details>

