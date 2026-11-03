"""Configuration for mixture of experts models."""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt, PositiveInt, field_validator, model_validator


class TEGroupedGemmQuantizationType(str, Enum):
  EMPTY = ""
  TE_NO_QUANT = "te_no_quant"
  TE_MXFP8 = "te_mxfp8"


class RematLocation(str, Enum):
  REMAT = "remat"
  DEVICE = "device"
  OFFLOAD = "offload"


class MoEGeneral(BaseModel):
  """General configuration for Mixture of Experts (MoE) layers."""

  num_experts: PositiveInt = Field(1, description="The total number of experts in each MoE layer.")
  num_experts_per_tok: PositiveInt = Field(1, description="The number of experts to route each token to.")
  capacity_factor: float = Field(-1.0, description="Expert capacity factor. If < 0, no token dropping.")
  ragged_buffer_factor: float = Field(-1.0, description="Ragged buffer factor. If < 0, ragged buffer is worst case size.")
  eval_ragged_buffer_factor: float = Field(
      -1.0, description="Evaluation ragged buffer factor, applied under eval logical axis rules. If < 0, worst case size."
  )
  moe_dropless_fallback: Literal["step", "layer"] | None = Field(None, description=(
      "What to do when the ragged buffer (ragged_buffer_factor > 0) would drop tokens. None: drop them."
      " 'step': roll the step's update back in-graph and replay the whole train step with a dropless buffer."
      " 'layer': per MoE layer, calculate the overflow from the router's top-k and take a chunked dropless"
      " branch (lax.cond) for that layer only; no step replay. 'step' requires use_ring_of_experts=True;"
      " 'layer' also supports the ragged all-to-all path."
  ))
  retry_dropless_first_steps: int = Field(0, description=(
      "With moe_dropless_fallback='step', run the first N steps after the loop's start step (i.e. steps with "
      "step - start_step < N; equal to step < N when training starts at step 0) directly with the dropless "
      "program, skipping the attempt with the normal program. 0 = off."
  ))
  first_phase_ragged_buffer_factor: float = Field(0.0, description=(
      "With retry_dropless_first_steps=N > 0, run the first N steps with a precompiled first-phase program whose "
      "RoutedMoE modules use this ragged buffer factor (instead of the dropless program); a first-phase step that "
      "still drops tokens is replayed with the dropless program. Must be >= ragged_buffer_factor. 0 = off."
  ))
  log_required_ragged_buffer_factor: bool = Field(False, description=(
      "Probe: log per train step and per MoE layer the minimum ragged_buffer_factor that would have avoided drops "
      "(max over shards of the tokens routed to a shard / the per-shard buffer at factor 1). Adds one scalar "
      "max all-reduce per MoE layer and token chunk, and a device-to-host fetch per step."
  ))
  warmup_programs_in_init: bool = Field(False, description=(
      "After the setup precompiles and before init_stop/run_start are logged, execute each precompiled train "
      "program (normal, first-phase, dropless) and eval program (eval, eval dropless) once through its jit on a "
      "synthetic batch (PRNG token ids, never the dataset), discard the outputs so parameters, optimizer state "
      "and routed biases are unchanged, block on completion and run a cross-host barrier. Moves the first "
      "execution (and the hosts' arrival skew) out of the scored window. Skipped when the programs are not "
      "precompiled (compiled_trainstep_file or AutoPGLE)."
  ))
  warmup_skip_train_dropless: bool = Field(False, description=(
      "Requires warmup_programs_in_init=True. Skips the train_dropless program in that warmup (other programs are "
      "still warmed). Its warmup runs on a copy of the donated train state, so HBM holds two state copies plus "
      "the dropless program reservation, which can OOM. Use only when that warmup OOMs: skipping moves the "
      "dropless program's first load and execution cost into the first dropless replay, if any."
  ))
  num_moe_token_chunks: PositiveInt = Field(1, description=(
      "Number of token chunks for the ring-of-experts MoE pipeline. 1"
      " disables chunking (identical to baseline). >1 splits the per-shard"
      " tokens along the sequence dimension so each chunk's EP all-gather /"
      " reduce-scatter overlaps the previous chunk's GMM compute. Requires"
      " use_ring_of_experts=True."
  ))
  moe_accumulate_chunk_wgrad: bool = Field(False, description=(
      "With num_moe_token_chunks>1, chain the expert weights through the token chunks: chunk 0 gathers and"
      " quantizes them once and later chunks reuse the result. In the backward pass each chunk's weight-gradient"
      " tgmm accumulates in place onto the later chunks' gradient (tgmm_v2 partial_sum) instead of"
      " materializing and summing one gradient per chunk. Requires use_tokamax_gmm=True, use_gmm_v2=True,"
      " num_moe_emb_chunks=0 and prefuse_moe_weights=False."
  ))
  moe_accumulate_wi_dlhs: bool = Field(False, description=(
      "Route the routed-MoE wi_1 GMM input through the wi_0 GMM so that in the backward pass the wi_1 DLHS"
      " is accumulated in place into the wi_0 DLHS (gmm_v2 partial_sum) instead of summed by a separate add."
      " Also folds per-tensor DLHS scales into the gmm_v2 kernel for all MoE GMMs. Requires"
      " use_tokamax_gmm=True, use_gmm_v2=True and prefuse_moe_weights=False."
  ))
  moe_flat_fsdp_weights: bool = Field(False, description=(
      "Store every routed-expert weight as a 2D [num_experts * rows, cols] matrix whose first dim is"
      " sharded over (expert, fsdp) instead of the 3D [E, rows, cols] layout sharded (expert | fsdp | -)."
      " The per-device FSDP shard is then a contiguous, tile-aligned row block at every FSDP size, the fp8"
      " weight all-gather / weight-grad reduce-scatter run on the major dim, and the gathered weight is a free"
      " bitcast of the [E, rows, cols] operand the gmm kernels expect (no relayout copies, no 7/8 sublane"
      " padding). Requires shard_embed_moe_on_fsdp with fixed weight calibration (explicit QAG) and the"
      " tokamax gmm_v2 custom-VJP path. 3D checkpoints loaded via load_parameters_path are converted to the flat"
      " layout on load; full-state resumes must come from a flat checkpoint."
  ))
  moe_gmm_v2_dlhs_transpose_rhs: bool = Field(False, description="Use native transpose_rhs in GMM v2 backward DLHS instead of an explicit RHS transpose.")
  moe_bwd_prequant_before_unsort: bool = Field(False, description=(
      "Fuse the TC unsort-reduce into the wo GMM custom VJP so backward quantization runs on the smaller"
      " [N_tokens, D // 128, 128] cotangent before the TC unsort-gather into the expert buffer."
      " With quantization, needs fixed weight, act and bwd calibration (per-tensor scales applied in token order)."
  ))
  moe_combine_bwd_direct_qarray: bool = Field(False, description=(
      "With moe_bwd_prequant_before_unsort=True, fuse the EP combine psum_scatter into the wo GMM custom VJP"
      " so backward quantizes the local cotangent once before the EP all-gather and TC unsort-gather."
      " Incompatible with moe_quantize_combine_bwd_method. Each EP shard scales its cotangent before the all-gather, so"
      " the fixed weight, act and bwd calibration required by moe_bwd_prequant_before_unsort must match across"
      " shards."
  ))
  moe_3d_dlhs_partial_sum: bool = Field(False, description=(
      "Skip redundant zero-initialization in 3D DLHS gmm_v2 when accumulating onto an existing 3D partial_sum"
      " (e.g. wi_0 DLHS accumulating onto wi_1 DLHS)."
  ))
  moe_chunk_barrier: bool = Field(False, description=(
      "Diagnostic (profiling, not production). When True, chain the chunked"
      " ring-of-experts MoE loop so each chunk's input is fenced with"
      " jax.lax.optimization_barrier on the previous chunk's output,"
      " forcing XLA to run the chunks sequentially (no interleave/fusion)."
      " Math is unchanged (barrier is identity), so loss stays bit-exact."
      " Used to test whether the token-AG/RS chunks overlap at all today."
      " Requires num_moe_token_chunks>1 and use_ring_of_experts=True to have any"
      " effect."
  ))
  moe_log_max_load_ratio: bool = Field(False, description=(
      "Debug only. Log learning/moe_max_load_ratio{,_mean}: the max (mean) over MoE layers of"
      " the largest (tokens routed to an expert shard) / (perfectly balanced load) over token"
      " chunks and expert shards. A step overflows the ragged buffer iff the max exceeds"
      " ragged_buffer_factor (up to buffer rounding). Requires sparse_matmul=True,"
      " te_moe_block=False, use_ring_of_experts=True and use_ragged_sort=True."
  ))
  moe_expert_input_dim: int = Field(-1, description="Dimension of tokens entering the MoE layer. If < 0, defaults to emb_dim.")
  base_moe_mlp_dim: int = Field(-1, description="Intermediate dimension at MoE layer.")
  padded_base_moe_mlp_dim: int | None = Field(None, description="Padded intermediate dimension at MoE layer for efficient GMM_v2 kernel execution.")
  load_balance_loss_weight: NonNegativeFloat = Field(0.0, description="Weight for the load balancing auxiliary loss.")
  moe_use_megatron_seq_aux_loss: bool = Field(False, description=(
      "When True, sigmoid and Sqrt(Softplus) routers (routed_score_func='sigmoid' or 'sqrtsoftplus', with"
      " te_moe_block=False) use Megatron-LM's sequence-wise auxiliary load-balancing loss (fp32 per-token"
      " normalized affinity scores,"
      " token fractions from an unbiased, ungrouped top-k over the full sequence, summed over MoE layers)."
      " When False (default), these routers use the Switch Transformer-style loss averaged over MoE layers."
  ))
  use_custom_sort_vjp: bool = Field(True, description="Whether to use a custom VJP sort for efficient backward pass processing in sparse matmul.")
  router_topk_matmul_vjp: bool = Field(False, description=(
      "DeepSeek routing: take the top-k routing weights with a custom VJP whose backward builds the dense"
      " [..., num_experts] gradient as a one-hot compare-select-sum over the k slots, instead of the scatter-add"
      " that jnp.take_along_axis transposes to. Forward and gradient values are unchanged."
  ))
  moe_topk_before_ep_all_gather: bool = Field(False, description=(
      "Ring of experts: run top-k on the local tokens and all-gather the top-k weights and expert ids"
      " instead of the full router logits. The token sort still runs after the all-gather."
      " Ignored with forced, hash or random routing."
  ))
  use_ring_of_experts: bool = Field(False, description="Whether to use Ring of Experts for sparse matmul expert parallelism.")
  te_moe_block: bool = Field(False, description="Whether to use TransformerEngine's fused EP MoEBlock for routing, dispatch, grouped GEMM, and combine.")
  te_moe_disable_cudnn_gmm_fusion: bool = Field(False, description=(
      "Force te_moe.moe(..., use_cudnn_fusion=False). When false, MaxText does not pass use_cudnn_fusion "
      "and uses TransformerEngine's default."
  ))
  te_moe_quantize_before_fsdp_all_gather: bool = Field(False, description="Quantize TE MoE weight shards before the FSDP all-gather; false quantizes after gathering.")
  te_ep_overflow_check_every_n_steps: PositiveInt = Field(20, description=(
      "Number of training steps buffered between host-side TE EP receive-capacity overflow checks when "
      "ragged_buffer_factor limits the TE receive capacity. "
      "Overflowing steps still skip their optimizer update immediately on device."
  ))
  te_gmm_quantization: TEGroupedGemmQuantizationType | None = Field(
      TEGroupedGemmQuantizationType.EMPTY, description="Quantization mode for TransformerEngine grouped GEMMs."
  )
  moe_quantize_token_all_gather: bool = Field(False, description=(
      "Whether to quantize token activations before the All-Gather across EP"
      " shards in Ring of Experts, reducing inter-chip token traffic"
      " proportionally to the quantized width (2x for FP8 vs BF16)."
      " Pipeline: quantize token -> EP all-gather -> sort -> GMM, so tokens"
      " stay quantized end to end. Requires use_ring_of_experts=True,"
      " use_gmm_v2=True, and quantization with fixed act calibration."
  ))
  moe_dispatch_no_expert_sharding: bool = Field(False, description=(
      "If true, shard the MoE dispatch/MLP batch dim without 'expert' so the expert GEMM "
      "stays expert-parallel (AllToAll); false keeps 'expert' on it (activation_batch_moe)."
  ))
  use_ragged_sort: bool = Field(False, description="Whether to use ragged kernel for sorting, improve performance when EP is enabled.")
  moe_quantize_combine_bwd_method: str | None = Field("", description=(
      "Quantization method for Token Combine backward All-Gather. "
      "'' = unquantized BF16; 'rowwise' = dynamic token-wise FP8 E5M2; "
      "'fixed,<bound>' = static per-tensor FP8 E5M2 (e.g. 'fixed,1.0', 'fixed,57344')."
  ))
  ragged_sort_use_single_sparsecore: bool = Field(False, description="Whether to run ragged sort kernels on 1 SparseCore instead of all SparseCores.")
  moe_tc_ragged_sort: bool = Field(False, description=(
      "Ring-of-experts ragged sort/unsort on TensorCore (DMA Pallas kernels ported from lineage)"
      " instead of the SparseCore ragged_gather / ragged_gather_reduce kernels. Requires"
      " use_ring_of_experts, use_ragged_sort and ragged_buffer_factor > 0."
  ))
  moe_tc_ragged_gather_block_size: int = Field(1024, description="Rows per grid step of the TC ragged gather.")
  moe_tc_ragged_reduce_block_size: int = Field(896, description="VMEM rows per block of the TC ragged gather-reduce.")
  moe_tc_ragged_mask_padding: bool = Field(True, description="Zero TC ragged-gather buffer rows past the valid count (the kernel leaves them uninitialized).")
  moe_tc_ragged_weights_on_activation: bool = Field(False, description=(
      "With moe_tc_ragged_sort: multiply routing weights into the expert activation (buffer, mlp)"
      " before the wo matmul instead of into the (buffer, emb) expert output before the unsort."
  ))
  moe_tc_ragged_flatten_block_size: int = Field(0, description=(
      "If > 0, relayout (N, D/128, 128) <-> (N, D) around the TC ragged kernels with a Pallas kernel over"
      " only the valid rows (lineage ragged_flatten), this many rows per grid step. 0 uses XLA reshapes."
  ))
  moe_tc_ragged_3d_gmm: bool = Field(False, description=(
      "With moe_tc_ragged_sort (tokamax gmm_v2): keep the sorted token buffer in the TC kernels' 3D layout"
      " (buffer, emb // 128, 128). The wi gmm_v2 / tgmm_v2 kernels consume it and the wo gmm_v2 produces it"
      " directly (relayout in VMEM), removing the (N, D/128, 128) <-> (N, D) relayouts around the sorted"
      " buffer. Embed-dim GMM tiles must be multiples of 128 that divide the embed dim; tiles that are not"
      " multiples of 1024 (e.g. 3584 / 1792) DMA whole (emb // 128, 128) slabs and relayout the window in VMEM."
  ))
  moe_tc_ragged_3d_dispatch: bool = Field(False, description=(
      "With moe_tc_ragged_sort and moe_tc_ragged_3d_gmm under ring of experts: reshape the local tokens to"
      " (batch, seq, emb // 128, 128) before the expert-parallel all-gather, and keep the TC unsort output and"
      " the combine reduce-scatter in that layout (only the local result is reshaped back). This removes the"
      " full-size (N, D/128, 128) <-> (N, D) relayouts of the gathered tokens and of the combine output, in"
      " both the forward and the backward pass. Raises if it cannot apply (e.g. emb not a multiple of 128)."
  ))
  moe_tc_window_routing: bool = Field(False, description=(
      "With moe_tc_ragged_sort: keep only this shard's [start, start + cap) window of the sorted slots (so the "
      "combine metadata sorts cap instead of N * topk slots), skip the unused inverse permutation, use uint32 "
      "bit-packed sorts, and gather buffer row weights with unique_indices=True (no backward scatter-sort)."
  ))
  moe_use_direct_token_gather: bool = Field(False, description="Whether to gather tokens directly in expert order instead of materializing Top-K copies.")
  use_gather_mosaic_kernel: bool = Field(False, description="Whether to use a custom mosaic kernel for token gather ops.")
  ragged_gather_fallback: bool = Field(False, description=(
      "When true, unconditionally use the JAX reference implementation instead of the ragged gather "
      "SparseCore kernel. When false (default), use the SparseCore kernel."
  ))
  ragged_gather_reduce_fallback: bool = Field(False, description=(
      "When true, unconditionally use the JAX reference implementation instead of the ragged gather "
      "reduce SparseCore kernel. When false (default), use the SparseCore kernel."
  ))
  ragged_gather_cost_estimate_flops: int = Field(-1, description=(
      "Flop cost estimate override for the ragged gather kernel. "
      "-1 means auto-compute, any > 0 value overrides the flop cost estimate."
  ))
  ragged_gather_reduce_cost_estimate_flops: int = Field(-1, description=(
      "Flop cost estimate override for the ragged gather reduce kernel. "
      "-1 means auto-compute, any > 0 value overrides the flop cost estimate."
  ))
  ragged_gather_cost_estimate_bytes_accessed: int = Field(-1, description=(
      "Bytes-accessed cost estimate override for the ragged gather kernel. "
      "-1 means auto-compute, any > 0 value overrides the bytes_accessed cost estimate."
  ))
  ragged_gather_reduce_cost_estimate_bytes_accessed: int = Field(-1, description=(
      "Bytes-accessed cost estimate override for the ragged gather reduce kernel. "
      "-1 means auto-compute, any > 0 value overrides the bytes_accessed cost estimate."
  ))
  moe_pin_sparse_core_all_gathers: bool = Field(False, description="Pin FSDP and EP all-gathers in MoE to dedicated SparseCores using compute_on.")
  moe_pin_sparse_core_ep_all_gathers: bool = Field(False, description=(
      "Pin only the ring-of-experts EP all-gathers (dispatch tokens and routing inputs) to a SparseCore"
      " (OR-ed with moe_pin_sparse_core_all_gathers)."
  ))
  moe_pin_sparse_core_fsdp_all_gathers: bool = Field(False, description=(
      "Pin only the MoE FSDP weight all-gathers to a SparseCore (OR-ed with"
      " moe_pin_sparse_core_all_gathers)."
  ))
  moe_pin_sparse_core_fsdp_all_gathers_fwd_only: bool = Field(False, description=(
      "When the MoE FSDP weight all-gathers are pinned, pin only the forward all-gather; its transpose"
      " (the weight-gradient reshard) runs outside compute_on, as with the pin off."
  ))
  moe_fsdp_all_gather_sparse_core_id: int = Field(0, description="SparseCore ID to pin MoE FSDP all-gathers to when moe_pin_sparse_core_all_gathers is True.")
  moe_ep_all_gather_sparse_core_id: int = Field(1, description="SparseCore ID to pin MoE EP all-gathers to when moe_pin_sparse_core_all_gathers is True.")
  use_random_routing: bool = Field(False, description="Whether to use random routing for debugging.")
  interleave_moe_layer_step: int = Field(1, description="Frequency of MoE layers, e.g., 2 means every 2nd layer is MoE.")
  moe_fsdp_use_two_stage_all_gather: bool = Field(False, description="Use two separate All-Gather calls for MoE weights sharded on both FSDP and FSDP-transpose.")
  shard_exp_on_fsdp: bool = Field(False, description=(
      "Shard the expert dimension of the MLP weights on the FSDP axis, "
      "and recommended only when num_experts is a multiple of fsdp_parallelism"
  ))
  shard_embed_moe_on_fsdp: bool = Field(False, description="Keep embed_moe sharded so we can manually QAG it over FSDP.")
  use_2d_fsdp_sharding: bool = Field(False, description="Use `fsdp` and `fsdp_transpose` axes for 2D FSDP sharding.")
  norm_topk_prob: bool = Field(False, description="Enable top-k probability normalization for router weights (Qwen3-specific).")
  return_routed_experts: bool = Field(False, description=(
      "Record each MoE layer's routing as a `selected_experts` intermediate, returned by the "
      "decoder as `expert_indices`. Required to capture routing from inference for router replay."
  ))
  float32_weight_sum: bool = Field(True, description=(
      "Controls the accumulation precision of the MoE combine reduction after GMM (the weighted sum of expert"
      " outputs by their routing weights). When True, casts operands to float32 before the combine"
      " einsum and accumulates in float32 before casting back to the model dtype."
  ))
  float32_gate_logits: bool = Field(False, description=(
      "Whether to run the MoE gate (router) module before GMM in fp32 for numerical stability and routing precision."
      " This is a compute dtype, not a storage dtype: the gate kernel is still stored in `weight_dtype` and"
      " is cast at use. When True it sets the `GateLogit` compute dtype, which (1) casts the gate input"
      " activations and kernel before the projection matmul, (2) makes the emitted logits fp32, so downstream"
      " consumers such as top-k selection and the load balance loss see fp32 values, and (3) applies to the"
      " routed bias add and the score function. If a decoder block supplies separate gate inputs, those are"
      " cast as well; for gemma4 it additionally sets the router norm dtype and the router scale cast. It"
      " does not control quantization -- see `quantize_router_proj`, which governs whether the gate"
      " projection matmul is quantized. Setting both is rejected at config init, because quantizing the"
      " projection discards the fp32 operand precision."
  ))
  prefuse_moe_weights: bool = Field(False, description=(
      "Whether to pre-fuse MoE weights (w0 and w1) during initialization. "
      "This enables a single FFN1 grouped GEMM in sparse MoE paths and passes fused weights directly in vllm_rpa mode."
  ))
  fuse_expert_scales: bool = Field(False, description=(
      "Whether to fuse the expert scaling factors into the expert weights. "
      "This can improve inference performance."
  ))

  @model_validator(mode="after")
  def validate_moe_chunks(self) -> "MoEGeneral":
    if self.num_moe_token_chunks > 1 and not self.use_ring_of_experts:
      raise ValueError("num_moe_token_chunks > 1 requires use_ring_of_experts=True.")
    return self

  @model_validator(mode="after")
  def validate_moe_sharding_strategy(self) -> "MoEGeneral":
    if (self.moe_pin_sparse_core_all_gathers or self.moe_pin_sparse_core_fsdp_all_gathers) and self.moe_fsdp_use_two_stage_all_gather:
      raise ValueError(
          "SparseCore pinning for MoE all-gathers (`moe_pin_sparse_core_all_gathers=True`) "
          "is not supported with `moe_fsdp_use_two_stage_all_gather=True`."
      )
    active_sharding_flags = sum([self.shard_exp_on_fsdp, self.use_2d_fsdp_sharding, self.shard_embed_moe_on_fsdp])
    if active_sharding_flags > 1:
      raise ValueError(
          "Only one of shard_exp_on_fsdp, use_2d_fsdp_sharding, or "
          "shard_embed_moe_on_fsdp can be True at the same time."
      )
    return self


class MoEKernels(BaseModel):
  """Configuration for MoE-specific kernels like Megablox."""

  megablox: bool = Field(True, description="Whether to use Megablox kernels for MoE. Automatically disabled when hardware='cpu'.")
  sparse_matmul: bool = Field(True, description="Whether to use sparse matmul kernels for MoE.")
  wi_tile_fwd_batch_seq: int = Field(512, description="forward pass tiling dimension for batch/sequence in GMM for wi.")
  wi_tile_fwd_embed_dim: int = Field(1024, description="forward pass tiling dimension for embedding in GMM for wi.")
  wi_tile_fwd_mlp_dim: int = Field(1024, description="forward pass tiling dimension for MLP in GMM for wi.")
  eval_wi_tile_fwd_batch_seq: int = Field(512, description="evaluation forward pass tiling dimension for batch/sequence in GMM for wi.")
  eval_wi_tile_fwd_embed_dim: int = Field(1024, description="evaluation forward pass tiling dimension for embedding in GMM for wi.")
  eval_wi_tile_fwd_mlp_dim: int = Field(1024, description="evaluation forward pass tiling dimension for MLP in GMM for wi.")
  wi_tile_dlhs_batch_seq: int = Field(512, description="bwd pass dlhs tiling dimension for batch/sequence in GMM for wi.")
  wi_tile_dlhs_embed_dim: int = Field(1024, description="bwd pass dlhs tiling dimension for embedding in GMM for wi.")
  wi_tile_dlhs_mlp_dim: int = Field(1024, description="bwd pass dlhs tiling dimension for MLP in GMM for wi.")
  wi_tile_drhs_batch_seq: int = Field(512, description="bwd pass drhs tiling dimension for batch/sequence in GMM for wi.")
  wi_tile_drhs_embed_dim: int = Field(1024, description="bwd pass drhs tiling dimension for embedding in GMM for wi.")
  wi_tile_drhs_mlp_dim: int = Field(1024, description="bwd pass drhs tiling dimension for MLP in GMM for wi.")
  wo_tile_fwd_batch_seq: int = Field(512, description="forward pass tiling dimension for batch/sequence in GMM for wo.")
  wo_tile_fwd_embed_dim: int = Field(1024, description="forward pass tiling dimension for embedding in GMM for wo.")
  wo_tile_fwd_mlp_dim: int = Field(1024, description="forward pass tiling dimension for MLP in GMM for wo.")
  eval_wo_tile_fwd_batch_seq: int = Field(512, description="evaluation forward pass tiling dimension for batch/sequence in GMM for wo.")
  eval_wo_tile_fwd_embed_dim: int = Field(1024, description="evaluation forward pass tiling dimension for embedding in GMM for wo.")
  eval_wo_tile_fwd_mlp_dim: int = Field(1024, description="evaluation forward pass tiling dimension for MLP in GMM for wo.")
  wo_tile_dlhs_batch_seq: int = Field(512, description="bwd pass dlhs tiling dimension for batch/sequence in GMM for wo.")
  wo_tile_dlhs_embed_dim: int = Field(1024, description="bwd pass dlhs tiling dimension for embedding in GMM for wo.")
  wo_tile_dlhs_mlp_dim: int = Field(1024, description="bwd pass dlhs tiling dimension for MLP in GMM for wo.")
  wo_tile_drhs_batch_seq: int = Field(512, description="bwd pass drhs tiling dimension for batch/sequence in GMM for wo.")
  wo_tile_drhs_embed_dim: int = Field(1024, description="bwd pass drhs tiling dimension for embedding in GMM for wo.")
  wo_tile_drhs_mlp_dim: int = Field(1024, description="bwd pass drhs tiling dimension for MLP in GMM for wo.")
  merge_gating_gmm: bool = Field(False, description="whether to merge the two gating gmm kernels into one.")
  num_moe_emb_chunks: int = Field(0, description="Number of chunks for overlapping token all-gather and GMM computation along embedding dimension.")
  use_tokamax_gmm: bool = Field(False, description="Whether to use the Tokamax library for GMM kernel implementation.")
  use_gmm_v2: bool = Field(False, description="Whether to use Tokamax GMM v2 for MoE kernel.")
  use_te_grouped_gemm: bool = Field(False, description=(
      "Whether to use TransformerEngine grouped_dense for sparse MoE grouped GEMM on GPU. "
      "Incompatible with megablox and use_tokamax_gmm; quantization is not supported yet. "
      "Requires one local device per process: TE v1 copies group_sizes D2H and deadlocks "
      "in single-process multi-device shard_map."
  ))
  use_gmm_v2_heuristic_tiling: bool = Field(False, description="Whether to use the heuristic tiling from Tokamax GMM v2, when use_gmm_v2=true.")


class DeepSeekMoE(BaseModel):
  """Configuration specific to DeepSeek-style MoE layers."""

  first_num_dense_layers: NonNegativeInt = Field(0, description="Number of initial dense layers in the model.")
  shared_experts: NonNegativeInt = Field(0, description="Number of shared experts.")
  routed_scaling_factor: float = Field(1.0, description="Scaling factor for routing scores.")
  routed_score_func: str = Field("", description="Scoring function for routing (e.g., 'softmax', 'sigmoid').")
  routed_bias: bool = Field(False, description="Whether to add a bias term for routing.")
  routed_bias_update_rate: float = Field(0.0, description="Update rate applied to the router bias term.")
  defer_small_all_reduces: bool = Field(
      False,
      description="Ring-of-experts sparse_matmul only: drop the per-layer all-reduces of the routed-bias expert counts"
      " from the scanned layer. Each MoE layer emits its local partial counts, the scan stacks them, and loss_fn"
      " reduces the stacked array once after the layer loop (one all-reduce). Same sums as the per-layer path. (The"
      " token-overflow flag of moe_dropless_fallback='step' is already reduced once per step without this flag.)",
  )
  log_moe_bias_norms: bool = Field(False, description="Whether to log the norms of MoE router biases.")
  log_step_diagnostics: bool = Field(
      False,
      description="Append to every training step's log line: pre-clip and post-clip global grad norm, param norm,"
      " max |grad|, the routed-bias checksum (sum over MoE layers of sum(bias)), the number of nonzero routed-bias"
      " update entries and the MoE overflow flag.",
  )
  mlp_bias: bool = Field(
      False, description="Whether to add a learnable bias for MLP matmul, "
      "and originally implemented to support the GPT-OSS model architecture",
  )
  n_routing_groups: int = Field(-1, description="Number of groups for routing, disabled by default.")
  first_num_hash_layers: int = Field(0, description="Number of hash routing layers, used in DeepSeek V4 (0 means disabled).")
  topk_routing_group: int = Field(-1, description="Number of top groups to route inputs to.")
  use_batch_split_schedule: bool = Field(
      False, description="Whether to split batch into micro-batches to hide communications that yields performance benefits."
  )
  batch_split_factor: int = Field(
      1, description="Factor by which to split the batch into micro-batches. Only used if use_batch_split_schedule is True."
  )
  use_lineage: bool = Field(False, description="Whether to use Lineage DeepSeek-V3 execution.")
  lineage_quantization: Literal[
      "none", "fp8_full", "fp8_full_splash_qk", "fp8_full_splash_qk_e3", "fp8_full_combine_per_hop",
      "fp8_full_splash_qk_e3_combine_per_hop",
  ] = Field("none", description="Quantization of the Lineage sparse-layer routed experts and MLA. Only used if use_lineage is True.")

  @field_validator("lineage_quantization", mode="before")
  @classmethod
  def _lineage_quantization_none(cls, value: Any) -> Any:
    return "none" if value is None else value

  lineage_expert_permutation: Literal["lpt"] | None = Field(
      None,
      description=(
          "Per-layer expert permutation algorithm for the Lineage sparse and MTP layers (see `dsv3.dsv3`'s"
          " `expert_permutation`). None disables expert shuffling. 'lpt' computes each step's permutation on device"
          " from the previous step's per-expert token counts with `dsv3_expert_shuffle.lpt_expert_permutation`; the"
          " first step uses the identity."
      ),
  )
  lineage_diag_band_skip: bool = Field(True, description="Enable row-band diagonal skip in Lineage Splash attention.")
  lineage_dq_reduction_steps: int = Field(1, description=("Lineage dkv dq reduction steps: 1 for VMEM reduction, 0 for None."))
