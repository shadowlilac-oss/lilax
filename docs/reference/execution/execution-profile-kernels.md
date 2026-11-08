# Attention kernel profile fields

These optional YAML sections belong to an [execution profile](execution-profile.md). They tune the selected kernel. They do not select a model or make an unsupported kernel available.

## `kernels`

Forcing the query layout requires `use_jax_splash=True` in model kernel selection.

<details>
<summary>Parameters</summary>

- `ragged_block_size` (`int`, optional, defaults to `256`). Ragged-attention block size.
- `force_q_layout` (`bool`, optional, defaults to `false`). Force query layout.

</details>

## `splash` block sizes and layouts

Q blocks tile queries. KV blocks tile keys and values. Compute blocks select the KV compute tile. The `dkv` and `dq` suffixes apply to the two backward passes. The `eval_` prefix applies during evaluation.

Layouts use the selected Splash implementation's layout names. Kernel choices, tile divisibility, local sequence lengths, and model head dimensions still constrain which combinations can execute.

FP8 Q and K options currently require MLA attention. Both scaled-Q options require Tokamax Splash, flash attention, and base-two exponentiation. They reject context strategies `ring` and `usp`. Prescaling also rejects absorbed MLA MQA and cannot be enabled together with scaled-Q reuse. Segment-mask omission requires stream-chunked evaluation only when evaluation is enabled.

KV-gradient megacore requires more than one KV head. The diagonal grid must be a power of two and at least two. Segment-mask omission requires one segment per row, `c4_mlperf` stream chunking, flash attention, and attention type `global`, `mla`, `local_sliding`, or `chunk`. It is unsupported with lineage. Ring, Ulysses, USP, and compressed attention accept `dq_reduction_steps` of `0` or `3`. Ring with sparse indexer masks requires `0`.

<details>
<summary>Parameters</summary>

- `sa_block_q` (`int`, optional, defaults to `512`). Forward Q block.
- `sa_block_kv` (`int`, optional, defaults to `512`). Forward KV block.
- `sa_block_kv_compute` (`int`, optional, defaults to `512`). Forward KV compute tile.
- `eval_sa_block_q` (`int`, optional, defaults to `512`). Evaluation Q block.
- `eval_sa_block_kv` (`int`, optional, defaults to `512`). Evaluation KV block.
- `eval_sa_block_kv_compute` (`int`, optional, defaults to `512`). Evaluation KV compute tile.
- `sa_block_q_dkv` (`int`, optional, defaults to `512`). Q block for KV gradients.
- `sa_block_kv_dkv` (`int`, optional, defaults to `512`). KV block for KV gradients.
- `sa_block_kv_dkv_compute` (`int`, optional, defaults to `512`). KV compute tile for KV gradients.
- `sa_block_q_dq` (`int`, optional, defaults to `512`). Q block for query gradients.
- `sa_block_kv_dq` (`int`, optional, defaults to `512`). KV block for query gradients.
- `sa_use_fused_bwd_kernel` (`bool`, optional, defaults to `false`). Fuse backward operations.
- `sa_bwd_dkv_megacore` (`bool`, optional, defaults to `false`). Parallelize KV-head groups in the static KV-gradient grid.
- `sa_q_layout` (`str`, optional, defaults to `"HEAD_DIM_MINOR"`). Q layout.
- `sa_k_layout` (`str`, optional, defaults to `"HEAD_DIM_MINOR"`). K layout.
- `sa_v_layout` (`str`, optional, defaults to `"HEAD_DIM_MINOR"`). V layout.
- `eval_sa_q_layout` (`str`, optional, defaults to `"HEAD_DIM_MINOR"`). Evaluation Q layout.
- `eval_sa_k_layout` (`str`, optional, defaults to `"HEAD_DIM_MINOR"`). Evaluation K layout.
- `eval_sa_v_layout` (`str`, optional, defaults to `"HEAD_DIM_MINOR"`). Evaluation V layout.
- `use_splash_scheduler` (`bool`, optional, defaults to `false`). Experimental scheduler.
- `ring_scan_unroll` (`NonNegativeInt`, optional, defaults to `1`). Ring scan unroll factor. Zero or a value at least the ring size fully unrolls. Larger factors increase program size and compilation work.
- `sa_fuse_reciprocal` (`bool`, optional, defaults to `true`). Fuse the reciprocal operation.
- `sa_use_base2_exp` (`bool`, optional, defaults to `true`). Base-two exponentiation.
- `sa_qk_diag_skip` (`bool`, optional, defaults to `false`). Masked diagonal QK work omission.
- `sa_sv_diag_skip` (`bool`, optional, defaults to `false`). Masked diagonal score-value work omission.
- `sa_qk_diag_grid` (`int`, optional, defaults to `2`). Diagonal grid subdivision.
- `experimental_sa_quant_q_fp8` (`bool`, optional, defaults to `false`). Cast Q to `float8_e4m3fn` without scaling.
- `experimental_sa_quant_k_fp8` (`bool`, optional, defaults to `false`). Cast K to `float8_e4m3fn` without scaling.
- `sa_dkv_reuse_scaled_q` (`bool`, optional, defaults to `false`). Experimental scaled-query reuse in fused KV gradients for lower memory use with `sa_use_base2_exp=True`. Supports BF16 and FP8 queries across attention types.
- `sa_q_prescaled_log2e` (`bool`, optional, defaults to `false`). Experimental query prescaling for improved FP8 MLA rounding with `sa_use_base2_exp=True` and `experimental_sa_quant_q_fp8=True`. Supersedes scaled-query reuse on non-absorbed MLA.
- `sa_ignore_segment_ids` (`bool`, optional, defaults to `false`). Omit segment masks under the constraints above.
- `use_max_logit_estimate` (`int`, optional, defaults to `-1`). `-1` disables estimation. Positive values supply the maximum-logit estimate.
- `cost_estimate_flops_fwd` (`int`, optional, defaults to `-1`). `-1` uses Splash's estimate. Nonnegative values supply forward FLOPs for communication overlap scheduling.
- `cost_estimate_flops_bwd` (`int`, optional, defaults to `-1`). Same override for backward scheduling.
- `dq_reduction_steps` (`int`, optional, defaults to `0`). Query-gradient reduction schedule.

</details>

## `splash` local-attention overrides

Resolution inherits the corresponding non-local setting when the override is null. For example, `local_sa_block_q` inherits `sa_block_q`, and `eval_local_sa_block_q` inherits `eval_sa_block_q`.

<details>
<summary>Parameters</summary>

- `local_sa_block_q` (`int | None`, optional, defaults to `null`). Local forward Q block.
- `local_sa_block_kv` (`int | None`, optional, defaults to `null`). Local forward KV block.
- `local_sa_block_kv_compute` (`int | None`, optional, defaults to `null`). Local forward KV compute tile.
- `eval_local_sa_block_q` (`int | None`, optional, defaults to `null`). Local evaluation Q block.
- `eval_local_sa_block_kv` (`int | None`, optional, defaults to `null`). Local evaluation KV block.
- `eval_local_sa_block_kv_compute` (`int | None`, optional, defaults to `null`). Local evaluation KV compute tile.
- `local_sa_block_q_dkv` (`int | None`, optional, defaults to `null`). Local Q block for KV gradients.
- `local_sa_block_kv_dkv` (`int | None`, optional, defaults to `null`). Local KV block for KV gradients.
- `local_sa_block_kv_dkv_compute` (`int | None`, optional, defaults to `null`). Local KV compute tile for KV gradients.
- `local_sa_block_q_dq` (`int | None`, optional, defaults to `null`). Local Q block for query gradients.
- `local_sa_block_kv_dq` (`int | None`, optional, defaults to `null`). Local KV block for query gradients.
- `local_sa_use_fused_bwd_kernel` (`bool | None`, optional, defaults to `null`). Local fused backward kernel.
- `local_sa_q_layout` (`str | None`, optional, defaults to `null`). Local Q layout.
- `local_sa_k_layout` (`str | None`, optional, defaults to `null`). Local K layout.
- `local_sa_v_layout` (`str | None`, optional, defaults to `null`). Local V layout.
- `eval_local_sa_q_layout` (`str | None`, optional, defaults to `null`). Local evaluation Q layout.
- `eval_local_sa_k_layout` (`str | None`, optional, defaults to `null`). Local evaluation K layout.
- `eval_local_sa_v_layout` (`str | None`, optional, defaults to `null`). Local evaluation V layout.
- `local_use_splash_scheduler` (`bool | None`, optional, defaults to `null`). Local scheduler.
- `local_sa_fuse_reciprocal` (`bool | None`, optional, defaults to `null`). Local reciprocal fusion.
- `local_sa_use_base2_exp` (`bool | None`, optional, defaults to `null`). Local base-two exponentiation.

</details>
