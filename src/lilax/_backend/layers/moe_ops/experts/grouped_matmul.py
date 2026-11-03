"""Grouped matmul of expert-sorted tokens on the configured GMM backend."""

import dataclasses
import random

from aqt.jax.v2 import aqt_tensor as aqt
import jax
from jax.experimental import xla_metadata
import jax.numpy as jnp
from lilax._backend.kernels import megablox as mblx
from lilax._backend.layers import quantizations
from lilax._backend.utils import max_utils
import qwix.pallas as qpl
import tokamax

set_xla_metadata = xla_metadata.set_xla_metadata


def forward_tile(config, name):
  """Reads a forward GMM tile size, preferring `eval_*` under eval axis rules."""
  return getattr(config, f"eval_{name}" if max_utils.is_eval(config) else name)


def _extract_vma(tensor):
  # Extract underlying array from QArray to inspect sharding annotation string.
  if isinstance(tensor, qpl.QArray):
    tensor = tensor.qvalue
  # Parses the varying mesh axes from JAX's type string for a tensor inside shard_map.
  # jax.typeof(t) renders as e.g. 'f32[128,256]{V:(expert, fsdp)}'; this extracts
  # ('expert', 'fsdp'). Returns () if the tensor has no varying axes.
  type_str = str(jax.typeof(tensor))
  if "{V:" in type_str:
    start = type_str.index("{V:") + 3
    end = type_str.index("}", start)
    vma_content = type_str[start:end].strip("()")
    return tuple(sorted(a.strip() for a in vma_content.split(",")))
  return tuple()


class GroupedMatmul:
  """Grouped matmul of expert-sorted tokens against per-expert kernels on the configured GMM backend."""

  def __init__(self, config, mesh, quant, dtype, num_local_experts):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.dtype = dtype
    self.num_local_experts = num_local_experts

  def _ragged_dot(self, inputs, kernel, tiling, group_sizes, expert_assignments, padding_amount):
    """Execute jax.lax.ragged_dot, with potential quantization"""
    m, k, n = inputs.shape[0], inputs.shape[1], kernel.shape[2]
    # Clamps the tile size using the minimum
    tiling = (
        min(tiling[0], m),
        min(tiling[1], k),
        min(tiling[2], n),
    )
    rhs_inputs = kernel
    if isinstance(kernel, aqt.QTensor):
      if kernel.bias or kernel.sparsity_mask or len(kernel.scale) > 1:
        raise ValueError("Unsupported usecase for ragged_dot with quantized kernel.")
      rhs_inputs = kernel.qvalue
    if self.config.quantization and self.config.use_qwix_quantization:
      # Use full contraction for QWIX quantization to allow quantization
      # fusion (max reduce over contracting dimension).
      tiling = (tiling[0], k, tiling[2])

    is_tpu = self.mesh.devices.flat[0] == "tpu"
    # TPU needs random mosaic_fusion_group; GPU/CPU needs deterministic ID for autotuner sync
    mosaic_group_id = f"{random.randint(0, 1000000000)}" if is_tpu else "0"
    with set_xla_metadata(
        ragged_dot_tiling=",".join([str(t) for t in tiling]),
        mosaic_fusion_group=mosaic_group_id,
    ):
      output = jax.lax.ragged_dot(
          lhs=inputs,
          rhs=rhs_inputs,
          group_sizes=group_sizes,
          preferred_element_type=self.dtype,
      )
    if isinstance(kernel, aqt.QTensor):
      # Multiply outputs by the kernely scale
      scales = jnp.take(kernel.scale[0].squeeze(), indices=expert_assignments, axis=0)
      if padding_amount > 0:
        scales = jax.lax.pad(
            scales,
            jnp.array(0.0, dtype=scales.dtype),
            [(0, padding_amount, 0), (0, 0, 0)],
        )
      output *= scales
    return output

  def _tokamax_group_sizes(self, group_sizes, inputs, _kernel):
    if self.config.quantization and self.config.use_qwix_quantization:
      return group_sizes
    elif self.config.attention in ("vllm_rpa", "vllm_batched_rpa"):
      return group_sizes
    else:
      return tokamax.RaggedDotGroupSizes(
          group_sizes,
          inputs.shape[0],
      )

  def _quantization_dtypes(self):
    # AQT describes its numerics through a `quant_dg`, while the fp8 schemes just name the
    # dtype they quantize to. Under qwix there is no quantization object here at all, and
    # the gmm takes its rule from the qwix rule instead.
    if self.quant is not None and not isinstance(self.quant, quantizations.ServeFp8WeightQuantization):
      quant_dg = getattr(self.quant, "quant_dg", None)
      if quant_dg is not None:
        return quant_dg.fwd.dg_quantizer.lhs.numerics.get_dtype(), quant_dg.fwd.dg_quantizer.rhs.numerics.get_dtype()
      quantize_dtype = getattr(self.quant, "quantize_dtype", None)
      return quantize_dtype, quantize_dtype
    return None, None

  def _te_grouped_dense(self, inputs, kernel, group_sizes, group_offset, bias=None):
    """Grouped GEMM via TransformerEngine `grouped_dense` (custom VJP).

    Group sizes are passed as routed, so TE walks only the rows that actually
    hold tokens and skips the ragged-buffer padding. Output rows past that
    point are zeroed by the TE kernel (not GEMMed) so GLU / dgrad / dbias
    cannot observe uninitialized leftover.

    Note: v1 copies group_sizes D2H inside the FFI. That is safe with one
    local device (including multi-process EP, one GPU per rank) but can
    deadlock in single-process multi-device shard_map on ROCm.
    """
    # pylint: disable=import-outside-toplevel
    from transformer_engine.jax.dense import grouped_dense

    num_local_experts = kernel.shape[0]
    start = 0 if group_offset is None else group_offset
    te_group_sizes = jax.lax.dynamic_slice_in_dim(group_sizes, start, num_local_experts)
    # With a group_offset the local groups' rows follow those of groups [0, group_offset), as in
    # megablox. TE has no group offset and reads from row 0, so rotate those rows to the front.
    row_start = None
    if not (isinstance(start, int) and start == 0):
      row_start = jnp.sum(jnp.where(jnp.arange(group_sizes.shape[0]) < start, group_sizes, 0))
      inputs = jnp.roll(inputs, -row_start, axis=0)
    # TE asserts lhs.dtype == rhs.dtype and reuses that dtype in its custom VJP,
    # so a wider `preferred_element_type` also fails in the backward pass. The
    # GPT-OSS GLU can hand `wo` f32 activations while the experts stay bf16;
    # jax.lax.ragged_dot tolerates that mix, TE does not, so narrow everything
    # to one dtype and restore self.dtype on the output.
    out_dtype = jnp.dtype(self.dtype)
    compute_dtype = min(
        (jnp.dtype(kernel.dtype), jnp.dtype(inputs.dtype), out_dtype),
        key=lambda dt: dt.itemsize,
    )
    inputs = jax.lax.convert_element_type(inputs, compute_dtype)
    kernel = jax.lax.convert_element_type(kernel, compute_dtype)
    if bias is not None:
      bias = jax.lax.convert_element_type(bias, compute_dtype)
    output = grouped_dense(
        inputs,
        kernel,
        te_group_sizes.astype(jnp.int32),
        contracting_dims=((1,), (1,)),
        bias=bias,
        preferred_element_type=compute_dtype,
    )
    if row_start is not None:
      output = jnp.roll(output, row_start, axis=0)
    return jax.lax.convert_element_type(output, out_dtype)

  def __call__(
      self,
      inputs,
      kernel,
      tiling,
      group_sizes,
      expert_assignments,
      weight_gather_axes,
      group_offset,
      partial_sum=None,
      return_lhs=False,
      return_rhs=False,
      out_is_3d=False,
      tc_routing=None,
      tc_unsort_cfg=None,
      bias=None,
  ):
    is_native_kernel = isinstance(kernel, qpl.QArray)
    if self.config.megablox and not self.config.use_tokamax_gmm and not self.config.moe_quantize_token_all_gather:
      lhs_vma_axes = _extract_vma(inputs)
      rhs_vma_axes = _extract_vma(kernel.qvalue if is_native_kernel else kernel)
    else:
      lhs_vma_axes = tuple()
      rhs_vma_axes = tuple()
    if inputs.shape[0] != expert_assignments.shape[0]:
      raise ValueError("The number of input tokens must match the number of expert assignments!")

    if self.config.use_te_grouped_gemm:
      if return_lhs or return_rhs or out_is_3d or tc_unsort_cfg is not None:
        raise NotImplementedError(
            "use_te_grouped_gemm does not support return_lhs, return_rhs, out_is_3d or a fused TC unsort "
            f"(got {return_lhs=}, {return_rhs=}, {out_is_3d=}, tc_unsort={tc_unsort_cfg is not None})."
        )
      # Highest priority when the flag is on. Skip megablox/tokamax tile padding:
      # TE v1 walks only sum(group_sizes) rows and zeros the leftover.
      # Dtype unification (f32 acts vs bf16 weights) happens inside _te_grouped_dense.
      output = self._te_grouped_dense(inputs, kernel, group_sizes, group_offset, bias)
      if partial_sum is not None:
        output = output + partial_sum
      return output

    tokamax_group_sizes = self._tokamax_group_sizes(group_sizes, inputs, kernel)
    orig_inputs_shape = inputs.shape  # save shape of inputs before potentially padding.
    # Pad only the underlying qvalue buffer to tile size boundaries, leaving scale intact.
    if isinstance(inputs, qpl.QArray):
      padded_qval, padding_amount = max_utils.maybe_pad(inputs.qvalue, forward_tile(self.config, "wi_tile_fwd_batch_seq"))
      inputs = dataclasses.replace(inputs, qvalue=padded_qval)
    else:
      inputs, padding_amount = max_utils.maybe_pad(inputs, forward_tile(self.config, "wi_tile_fwd_batch_seq"))
    if padding_amount > 0 and partial_sum is not None:
      partial_sum = jnp.pad(partial_sum, ((0, padding_amount),) + ((0, 0),) * (partial_sum.ndim - 1))
    if not isinstance(inputs, qpl.QArray):
      inputs = inputs.astype(self.dtype)
    if not is_native_kernel:
      kernel = kernel.astype(self.dtype)
    lhs_quantize_dtype, rhs_quantize_dtype = self._quantization_dtypes()

    # Interpret the megablox Pallas kernel only when the TARGET is NOT TPU (CPU or GPU,
    # e.g. equiv_chunk_test executing locally). During train_compile the local backend is
    # CPU (JAX_PLATFORMS=cpu) but self.mesh targets tpu7x -> compile natively; interpret
    # mode on a TPU target breaks check_vma and bloats HBM temporaries.
    megablox_interpret = self.mesh.devices.flat[0].platform != "tpu"

    # We support various implementations for gmm - TE grouped_dense, tokamax gmm (v1, v2),
    # older forked megablox, or jax.lax.ragged_dot.
    # Determine whether we can use: tokamax gmm v1 (quantized)
    # Pre-quantized QArray inputs must route through gmm_v2 custom VJP.
    is_tokamax_v1_unquantized = (
        self.config.use_tokamax_gmm
        and not self.config.quantization
        and not self.config.use_gmm_v2
        and not isinstance(inputs, qpl.QArray)
        and self.config.hardware != "cpu"
    )
    # Use custom vjp: tokamax gmm v1 (quantized), tokamax gmm v2 (quantized, unquantized), older forked megablox.
    # On CPU (hardware="cpu"), bypass TPU Pallas custom-VJP kernels and fall back to jax.lax.ragged_dot.
    use_custom_vjp_gmm = (self.config.use_tokamax_gmm or self.config.megablox) and self.config.hardware != "cpu"

    if is_tokamax_v1_unquantized:
      # tokamax v1 (unquantized)
      output = tokamax.ragged_dot(
          lhs=inputs,
          rhs=kernel,
          group_sizes=tokamax_group_sizes,
          precision=jax.lax.Precision.DEFAULT,
          preferred_element_type=self.dtype,
          implementation="mosaic",
          # `group_offset` is not yet supported
          group_offset=None,
      )
    elif use_custom_vjp_gmm:
      # tokamax gmm v1 (quantized), tokamax gmm v2 (quantized, unquantized), older forked megablox
      flat_fsdp_shard = None
      if self.config.moe_flat_fsdp_weights and not is_native_kernel and kernel.ndim == 2:
        # `kernel` is the local [rows_local, cols] block of the flat (expert, fsdp)-sharded weight. After the
        # dim-0 FSDP all-gather it holds this shard's E_local experts back to back: a free bitcast to
        # [E_local, rows, cols]. With no FSDP gather axis the local block already is the whole expert group.
        num_local_experts = self.num_local_experts
        gather_factor = 1
        for axis_name, axis_idx in weight_gather_axes:
          if axis_idx == 0:
            gather_factor *= self.mesh.shape[axis_name]
        gathered_rows = kernel.shape[0] * gather_factor
        if gathered_rows % num_local_experts != 0:
          raise ValueError(
              f"moe_flat_fsdp_weights: {gathered_rows} gathered rows not divisible by {num_local_experts} experts"
          )
        gathered_shape = (num_local_experts, gathered_rows // num_local_experts, kernel.shape[1])
        if weight_gather_axes:
          flat_fsdp_shard = gathered_shape
        else:
          kernel = kernel.reshape(gathered_shape)
      output = mblx.gmm(
          lhs=inputs,
          rhs=kernel,
          group_sizes=group_sizes,
          preferred_element_type=self.dtype,
          tiling=tiling,
          group_offset=group_offset,
          lhs_quantize_dtype=lhs_quantize_dtype,
          rhs_quantize_dtype=rhs_quantize_dtype,
          use_qwix_quantization=self.config.use_qwix_quantization,
          use_tokamax_backend=self.config.use_tokamax_gmm,
          weight_gather_axes=weight_gather_axes,
          lhs_vma_axes=lhs_vma_axes,
          rhs_vma_axes=rhs_vma_axes,
          use_gmm_v2=self.config.use_gmm_v2,
          use_gmm_v2_heuristic_tiling=self.config.use_gmm_v2_heuristic_tiling,
          use_dlhs_transpose_rhs=self.config.moe_gmm_v2_dlhs_transpose_rhs,
          partial_sum=partial_sum,
          interpret=megablox_interpret,
          return_lhs=return_lhs,
          fuse_dlhs_scale=self.config.moe_accumulate_wi_dlhs,
          return_rhs=return_rhs,
          flat_fsdp_shard=flat_fsdp_shard,
          fuse_3d_dlhs_ps=self.config.moe_3d_dlhs_partial_sum,
          **({"out_is_3d": True} if out_is_3d else {}),
          **({"tc_routing": tc_routing, "tc_unsort_cfg": tc_unsort_cfg} if tc_unsort_cfg is not None else {}),
      )
    else:
      # jax.lax.ragged_dot
      output = self._ragged_dot(
          inputs,
          kernel,
          tiling,
          group_sizes,
          expert_assignments,
          padding_amount,
      )

    def unpad(t):
      if padding_amount == 0:
        return t
      # Mirror the padding above: only the qvalue of a QArray was padded.
      if isinstance(t, qpl.QArray):
        return dataclasses.replace(t, qvalue=t.qvalue[: orig_inputs_shape[0]])
      return t[: orig_inputs_shape[0]]

    # With tc_unsort_cfg the output is already reduced to token order, so it has no padding rows.
    unpad_out = (lambda t: t) if tc_unsort_cfg is not None else unpad
    # The returned lhs is padded like the output; the returned rhs (weights) is not.
    if return_lhs and return_rhs:
      output, lhs_out, rhs_out = output
      return unpad_out(output), unpad(lhs_out), rhs_out
    if return_lhs:
      output, lhs_out = output
      return unpad_out(output), unpad(lhs_out)
    if return_rhs:
      output, rhs_out = output
      return unpad_out(output), rhs_out
    return unpad_out(output)
