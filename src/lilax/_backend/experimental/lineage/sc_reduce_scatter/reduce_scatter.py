"""FP8 reduce-scatter over a chip ring and die pair on SparseCore 0."""

import functools

import jax
import jax.experimental.pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jaxtyping as jt
import typeguard

from lilax._backend.experimental.lineage.sc_reduce_scatter import buffers
from lilax._backend.experimental.lineage.sc_reduce_scatter import plan as plan_lib
from lilax._backend.experimental.lineage.sc_reduce_scatter import scalar_subcore
from lilax._backend.experimental.lineage.sc_reduce_scatter import vector_subcore


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def reduce_scatter_sc0(
    x: jt.Num[jax.Array, "num_tokens d0 d1"],
    *,
    token_amax: jt.Float[jax.Array, "num_tokens"] | None = None,
    per_hop_scaling: bool = False,
    ring_axis: str = "z",
    core_axis: str = "core",
    num_microbatches: int = 2,
    vmem_tile_rows: int | None = None,
    collective_id: int = 0,
) -> jt.Num[jax.Array, "out_tokens d0 d1"]:
  """Reduce-scatter tokens over a chip ring and die axis on SparseCore 0.

  Ring traffic uses FP8. Optional global or per-hop token scaling preserves
  the dynamic range of bfloat16 inputs. The output slice matches a tiled
  psum_scatter over (ring_axis, core_axis).
  """
  plan = plan_lib.make_reduce_scatter_plan(
      x, has_amax_input=token_amax is not None, per_hop_scaling=per_hop_scaling, ring_axis=ring_axis,
      core_axis=core_axis, num_microbatches=num_microbatches, vmem_tile_rows=vmem_tile_rows,
  )
  scs_mesh = plsc.ScalarSubcoreMesh(axis_name="sc_core", num_cores=1)
  tec_mesh = plsc.VectorSubcoreMesh(
      core_axis_name="sc_core", subcore_axis_name="subcore", num_cores=1, num_subcores=plan_lib.NUM_SUBCORES,
  )
  x_2d = x.reshape(plan.num_tokens * plan.dim0, plan.dim1)
  manual_axis_type = getattr(jax.typeof(x), "manual_axis_type", None)
  inputs = (x_2d,)
  if plan.has_amax_input:
    inputs += (token_amax,)
  out_2d, *_ = pl.kernel(
      body=[functools.partial(scalar_subcore.scalar_subcore_kernel, plan),
            functools.partial(vector_subcore.vector_subcore_kernel, plan)],
      mesh=[scs_mesh, tec_mesh], out_type=buffers.kernel_out_types(plan, x.dtype, manual_axis_type),
      scratch_types=buffers.kernel_scratch_types(plan, x.dtype, scs_mesh, tec_mesh),
      compiler_params=pltpu.CompilerParams(collective_id=collective_id, use_tc_tiling_on_sc=True),
      name="reduce_scatter_sc0",
  )(*inputs)
  return out_2d.reshape(plan.out_tokens, plan.dim0, plan.dim1)
