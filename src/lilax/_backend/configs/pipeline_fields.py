"""Native pipeline parallelism configuration fields."""

from pydantic.fields import Field
from pydantic.main import BaseModel


class PipelineParallelism(BaseModel):
  """Configuration for pipeline parallelism."""

  pipeline_fsdp_ag_per_repeat: bool = Field(
      False,
      description="Enable weight prefetching for circular pipeline parallelism.",
  )
  pipeline_bsw_no_carry: bool = Field(
      True,
      description=(
          "With pipeline_fsdp_ag_per_repeat: re-gather the previous repeat's weights in every repeat instead of"
          " carrying the gathered buffer across repeats (saves num_repeats x one repeat of gathered weights in HBM,"
          " costs one extra FSDP all-gather / reduce-scatter per repeat). Numerically identical. False carries the"
          " buffer."
      ),
  )
  pipeline_bsw_phase_split: bool = Field(
      False,
      description=(
          "With pipeline_fsdp_ag_per_repeat: run the iterations in which all stages read the current repeat's weights"
          " without the per-iteration buffer-slot select (no weight copy, one gradient accumulator). Helps when"
          " num_pipeline_microbatches >> num_stages. Numerically identical up to summation order."
      ),
  )
  pipeline_bsw_in_compute_dtype: bool = Field(
      False,
      description=(
          "With pipeline_fsdp_ag_per_repeat: hold the prefetched weights and their per-repeat gradient accumulation in"
          " `dtype` instead of `weight_dtype`. Changes numerics; halves the prefetch all-gather bytes and memory."
      ),
  )
  num_layers_per_pipeline_stage: int = Field(1, description="Number of layers to place on each pipeline stage.")
  num_pipeline_repeats: int = Field(
      -1,
      description="Number of pipeline repeats. Calculated from other params if -1.",
  )
  pipeline_parallel_layers: int = Field(-1, description="Number of layers to pipeline. -1 pipelines all decoder layers.")
  num_pipeline_microbatches: int = Field(
      -1,
      description="Number of microbatches for the pipeline. -1 defaults to num_stages.",
  )
  pipeline_delay_activation_forwarding: bool = Field(
      False, description="Delays activation forwarding to aid XLA optimization."
  )
  pipeline_fsdp_ag_once: bool = Field(False, description="If True, all-gather FSDP weights once per pipeline repeat.")
  scan_pipeline_iterations: bool = Field(True, description="Use jax.lax.scan over pipeline iterations.")
  scan_pipeline_repeats: bool = Field(False, description="Use jax.lax.scan over pipeline repeats.")
  scan_layers_per_stage: bool = Field(False, description="Use jax.lax.scan over layers within a stage.")
  set_remat_policy_on_pipeline_iterations: bool = Field(True, description="Set remat policy on the pipeline scan.")
  set_remat_policy_on_layers_per_stage: bool = Field(False, description="Set remat policy on the inner layer scan.")
