import logging

from pydantic import BaseModel, Field, NonNegativeInt, model_validator

logger = logging.getLogger("lilax._backend.configs.inert")


class DilocoParams(BaseModel):
  """Diloco Hyperparameters"""

  enable_diloco: bool = Field(False, description="Enable Diloco parallelism")
  diloco_sync_period: int = Field(36, description="Diloco sync period.")
  diloco_outer_lr: float = Field(0.3, description="learning rate for outer optimizer.")
  diloco_outer_momentum: float = Field(0.9, description="momentum for outer optimizer.")
  dcn_bandwidth_limit: str = Field("", description="Programmatic DCN egress bandwidth limit per VM (e.g., '28gbit'). Empty means no limit.")
  dcn_bandwidth_burst: str = Field("10mb", description="Burst size for Token Bucket Filter (TBF) traffic shaping.")
  dcn_bandwidth_latency: str = Field("50ms", description="Latency threshold for Token Bucket Filter (TBF) traffic shaping.")
  dcn_bandwidth_interface: str = Field("eth0", description="Network interface to apply bandwidth limits on.")
  enable_streaming_diloco: bool = Field(False, description=(
      "Enable streaming DiLoCo parallelism (https://arxiv.org/abs/2501.18512). Streaming DiLoCo partitions"
      " model parameters into fragments and pipelines cross-island synchronization one fragment per inner step,"
      " overlapping inter-cluster communication with accelerator computation."
  ))
  num_diloco_fragments: int | None = Field(None, description=(
      "Total number of fragments to partition the model layers into (including 1 fragment for non-scanned"
      " parameters). Required when enable_streaming_diloco is True."
  ))
  use_sequential_layers: bool = Field(False, description="Whether to sync layers sequentially (or interleaved).")
  diloco_bucketize_non_scanned: bool = Field(False, description=(
      "Split non-scanned matrices (ndim >= 2, e.g. embedding table, output projection) across the layer"
      " fragments instead of syncing them whole in fragment 0, so every streaming DiLoCo step moves a similar"
      " number of bytes. Each matrix is split along its longest unsharded axis with at least"
      " num_diloco_fragments - 1 indices (vocab under FSDP), whatever its size; a matrix whose long axes are all"
      " sharded (e.g. vocab under tensor parallelism) stays whole in fragment 0. Requires"
      " num_diloco_fragments >= 3."
  ))
  num_communication_overlapping_steps: NonNegativeInt = Field(0, description="Steps of communication overlap with computation. \\tau from the paper.")
  communication_overlapping_alpha: float = Field(0.0, ge=0.0, le=1.0, description=(
      "Interpolation factor between local and global parameters. alpha=1"
      " means no communication between islands, alpha=0 means discards any"
      " updates done in the inner optimizer in the first"
      " `num_communication_overlapping_steps` steps. alpha=0.5 does a"
      " uniform average between the local fragment parameters and the"
      " globally shared one."
  ))

  @model_validator(mode="after")
  def validate_streaming_diloco_params(self) -> "DilocoParams":
    if self.enable_streaming_diloco:
      if not self.enable_diloco:
        raise ValueError("enable_diloco must be True when enable_streaming_diloco is True.")
      if self.num_diloco_fragments is None:
        raise ValueError("num_diloco_fragments must be specified when enable_streaming_diloco is True.")
      if self.num_diloco_fragments < 2:
        raise ValueError(
            f"num_diloco_fragments ({self.num_diloco_fragments}) must be at least 2 when enable_streaming_diloco "
            "is True (1 for non-scanned parameters, at least 1 for scanned layers)."
        )
      if self.diloco_sync_period % self.num_diloco_fragments != 0:
        steps_between_syncs = max(1, int(round(self.diloco_sync_period / self.num_diloco_fragments)))
        logger.warning(
            f"diloco_sync_period ({self.diloco_sync_period}) is not a multiple of num_diloco_fragments"
            f" ({self.num_diloco_fragments}): streaming DiLoCo syncs a fragment every {steps_between_syncs} steps, so"
            f" each fragment is synced every {steps_between_syncs * self.num_diloco_fragments} steps."
        )
    if self.diloco_bucketize_non_scanned and not self.enable_streaming_diloco:
      raise ValueError("diloco_bucketize_non_scanned requires enable_streaming_diloco=True.")
    if self.diloco_bucketize_non_scanned and self.num_diloco_fragments < 3:
      raise ValueError(
          f"diloco_bucketize_non_scanned requires num_diloco_fragments >= 3, got {self.num_diloco_fragments}: with a"
          " single layer fragment, bucketized matrices would move whole into it."
      )
    return self
