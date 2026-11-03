"""Native MoE grouped-matmul kernel and tiling configuration fields."""

from pydantic.fields import Field
from pydantic.main import BaseModel


class MoEKernels(BaseModel):
  """Configuration for MoE-specific kernels like Megablox."""

  megablox: bool = Field(
      True,
      description="Whether to use Megablox kernels for MoE. Automatically disabled when hardware='cpu'.",
  )
  sparse_matmul: bool = Field(True, description="Whether to use sparse matmul kernels for MoE.")
  wi_tile_fwd_batch_seq: int = Field(
      512,
      description="forward pass tiling dimension for batch/sequence in GMM for wi.",
  )
  wi_tile_fwd_embed_dim: int = Field(1024, description="forward pass tiling dimension for embedding in GMM for wi.")
  wi_tile_fwd_mlp_dim: int = Field(1024, description="forward pass tiling dimension for MLP in GMM for wi.")
  eval_wi_tile_fwd_batch_seq: int = Field(
      512,
      description="evaluation forward pass tiling dimension for batch/sequence in GMM for wi.",
  )
  eval_wi_tile_fwd_embed_dim: int = Field(
      1024, description="evaluation forward pass tiling dimension for embedding in GMM for wi."
  )
  eval_wi_tile_fwd_mlp_dim: int = Field(
      1024, description="evaluation forward pass tiling dimension for MLP in GMM for wi."
  )
  wi_tile_dlhs_batch_seq: int = Field(
      512,
      description="bwd pass dlhs tiling dimension for batch/sequence in GMM for wi.",
  )
  wi_tile_dlhs_embed_dim: int = Field(1024, description="bwd pass dlhs tiling dimension for embedding in GMM for wi.")
  wi_tile_dlhs_mlp_dim: int = Field(1024, description="bwd pass dlhs tiling dimension for MLP in GMM for wi.")
  wi_tile_drhs_batch_seq: int = Field(
      512,
      description="bwd pass drhs tiling dimension for batch/sequence in GMM for wi.",
  )
  wi_tile_drhs_embed_dim: int = Field(1024, description="bwd pass drhs tiling dimension for embedding in GMM for wi.")
  wi_tile_drhs_mlp_dim: int = Field(1024, description="bwd pass drhs tiling dimension for MLP in GMM for wi.")
  wo_tile_fwd_batch_seq: int = Field(
      512,
      description="forward pass tiling dimension for batch/sequence in GMM for wo.",
  )
  wo_tile_fwd_embed_dim: int = Field(1024, description="forward pass tiling dimension for embedding in GMM for wo.")
  wo_tile_fwd_mlp_dim: int = Field(1024, description="forward pass tiling dimension for MLP in GMM for wo.")
  eval_wo_tile_fwd_batch_seq: int = Field(
      512,
      description="evaluation forward pass tiling dimension for batch/sequence in GMM for wo.",
  )
  eval_wo_tile_fwd_embed_dim: int = Field(
      1024, description="evaluation forward pass tiling dimension for embedding in GMM for wo."
  )
  eval_wo_tile_fwd_mlp_dim: int = Field(
      1024, description="evaluation forward pass tiling dimension for MLP in GMM for wo."
  )
  wo_tile_dlhs_batch_seq: int = Field(
      512,
      description="bwd pass dlhs tiling dimension for batch/sequence in GMM for wo.",
  )
  wo_tile_dlhs_embed_dim: int = Field(1024, description="bwd pass dlhs tiling dimension for embedding in GMM for wo.")
  wo_tile_dlhs_mlp_dim: int = Field(1024, description="bwd pass dlhs tiling dimension for MLP in GMM for wo.")
  wo_tile_drhs_batch_seq: int = Field(
      512,
      description="bwd pass drhs tiling dimension for batch/sequence in GMM for wo.",
  )
  wo_tile_drhs_embed_dim: int = Field(1024, description="bwd pass drhs tiling dimension for embedding in GMM for wo.")
  wo_tile_drhs_mlp_dim: int = Field(1024, description="bwd pass drhs tiling dimension for MLP in GMM for wo.")

  merge_gating_gmm: bool = Field(False, description="whether to merge the two gating gmm kernels into one.")

  num_moe_emb_chunks: int = Field(
      0, description="Number of chunks for overlapping token all-gather and GMM computation along embedding dimension."
  )

  # tokamax gmm
  use_tokamax_gmm: bool = Field(
      False,
      description="Whether to use the Tokamax library for GMM kernel implementation.",
  )

  use_gmm_v2: bool = Field(
      False,
      description="Whether to use Tokamax GMM v2 for MoE kernel.",
  )

  use_te_grouped_gemm: bool = Field(
      False,
      description=(
          "Whether to use TransformerEngine grouped_dense for sparse MoE grouped GEMM on GPU. "
          "Incompatible with megablox and use_tokamax_gmm; quantization is not supported yet. "
          "Requires one local device per process: TE v1 copies group_sizes D2H and deadlocks "
          "in single-process multi-device shard_map."
      ),
  )

  use_gmm_v2_heuristic_tiling: bool = Field(
      False,
      description="Whether to use the heuristic tiling from Tokamax GMM v2, when use_gmm_v2=true.",
  )
