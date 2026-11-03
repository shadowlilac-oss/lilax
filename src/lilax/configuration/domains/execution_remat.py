from pydantic import BaseModel, Field

from lilax.configuration.domains.model_moe import RematLocation


class RematAndOffload(BaseModel):
  """Configuration for gradient checkpointing (rematerialization) and offloading."""

  remat_policy: str | None = Field("full", description="The rematerialization policy, trading off speed and memory.")
  remat_policy_for_vit: str | None = Field("minimal", description="Remat policy for multimodal model's vision encoder.")
  decoder_layer_input: RematLocation = Field(RematLocation.DEVICE, description="Remat policy for the decoder layer's input.")
  indexer_cutoff_threshold: RematLocation = Field(
      RematLocation.REMAT, description="Remat policy for the indexer cutoff threshold (shape: [batch, seq_len])."
  )
  context: RematLocation = Field(RematLocation.REMAT, description="Remat policy for the attention context.")
  mlpwi: RematLocation = Field(RematLocation.REMAT, description="Remat policy for the first MLP layer's intermediate output.")
  mlpwi_0: RematLocation = Field(RematLocation.REMAT, description="Remat policy for the first part of a gated MLP's output.")
  mlpwi_1: RematLocation = Field(RematLocation.REMAT, description="Remat policy for the second part of a gated MLP's output.")
  moe_x_sorted: RematLocation = Field(
      RematLocation.REMAT,
      description=(
          "Remat policy for the routed (post-dispatch, expert-sorted) MoE input plus its small "
          "routing/metadata bundle. 'device' saves them across the remat boundary so the backward "
          "does not re-run the dispatch token all-gather and ragged sort; the expert GMMs re-run "
          "from the saved tensor. Default 'remat' recomputes (existing behavior)."
      ),
  )
  moe_routing_maps: RematLocation = Field(
      RematLocation.REMAT,
      description=(
          "Remat policy for the MoE routing maps (top-k expert ids, ragged-sort permutations, sorted token indices "
          "and group sizes)."
      ),
  )
  moe_tc_routing_checkpoint: RematLocation = Field(
      RematLocation.REMAT,
      description=(
          "Remat policy for the TC per-buffer-row routing weights (moe_tc_ragged_sort with "
          "moe_tc_ragged_weights_on_activation); 'device' saves them so the backward skips the weight gather."
      ),
  )
  mlpwo: RematLocation = Field(RematLocation.REMAT, description="Remat policy for the second MLP layer's output.")
  te_moe_dispatch_remat_location: RematLocation = Field(
      RematLocation.REMAT, description="Remat policy for TE EP dispatch outputs: recompute, save on device, or offload."
  )
  moe_mlpwi_0: RematLocation = Field(RematLocation.REMAT, description="Remat policy for the first part of a gated MoE's output.")
  moe_mlpwi_1: RematLocation = Field(RematLocation.REMAT, description="Remat policy for the second part of a gated MoE's output.")
  moe_mlpwo: RematLocation = Field(RematLocation.REMAT, description="Remat policy for the second MoE layer's output.")
  query_proj: RematLocation = Field(RematLocation.REMAT, description="Remat policy for the query projection.")
  key_proj: RematLocation = Field(RematLocation.REMAT, description="Remat policy for the key projection.")
  value_proj: RematLocation = Field(RematLocation.REMAT, description="Remat policy for the value projection.")
  kv_proj: RematLocation = Field(RematLocation.REMAT, description="Remat policy for the unified KV projection.")
  query_wa_proj: RematLocation = Field(RematLocation.REMAT, description="Remat policy for the MLA query weighted attention projection.")
  kv_wa_proj: RematLocation = Field(RematLocation.REMAT, description="Remat policy for the MLA key and value weighted attention projection.")
  qkv_proj: RematLocation = Field(RematLocation.REMAT, description="Remat policy for fused QKV projection.")
  out_proj: RematLocation = Field(RematLocation.REMAT, description="Remat policy for the attention output projection.")
  mla_q: RematLocation = Field(RematLocation.REMAT, description="Remat policy for the mla's query projectiont.")
  mla_kv: RematLocation = Field(RematLocation.REMAT, description="Remat policy for the mla's key and value projection.")
  attention_out: RematLocation = Field(RematLocation.REMAT, description="Remat policy for the attention output.")
  engram: RematLocation = Field(RematLocation.REMAT, description="Remat policy for the engram output.")
  optimizer_memory_host_offload: bool = Field(False, description="Offload optimizer state to host memory.")
  parameter_memory_host_offload: bool = Field(False, description="Offload parameters to host memory.")
