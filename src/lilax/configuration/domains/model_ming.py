"""Ming model configuration."""

from pydantic import BaseModel, Field


class Ming(BaseModel):
  """Configuration specific to inclusionAI Ming (BailingMoeV3) models with Kimi Delta Attention."""

  kda_lower_bound: float = Field(
      -5.0,
      description="Lower bound of the KDA forget gate in log space: g = lower_bound * sigmoid(exp(A_log) * (f + dt_bias)).",
  )
  ming_audio_chunk_size: int = Field(
      3000, description="Number of mel frames per independent whisper audio-encoder chunk (30 s at 100 frames/s)."
  )
  ming_norm_audio_embeds: bool = Field(
      True, description="L2-normalize the projected audio embeddings (norm_query_embeds)."
  )
