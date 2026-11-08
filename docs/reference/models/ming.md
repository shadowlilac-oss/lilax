# Ming audio settings

Return to the [configuration reference](configuration.md).

Import these configurations from `lilax.configuration`. Keep the settings loaded with the matching model family. See [model configuration](../../guides/models/configuration.md) for loading and updating a focused domain.

These are constructor defaults. A loaded checkpoint can supply different values. Changing architecture dimensions or routing does not convert existing weights.

## MingConfig

Import from `lilax.configuration`. Pydantic model whose keyword fields configure Ming audio processing.

<details>
<summary>Parameters</summary>

- `kda_lower_bound` (`float`, optional, defaults to `-5.0`). Lower bound for the KDA forget gate in log space.
- `ming_audio_chunk_size` (`int`, optional, defaults to `3000`). Mel frames per independent Whisper encoder chunk, or 30 seconds at 100 frames per second.
- `ming_norm_audio_embeds` (`bool`, optional, defaults to `True`). L2 normalization of projected audio embeddings (`norm_query_embeds`).

</details>

