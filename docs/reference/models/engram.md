# Engram settings

Return to the [configuration reference](configuration.md).

Import these configurations from `lilax.configuration`. Keep the settings loaded with the matching model family. See [model configuration](../../guides/models/configuration.md) for loading and updating a focused domain.

These are constructor defaults. A loaded checkpoint can supply different values. Changing architecture dimensions or routing does not convert existing weights.

## EngramConfig

Import from `lilax.configuration`. Pydantic model whose keyword fields configure Engram retrieval.

<details>
<summary>Parameters</summary>

- `engram_layers` (`list[int]`, optional, defaults to `[]`). Transformer layer indices where Engram is integrated.
- `engram_num_heads` (`int`, optional, defaults to `8`). Number of Engram heads.
- `engram_head_dim` (`int`, optional, defaults to `1280`). Width of each head's retrieved embedding.
- `engram_vocab_bases` (`list[int]`, optional, defaults to `[]`). Minimum head vocabulary sizes per n-gram order.
- `engram_max_ngram_size` (`int`, optional, defaults to `3`). Maximum n-gram order.
- `engram_kernel_size` (`int`, optional, defaults to `4`). Temporal convolution window.
- `engram_seed` (`int`, optional, defaults to `0`). Hash mapping seed.

</details>

