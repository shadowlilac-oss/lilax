# Text metrics

Import from `lilax.evaluation.metrics`.

## score_rouge

```python
score_rouge(responses: list[str], references: list[str], use_stemmer: bool = True) -> dict
```

Computes ROUGE scores. Unequal input lengths raise `ValueError`.

Responses and references are stripped and split into sentences before scoring. Returned scores are rounded percentage values. The current implementation accepts `use_stemmer` but does not forward it to the metric.

See the [ROUGE workflow](../../guides/inference/text-metrics.md).

Returns a dictionary of ROUGE scores and `gen_num`, the prediction count. Importing the scorer requests NLTK sentence-tokenizer data. Scoring loads the upstream `rouge` metric.

<details>
<summary>Parameters</summary>

- `responses` (`list[str]`). Generated text.
- `references` (`list[str]`). Corresponding reference text.
- `use_stemmer` (`bool`, optional, defaults to `True`). Accepted but not forwarded to the metric.

</details>
