# Score generated text with ROUGE

Score response strings against reference strings after generation.

```python
from lilax.evaluation.metrics import score_rouge

responses = ["A checkpoint stores training state."]
references = ["A checkpoint saves the state of training."]
metrics = score_rouge(responses, references)
print(metrics)
```

The two lists must have equal lengths. Scores use a percentage scale and include `gen_num`, the response count. The scorer loads the ROUGE metric and NLTK sentence-tokenization resources.

See [metric reference](../../reference/inference/text-metrics.md) and [generation](generation.md).
