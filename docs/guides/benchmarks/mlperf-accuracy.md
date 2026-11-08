# Score an MLPerf accuracy log

Score saved generated token IDs against the processed reference dataset. This operation does not run a new model benchmark.

```python
from lilax.benchmarks.mlperf.accuracy import evaluate_accuracy

metrics = evaluate_accuracy(
	"/checkpoints/matching-tokenizer", "/results/mlperf_log_accuracy.json", "/data/references.parquet",
)
print(metrics["rouge1"], metrics["tokens_per_sample"])
```

The tokenizer checkpoint must match the logged token IDs. `accuracy_log` is the MLPerf accuracy JSON file. The reference file supplies the ground-truth records used by that benchmark.

The log is a JSON list of records with `qsl_idx` and `data`. `data` is a hex string encoding generated token IDs, interpreted as `int64` by default. The reference file needs an `output` column indexed by `qsl_idx`. The returned dictionary contains `rouge1`, `rouge2`, `rougeL`, `rougeLsum`, `gen_len`, `gen_num`, `gen_tok_len`, and `tokens_per_sample`. The ROUGE values are percentages, not raw fractions.

## Score in parallel

```python
from lilax.benchmarks.mlperf.accuracy import evaluate_accuracy_fast

metrics = evaluate_accuracy_fast(
	"/results/mlperf_log_accuracy.json", "/data/references.parquet",
	checkpoint_path="/checkpoints/matching-tokenizer",
)
```

The scorers load tokenization and metric resources and report ROUGE and generation statistics. See [MLPerf reference](../../reference/benchmarks/mlperf-accuracy.md).
