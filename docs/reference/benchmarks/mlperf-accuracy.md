# MLPerf accuracy scoring

Import from `lilax.benchmarks.mlperf.accuracy`. See [accuracy guide](../../guides/benchmarks/mlperf-accuracy.md).

## evaluate_accuracy

```python
evaluate_accuracy(checkpoint_path, mlperf_accuracy_file, dataset_file, *, verbose=False, dtype="int64")
```

Loads a tokenizer from the checkpoint and scores logged token bytes. Metric and NLTK resources may be downloaded.
Scores saved outputs rather than generating predictions.

Returns a dictionary of ROUGE scores and generation statistics.

<details>
<summary>Parameters</summary>

- `checkpoint_path` (`str`). The tokenizer checkpoint or repository.
- `mlperf_accuracy_file` (`str`). The saved MLPerf accuracy JSON log.
- `dataset_file` (`str`). The reference dataset in Parquet, CSV, JSON, or JSONL format.
- `verbose` (`bool`, optional, defaults to `False`). Unused verbosity flag.
- `dtype` (`str`, optional, defaults to `"int64"`). Logged token decoding dtype, `"int32"`, `"int64"`, or `"float"`.

</details>

## evaluate_accuracy_fast

```python
evaluate_accuracy_fast(
	mlperf_accuracy_file, dataset_file, *, checkpoint_path=None, tokenizer_path=None,
	verbose=False, dtype="int64", num_workers=None,
)
```

Scores saved outputs in CPU worker processes. Requires a checkpoint or tokenizer path, otherwise raises `ValueError`.
Metric and NLTK resources may be downloaded.

Returns a dictionary of ROUGE scores and generation statistics.

<details>
<summary>Parameters</summary>

- `mlperf_accuracy_file` (`str`). The saved MLPerf accuracy JSON log.
- `dataset_file` (`str`). The reference dataset in Parquet, CSV, JSON, or JSONL format.
- `checkpoint_path` (`str`, optional, defaults to `None`). AutoTokenizer checkpoint with precedence over `tokenizer_path`.
- `tokenizer_path` (`str`, optional, defaults to `None`). A Llama tokenizer file when no checkpoint is supplied.
- `verbose` (`bool`, optional, defaults to `False`). Unused verbosity flag.
- `dtype` (`str`, optional, defaults to `"int64"`). Logged token decoding dtype, `"int32"`, `"int64"`, or `"float"`.
- `num_workers` (`int`, optional, defaults to `None`). CPU workers. Omission uses available CPU count.

</details>
