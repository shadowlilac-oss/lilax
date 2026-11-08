# Megatron index builders

Import from `lilax.data_pipeline.api`. See [index guide](../../guides/data/mmap-indexes.md).

## Discovery and complete indexes

Inputs require Megatron `MMIDIDX` index headers and binary token storage.

### resolve_shard_prefixes

```python
resolve_shard_prefixes(paths)
```

Returns sorted prefixes. `discover_shards` is an alias. Missing indexes raise `FileNotFoundError`.

<details>
<summary>Parameters</summary>

- `paths` (`str | list[str]`). Prefixes or directories to discover.

</details>

### get_document_sizes

```python
get_document_sizes(path_prefixes: list[str]) -> np.ndarray
```

Returns a NumPy array of document sizes.

<details>
<summary>Parameters</summary>

- `path_prefixes` (`list[str]`). Shard prefixes.

</details>

### build_indices

```python
build_indices(
	input_paths, seq_length, num_samples=None, num_epochs=None, seed=1234,
	split=None, split_index=0, add_extra_token=1,
)
```

Returns `(document_index, sample_index, shuffle_index, file_hash)`. Exactly one duration option is required. Empty documents or tokens raise `ValueError`. Sample indexes contain `(num_samples + 1, 2)` boundaries with document-index positions and offsets.

<details>
<summary>Parameters</summary>

- `input_paths` (`list[str]`). Input shards.
- `seq_length` (`int`). Sample sequence length.
- `num_samples` (`int | None`, optional, defaults to `None`). Requested samples. Mutually exclusive with epochs.
- `num_epochs` (`int | None`, optional, defaults to `None`). Requested epochs. Mutually exclusive with samples.
- `seed` (`int`, optional, defaults to `1234`). Shuffle seed.
- `split` (`str | None`, optional, defaults to `None`). Comma-separated split ratios.
- `split_index` (`int`, optional, defaults to `0`). Selected split.
- `add_extra_token` (`int`, optional, defaults to `1`). Extra next-token target. The reader requires one.

</details>

### convert

```python
convert(
	input_paths, output_dir, seq_length, num_samples=None, num_epochs=None, seed=1234,
	split=None, split_index=0, add_extra_token=1,
)
```

Writes arrays and metadata using `build_indices`, with the same duration and input requirements.
Returns a dictionary with `document_index`, `sample_index`, `shuffle_index`, and `metadata` paths.

<details>
<summary>Parameters</summary>

- `input_paths` (`list[str]`). Input shards.
- `output_dir` (`str | os.PathLike`). Index output directory.
- `seq_length` (`int`). Sample length.
- `num_samples` (`int | None`, optional, defaults to `None`). Requested samples, instead of epochs.
- `num_epochs` (`int | None`, optional, defaults to `None`). Requested epochs, instead of samples.
- `seed` (`int`, optional, defaults to `1234`). Shuffle seed.
- `split` (`str | None`, optional, defaults to `None`). Comma-separated split ratios.
- `split_index` (`int`, optional, defaults to `0`). Selected split.
- `add_extra_token` (`int`, optional, defaults to `1`). Extra next-token target.

</details>

## Individual arrays

### build_document_index

```python
build_document_index(num_docs, num_epochs, seed, separate_last_epoch=False)
```

Returns shuffled document IDs.

<details>
<summary>Parameters</summary>

- `num_docs` (`int`). Document count.
- `num_epochs` (`int`). Epoch count.
- `seed` (`int`). Shuffle seed.
- `separate_last_epoch` (`bool`, optional, defaults to `False`). Shuffle the final epoch separately.

</details>

### build_sample_index

```python
build_sample_index(doc_sizes, doc_index, seq_length, drop_last=True, add_extra_token=1)
```

Returns token boundaries as `(document-index position, token offset)` rows.

<details>
<summary>Parameters</summary>

- `doc_sizes` (`np.ndarray`). Document token counts.
- `doc_index` (`np.ndarray`). Ordered document IDs.
- `seq_length` (`int`). Sample length.
- `drop_last` (`bool`, optional, defaults to `True`). Omit an incomplete final sample.
- `add_extra_token` (`int`, optional, defaults to `1`). Extra next-token target.

</details>

### build_shuffle_index

```python
build_shuffle_index(num_samples, total_size, seed)
```

Returns a sample permutation.

<details>
<summary>Parameters</summary>

- `num_samples` (`int`). Initial sample range shuffled independently.
- `total_size` (`int`). Total range, including separately shuffled remaining samples.
- `seed` (`int`). Shuffle seed.

</details>

### compute_num_epochs

```python
compute_num_epochs(total_tokens, num_samples, seq_length, add_extra_token=1)
```

Returns epochs required for the sample request.

<details>
<summary>Parameters</summary>

- `total_tokens` (`int`). Tokens per epoch.
- `num_samples` (`int`). Requested samples.
- `seq_length` (`int`). Sample length.
- `add_extra_token` (`int`, optional, defaults to `1`). Extra next-token target.

</details>

### should_separate_last_epoch

```python
should_separate_last_epoch(
	num_epochs, tokens_per_epoch, num_samples, seq_length, add_extra_token=1, threshold=0.80,
)
```

Returns whether to shuffle the final epoch separately.

<details>
<summary>Parameters</summary>

- `num_epochs` (`int`). Epoch count.
- `tokens_per_epoch` (`int`). Token count per epoch.
- `num_samples` (`int`). Requested samples.
- `seq_length` (`int`). Sample length.
- `add_extra_token` (`int`, optional, defaults to `1`). Extra target token.
- `threshold` (`float`, optional, defaults to `0.80`). Final-epoch sample fraction threshold.

</details>

### compute_index_hash

```python
compute_index_hash(
	input_paths, num_epochs, separate_final_epoch, seed, seq_length, split=None, split_index=0, add_extra_token=1,
)
```

Returns a parameter-derived MD5 string.

<details>
<summary>Parameters</summary>

- `input_paths` (`list[str] | list[list[str]]`). Input shard identities.
- `num_epochs` (`int`). Epoch count.
- `separate_final_epoch` (`bool`). Final-epoch separation choice.
- `seed` (`int`). Shuffle seed.
- `seq_length` (`int`). Sample length.
- `split` (`str | None`, optional, defaults to `None`). Split ratios.
- `split_index` (`int`, optional, defaults to `0`). Selected split.
- `add_extra_token` (`int`, optional, defaults to `1`). Extra target token.

</details>

## Blends

### convert_blend

```python
convert_blend(
	dataset_specs, total_samples, seq_length, seed=1234, margin=0.5, max_workers=None,
	split=None, split_index=0, add_extra_token=1, blend_index_output_dir=None,
)
```

Writes blended indexes. Specifications must be nonempty, weights nonnegative, and total weight positive.
Returns per-dataset dictionaries with `paths`, normalized `weight`, and `buffer_samples`.

<details>
<summary>Parameters</summary>

- `dataset_specs` (`list[dict]`). Each has `input`, `weight`, and `output_dir`.
- `total_samples` (`int`). Requested training samples.
- `seq_length` (`int`). Sample length.
- `seed` (`int`, optional, defaults to `1234`). Shuffle seed.
- `margin` (`float`, optional, defaults to `0.5`). Overprovision percentage. `0.5` means half a percent.
- `max_workers` (`int | None`, optional, defaults to `None`). Concurrent builds. Default is dataset count.
- `split` (`str | None`, optional, defaults to `None`). Split ratios.
- `split_index` (`int`, optional, defaults to `0`). Selected split.
- `add_extra_token` (`int`, optional, defaults to `1`). Extra target token. The reader requires one.
- `blend_index_output_dir` (`str | os.PathLike | None`, optional, defaults to `None`). Also write global blend dispatch arrays here when set.

</details>

### build_blend_indices

```python
build_blend_indices(
	datasets, output_dir, seq_length, total_samples, seed=1234, margin=0.5,
	max_workers=None, split=None, split_index=0, add_extra_token=1,
)
```

Parses `"path,weight;path,weight"`, selects child directories, and writes blended indexes.
Returns the `convert_blend` result, subject to its weight restrictions.

<details>
<summary>Parameters</summary>

- `datasets` (`str`). Semicolon-separated path and weight pairs.
- `output_dir` (`str | os.PathLike`). Parent output directory for children and blend dispatch.
- `seq_length` (`int`). Sample length.
- `total_samples` (`int`). Requested samples.
- `seed` (`int`, optional, defaults to `1234`). Shuffle seed.
- `margin` (`float`, optional, defaults to `0.5`). Overprovision percentage.
- `max_workers` (`int | None`, optional, defaults to `None`). Concurrent builds.
- `split` (`str | None`, optional, defaults to `None`). Split ratios.
- `split_index` (`int`, optional, defaults to `0`). Selected split.
- `add_extra_token` (`int`, optional, defaults to `1`). Extra target token. The reader requires one.

</details>
