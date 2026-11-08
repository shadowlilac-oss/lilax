# File exports

Import from `lilax.data_operations.hf_export`. See [export guide](../../guides/data/export.md).

## export_arrayrecord

```python
export_arrayrecord(
	dataset, output, config=None, split="train", name_prefix=None, file_size_mb=500,
	group_size=1, workers=None, token=None,
)
```

Streams a Hub dataset into ArrayRecord shards. `gs://` output paths raise `ValueError`.
Existing files with matching shard names can be overwritten. A fresh output directory keeps independent exports separate.

Returns `None`.

<details>
<summary>Parameters</summary>

- `dataset` (`str`). Hub repository ID.
- `output` (`str`). Local or GCSFUSE-mounted output directory.
- `config` (`str | None`, optional, defaults to `None`). JSON-encoded `load_dataset` keyword object, not a bare configuration name or Python dictionary. `'{"name": "main"}'` selects configuration `main`.
- `split` (`str`, optional, defaults to `"train"`). Dataset split.
- `name_prefix` (`str | None`, optional, defaults to `None`). Shard prefix. `None` derives it from the repository name.
- `file_size_mb` (`int`, optional, defaults to `500`). Target shard size in decimal megabytes.
- `group_size` (`int`, optional, defaults to `1`). ArrayRecord record group size.
- `workers` (`int | None`, optional, defaults to `None`). Worker count. The default uses CPU count, capped by available dataset shards.
- `token` (`str | bool | None`, optional, defaults to `None`). Hub authentication passed to dataset loading.

</details>

## export_parquet

```python
export_parquet(
	dataset, output, num_files, config=None, split="train", name_prefix=None, compression="zstd",
	row_group_size=None, total_size_gb=None, workers=None, token=None,
)
```

Streams a Hub dataset into Parquet files. Unsupported compression raises `ValueError`.

Returns `None`.

`config` uses the same JSON-encoded keyword-argument object as `export_arrayrecord`. For a repository with a configuration named `main`:

```python
from lilax.data_operations.hf_export import export_parquet

export_parquet("your-org/your-dataset", "./export", num_files=2, config='{"name": "main"}')
```

Replace the repository and configuration name with your dataset's values.

<details>
<summary>Parameters</summary>

- `dataset` (`str`). Hub repository ID.
- `output` (`str`). Local, GCSFUSE-mounted, or GCS output directory.
- `num_files` (`int`). Requested output file count.
- `config` (`str | None`, optional, defaults to `None`). JSON-encoded dataset keyword object, as for `export_arrayrecord`.
- `split` (`str`, optional, defaults to `"train"`). Dataset split.
- `name_prefix` (`str | None`, optional, defaults to `None`). Shard prefix. `None` derives it from the repository name.
- `compression` (`str`, optional, defaults to `"zstd"`). One of `"zstd"`, `"snappy"`, `"gzip"`, or `"none"`.
- `row_group_size` (`int | None`, optional, defaults to `None`). Parquet row group size.
- `total_size_gb` (`float | None`, optional, defaults to `None`). Estimated source size in decimal gigabytes. Without it, size is estimated from repository files.
- `workers` (`int | None`, optional, defaults to `None`). CPU worker count, capped by dataset shards.
- `token` (`str | bool | None`, optional, defaults to `None`). Hub authentication.

</details>
