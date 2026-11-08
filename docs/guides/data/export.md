# Export Hub datasets to files

Export a Hub dataset to Parquet or ArrayRecord for file-based workflows. These functions accept a repository ID rather than an in-memory `Dataset`.

## Export Parquet

```python
from lilax.data_operations.hf_export import export_parquet

if __name__ == "__main__":
	export_parquet("trl-lib/Capybara", "./capybara-parquet", num_files=4, workers=1)
```

The main guard protects multiprocessing startup. File counts and worker counts illustrate the interface.

Load the exported files as the next step for a public Trainer workflow:

```python
from lilax.data import load_dataset

dataset = load_dataset("parquet", data_files="./capybara-parquet/*.parquet", split="train")
```

## Export ArrayRecord

Use a fresh output directory for an independent export. Existing files with matching shard names can be overwritten.

```python
from lilax.data_operations.hf_export import export_arrayrecord

if __name__ == "__main__":
	export_arrayrecord("trl-lib/Capybara", "./capybara-arrayrecord", workers=1)
```

Use a local directory or a GCSFUSE mount for ArrayRecord. A `gs://` output raises `ValueError`. Parquet export accepts cloud paths through its filesystem handling.

Public Trainer inputs do not consume these generic ArrayRecord exports. `TeacherLogitsDataset` has a separate teacher-logits record contract. Use Parquet and `load_dataset` for ordinary public training.

See [file export reference](../../reference/data/data-export.md) for compression and shard settings.
