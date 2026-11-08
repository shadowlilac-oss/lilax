# Build indexes for Megatron token files

Build indexes for existing Megatron-format `.idx` and `.bin` token files. These operations do not tokenize a Hugging Face dataset.

Public `Trainer` inputs do not consume these mmap indexes. For public training, load a Hugging Face dataset, for example from [exported Parquet files](export.md#export-parquet).

## Write index files

```python
from lilax.data_pipeline.api import convert

paths = convert(["/data/corpus/tokens"], "/data/corpus/indexes", 512, num_epochs=1)
print(paths["sample_index"])
```

Supply the path prefix shared by an existing `.idx` and `.bin` pair, an output directory, and your sequence length. The returned dictionary contains document, sample, shuffle, and metadata paths.

## Build arrays without writing

```python
from lilax.data_pipeline.api import build_indices

document_index, sample_index, shuffle_index, file_hash = build_indices(
	["/data/corpus/tokens"], 512, num_samples=100,
)
print(sample_index.shape)
```

Choose exactly one of `num_samples` and `num_epochs`. The builder indexes full computed epochs and may produce more samples than requested.

The prefix above requires existing `/data/corpus/tokens.idx` and `/data/corpus/tokens.bin`. `sample_index` has shape `(computed_samples + 1, 2)` with document positions and offsets. The other arrays contain document IDs and sample permutations. `file_hash` is a string.

## Blend datasets

Use `convert_blend` with a list of dictionaries containing `input`, `weight`, and `output_dir`. Weights are your task inputs. Zero-weight entries are removed. `build_blend_indices` accepts a semicolon-separated string of `path,weight` entries instead.

Keep `add_extra_token=1` for compatibility with the reader. Other values produce incorrect sample boundaries at runtime.

See [index reference](../../reference/data/mmap-indexes.md).
