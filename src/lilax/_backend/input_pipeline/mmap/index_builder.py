"""Build Megatron-LM indices for individual datasets and blends."""

import os

from lilax._backend.input_pipeline.mmap._mmap_index_utils import (
    build_document_index,
    build_indices,
    build_sample_index,
    build_shuffle_index,
    compute_index_hash,
    compute_num_epochs,
    convert,
    convert_blend,
    get_document_sizes,
    resolve_shard_prefixes,
    should_separate_last_epoch,
)

discover_shards = resolve_shard_prefixes


def build_blend_indices(
    datasets, output_dir, seq_length, total_samples, seed=1234, margin=0.5,
    max_workers=None, split=None, split_index=0, add_extra_token=1,
):
  """Build child and dispatch indices from semicolon-separated path and weight pairs."""
  dataset_specs = []
  for i, raw in enumerate(datasets.strip().split(";")):
    raw = raw.strip()
    if not raw:
      continue
    parts = raw.rsplit(",", 1)
    if len(parts) != 2:
      raise ValueError(f"Invalid dataset spec '{raw}': expected 'input_path,weight'")
    input_path, weight_str = parts[0].strip(), parts[1].strip()
    dataset_specs.append(
        {"input": [input_path], "weight": float(weight_str), "output_dir": os.path.join(output_dir, f"dataset_{i}")}
    )

  return convert_blend(
      dataset_specs=dataset_specs,
      total_samples=total_samples,
      seq_length=seq_length,
      seed=seed,
      margin=margin,
      max_workers=max_workers,
      split=split,
      split_index=split_index,
      add_extra_token=add_extra_token,
      blend_index_output_dir=output_dir,
  )
