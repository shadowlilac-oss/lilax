"""Map input arguments to configuration overrides."""

MMAP_FIELDS = (
    "mmap_eod_id", "blend_cache_dir", "blend_index_dir", "reset_attention_mask", "eod_mask_loss",
    "packing_max_segments_per_sample", "mmap_split_sentences", "mmap_npy_split",
)

TFDS_C4_FIELDS = ("use_stream_chunking", "train_shard_in_read", "train_interleave_cycle_length")


def input_overrides(args):
  return (
      ("dataset_type", args.dataset_type), ("grain_file_type", args.grain_file_type),
      ("grain_train_files", args.grain_train_files), ("grain_eval_files", args.grain_eval_files),
      ("grain_ram_budget_mb", args.grain_ram_budget_mb),
      ("grain_index_storage_option", args.grain_index_storage_option),
      ("colocated_python_data_input", args.colocated_python_data_input),
      ("expansion_factor_real_data", args.expansion_factor_real_data),
      ("generate_padding_batch_train", args.generate_padding_batch_train),
      ("generate_padding_batch_eval", args.generate_padding_batch_eval),
  )


def input_field_overrides(args):
  return {name: getattr(args, name) for name in MMAP_FIELDS + TFDS_C4_FIELDS if getattr(args, name) is not None}
