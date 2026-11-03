def validate_data_input_options(self):
  if self.colocated_python_data_input and not self.enable_single_controller:
    raise ValueError(
        "Colocated python data input is only supported with Pathways (single"
        " controller) enabled (`enable_single_controller=True`)."
    )
  if self.grain_use_elastic_iterator and self.grain_file_type != "arrayrecord":
    raise ValueError(
        "`grain_use_elastic_iterator=True` only supports `grain_file_type=arrayrecord`. "
        "tfrecord and parquet pipelines use `InterleaveIterDataset` (a many-to-one "
        "IterDataset transform), which `ElasticIterator` forbids. "
        f"Got grain_file_type={self.grain_file_type}."
    )
  if self.grain_use_elastic_iterator and self.packing:
    raise ValueError("`grain_use_elastic_iterator=True` requires `packing=False`.")


def validate_elastic_iterator_options(self):
  if self.grain_use_elastic_iterator and not self.use_truncation:
    raise ValueError(
        "`grain_use_elastic_iterator=True` requires `use_truncation=True`. "
        "`TokenizeAndChunk` uses `apply`, which produces a many-to-one "
        "IterDataset transform that `ElasticIterator` forbids."
    )
  if self.grain_use_elastic_iterator and (
      self.grain_train_mixture_config_path or ";" in (self.grain_train_files or "")
  ):
    raise ValueError(
        "`grain_use_elastic_iterator=True` does not support dataset mixtures. "
        "Set `grain_train_mixture_config_path` to empty and use a single "
        "`grain_train_files` pattern (no ';' separator)."
    )
