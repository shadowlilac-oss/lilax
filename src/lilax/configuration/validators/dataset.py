from lilax.configuration.domain_types import DatasetType


def validate_dataset_options(self):
  if self.dataset_type == DatasetType.HF:
    if self.hf_eval_files:
      self.hf_eval_split = "train"
    if self.grain_worker_count > 1:
      raise ValueError("Only supports <= 1 for now, more workers results in duplicated data")
  elif self.dataset_type == DatasetType.GRAIN:
    use_hf_parquet = self.hf_path and self.grain_file_type == "parquet"
    use_tfds_tfrecord_train = (
        self.grain_file_type == "tfrecord" and self.dataset_path and self.dataset_name and self.train_split
    )
    use_tfds_tfrecord_eval = (
        self.grain_file_type == "tfrecord" and self.dataset_path and self.eval_dataset_name and self.eval_split
    )
    if not self.grain_train_files and not self.grain_train_mixture_config_path and not (
        use_hf_parquet or use_tfds_tfrecord_train
    ):
      raise ValueError(
          "When dataset_type=grain, set grain_train_files, "
          "grain_train_mixture_config_path, use hf_path with grain_file_type=parquet, or use dataset_path, "
          "dataset_name, and train_split with grain_file_type=tfrecord."
      )
    if self.eval_interval > 0 and not self.grain_eval_files and not (use_hf_parquet or use_tfds_tfrecord_eval):
      raise ValueError(
          "Please specify grain_eval_files, use hf_path with grain_file_type=parquet, or use dataset_path, "
          "eval_dataset_name, and eval_split with grain_file_type=tfrecord; otherwise set eval_interval to <=0."
      )
  elif self.dataset_type == DatasetType.TFDS:
    if self.use_dpo:
      raise ValueError(
          "TFDS dataset_type=tfds is not supported for DPO training"
          " (config.use_dpo=True). Please use dataset_type=grain or dataset_type=hf instead."
      )
    if not self.dataset_name:
      raise ValueError("dataset_name can't be empty when dataset_type=tfds")
    if self.eval_interval > 0 and not self.eval_split:
      raise ValueError("Please specify eval_split or set eval_interval to <=0.")
  if self.eval_interval > 0 >= self.eval_steps and self.generate_padding_batch_eval:
    raise ValueError("`eval_steps` must be > 0 when `generate_padding_batch_eval` is True.")
