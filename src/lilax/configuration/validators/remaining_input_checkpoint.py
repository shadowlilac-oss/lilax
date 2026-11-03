from lilax.configuration.domain_types import DatasetType, DecoderBlockType, LearningRateScheduleType


def validate_input_and_checkpoint_options(self) -> "MaxTextConfig":
  if self.opt_type == "muon" and self.decoder_block not in (
      DecoderBlockType.DEEPSEEK, DecoderBlockType.DEEPSEEK4, DecoderBlockType.QWEN3, DecoderBlockType.QWEN3_MOE,
      DecoderBlockType.QWEN3_CUSTOM_MOE, DecoderBlockType.QWEN3_NEXT, DecoderBlockType.GPT_OSS,
      DecoderBlockType.GEMMA3, DecoderBlockType.LLAMA2,
  ):
    raise ValueError(
        "Muon dimension numbers haven't been tested for this model. Run this command first: "
        f"`python3 -m lilax._backend.optimizers.muon_config.dimensions {self.model_name} True`"
    )
  if self.deepstack_visual_indexes_for_vit and self.scan_layers:
    raise ValueError(
        "Deepstack visual embedding injection requires scan_layers=False. "
        "Set scan_layers=False in your config to use deepstack features."
    )
  if self.lr_schedule_type == LearningRateScheduleType.WSD:
    total_fraction = self.warmup_steps_fraction + self.wsd_decay_steps_fraction
    if total_fraction > 1.0:
      raise ValueError(
          f"Invalid WSD schedule: warmup_steps_fraction ({self.warmup_steps_fraction}) + "
          f"wsd_decay_steps_fraction ({self.wsd_decay_steps_fraction}) must not exceed 1.0. "
          f"Current sum: {total_fraction}"
      )
  if self.distill_beta > 0.0 and not self.scan_layers:
    raise ValueError("a value of self.distill_beta > 0.0 requires self.scan_layers = True")
  if self.distill_alpha_end is not None and not 0.0 <= self.distill_alpha_end <= 1.0:
    raise ValueError(f"distill_alpha_end must be in [0, 1], got {self.distill_alpha_end}")
  if self.distill_temperature_end is not None and self.distill_temperature_end <= 0.0:
    raise ValueError(f"distill_temperature_end must be > 0, got {self.distill_temperature_end}")
  if self.distill_beta_end is not None and self.distill_beta_end < 0.0:
    raise ValueError(f"distill_beta_end must be >= 0, got {self.distill_beta_end}")
  if self.distill_beta == 0.0 and self.distill_beta_end is not None and self.distill_beta_end > 0.0:
    raise ValueError(
        f"distill_beta=0.0 but distill_beta_end={self.distill_beta_end}. "
        "Feature extraction is disabled when distill_beta starts at 0.0. "
        "Set distill_beta to a small positive value (e.g., 1e-6) to enable feature extraction."
    )
  for param_name, schedule, end_value in (
      ("distill_alpha", self.distill_alpha_schedule, self.distill_alpha_end),
      ("distill_temperature", self.distill_temperature_schedule, self.distill_temperature_end),
      ("distill_beta", self.distill_beta_schedule, self.distill_beta_end),
  ):
    if schedule != "constant" and end_value is None:
      raise ValueError(
          f"{param_name}_schedule is '{schedule}' but {param_name}_end is None. "
          f"Set {param_name}_end to a target value or use schedule='constant'."
      )
  if self.load_parameters_path and self.load_full_state_path:
    raise ValueError("At most one of `load_parameters_path` or `load_full_state_path` should be set.")
  if self.source_checkpoint_layout == "safetensors_dynamic" and self.enable_single_controller:
    raise ValueError(
        "`source_checkpoint_layout='safetensors_dynamic'` is not supported"
        " on the Pathways backend (`enable_single_controller=True`)."
    )
  if self.elastic_enabled and not self.enable_single_controller:
    raise ValueError("Elastic training is only supported with Pathways (`enable_single_controller=True`).")
  if self.elastic_backup_kind not in ("snapshot", "checkpoint"):
    raise ValueError(
        "elastic_backup_kind must be one of 'snapshot' or 'checkpoint', got" f" '{self.elastic_backup_kind}'."
    )
  if self.elastic_enabled and self.elastic_backup_kind == "snapshot":
    if self.enable_diloco:
      raise ValueError("elastic_backup_kind='snapshot' is not supported with DiLoCo (`enable_diloco=True`).")
    if self.enable_rampup_batch_size:
      raise ValueError("elastic_backup_kind='snapshot' is not supported with `enable_rampup_batch_size=True`.")
    if self.dataset_type != DatasetType.SYNTHETIC and (
        self.dataset_type != DatasetType.GRAIN or not self.grain_use_elastic_iterator
    ):
      raise ValueError(
          "elastic_backup_kind='snapshot' requires `dataset_type=grain` with `grain_use_elastic_iterator=True`"
          " (or synthetic data), so that the data iterator position can be snapshotted with the train state."
      )
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
  if self.use_dpo and self.packing:
    raise ValueError("DPO does not support packing. Set `packing=False`.")
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
  if self.load_full_state_path and not self.enable_checkpointing:
    raise ValueError("You must set enable_checkpointing=True to resume from load_full_state_path.")
  if self.enable_multi_tier_checkpointing:
    if not self.local_checkpoint_directory:
      raise ValueError("`local_checkpoint_directory` must be set for multi-tier checkpointing.")
    if self.local_checkpoint_period <= 0:
      raise ValueError("`local_checkpoint_period` must be > 0 for multi-tier checkpointing.")
    if (self.multi_tier_checkpointing_backup_interval_minutes is None) == (
        self.multi_tier_checkpointing_backup_interval_steps is None
    ):
      raise ValueError(
          "Exactly one of `multi_tier_checkpointing_backup_interval_minutes`"
          " or `multi_tier_checkpointing_backup_interval_steps` must be specified."
      )
    if (
        self.multi_tier_checkpointing_backup_interval_steps is not None
        and self.multi_tier_checkpointing_backup_interval_steps < self.local_checkpoint_period
    ):
      raise ValueError(
          "`multi_tier_checkpointing_backup_interval_steps`"
          f" ({self.multi_tier_checkpointing_backup_interval_steps}) must be"
          f" >= `local_checkpoint_period` ({self.local_checkpoint_period})."
      )
  if self.colocated_python_checkpointing and not self.enable_single_controller:
    raise ValueError("`colocated_python_checkpointing` is only supported with `enable_single_controller` set to True.")
  if self.enable_emergency_checkpoint:
    if not self.local_checkpoint_directory:
      raise ValueError("`local_checkpoint_directory` must be set for emergency checkpointing.")
    if self.local_checkpoint_period <= 0:
      raise ValueError("`local_checkpoint_period` must be > 0 for emergency checkpointing.")
  if self.use_sft and self.use_dpo:
    raise ValueError("Only one of `use_sft` or `use_dpo` can be True.")
  if self.use_multimodal:
    if self.model_name in ("gemma4-e2b", "gemma4-e4b"):
      raise ValueError(f"Multimodal is not yet supported for {self.model_name}; only text inputs are supported.")
    valid_mm_models = (
        "gemma3-4b", "gemma3-12b", "gemma3-27b", "gemma4-26b", "gemma4-31b", "llama4-17b-16e",
        "llama4-17b-128e", "qwen3-omni-30b-a3b", "qwen3-vl-2b", "qwen3-vl-4b", "qwen3-vl-30b-a3b",
        "qwen3.5-9b", "qwen3.5-35b-a3b", "qwen3.5-397b-a17b", "maxtext-omni-gemma3-qwen3",
        "weaver-mini", "weaver-max", "weaver-mini-diffuser", "ming-3.0-audio-tiny",
    )
    if self.model_name not in valid_mm_models and self.model_name != "default":
      raise ValueError(f"Multimodal is only supported for {valid_mm_models}, not {self.model_name}")
    if self.use_sft:
      if not self.sft_train_on_completion_only:
        raise ValueError("For multimodal SFT, `sft_train_on_completion_only` must be True.")
      if self.packing:
        raise ValueError("For multimodal SFT, `packing` is not yet supported.")
  if self.use_dpo:
    if self.dpo.max_prompt_length is None:
      self.dpo.max_prompt_length = self.max_target_length // 2
    if self.dpo.max_prompt_length >= self.max_target_length:
      raise ValueError(
          f"dpo.max_prompt_length ({self.dpo.max_prompt_length}) must be less than max_target_length"
          f" ({self.max_target_length})."
      )
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
  for prefix in ("ici", "dcn"):
    axis_map = {
        "diloco": getattr(self, f"{prefix}_diloco_parallelism"),
        "data": getattr(self, f"{prefix}_data_parallelism"),
        "stage": getattr(self, f"{prefix}_pipeline_parallelism"),
        "fsdp": getattr(self, f"{prefix}_fsdp_parallelism"),
        "fsdp_transpose": getattr(self, f"{prefix}_fsdp_transpose_parallelism"),
        "sequence": getattr(self, f"{prefix}_sequence_parallelism"),
        "context": getattr(self, f"{prefix}_context_parallelism"),
        "context_usp_ulysses": getattr(self, f"{prefix}_context_usp_ulysses_parallelism"),
        "context_autoregressive": getattr(self, f"{prefix}_context_autoregressive_parallelism"),
        "tensor": getattr(self, f"{prefix}_tensor_parallelism"),
        "tensor_sequence": getattr(self, f"{prefix}_tensor_sequence_parallelism"),
        "model": getattr(self, f"{prefix}_tensor_parallelism"),
        "expert": getattr(self, f"{prefix}_expert_parallelism"),
        "autoregressive": getattr(self, f"{prefix}_autoregressive_parallelism"),
        "attn_dp": 1, "attn_dp_expert": 1, "dcp": 1, "pcp": 1,
    }
    configured = getattr(self, f"{prefix}_parallelism")
    if configured and len(configured) == len(self.mesh_axes) and any(axis not in axis_map for axis in self.mesh_axes):
      parallelism = list(configured)
    else:
      parallelism = [axis_map.get(axis, 1) for axis in self.mesh_axes]
    setattr(self, f"{prefix}_parallelism", parallelism)
  return self
