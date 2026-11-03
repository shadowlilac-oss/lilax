"""Arguments for supervised, preference, and distillation training."""

from .training_args import TrainingArguments

class _PostTrainingArguments(TrainingArguments):
  def __init__(self, *args, save_only_lora_params=None, restore_only_lora_params=None, **kwargs):
    super().__init__(*args, **kwargs)
    for name, value in (("save_only_lora_params", save_only_lora_params),
                        ("restore_only_lora_params", restore_only_lora_params)):
      if value is not None and type(value) is not bool:
        raise ValueError(f"{name} must be None or a boolean.")
    self.save_only_lora_params = save_only_lora_params
    self.restore_only_lora_params = restore_only_lora_params


class SFTConfig(_PostTrainingArguments):
  def __init__(
      self, *args, max_length=None, packing=None, completion_only_loss=None, train_data_columns=None,
      eval_data_columns=None, tokenize_train_data=None, tokenize_eval_data=None, num_epoch=None,
      enable_data_shuffling=None, data_shuffle_seed=None, chat_template=None, chat_template_path=None,
      formatting_func_path=None, formatting_func_kwargs=None, max_segments_per_seq=None, grain_worker_count=None,
      hf_path=None, hf_name=None, hf_data_dir=None, hf_train_files=None, hf_eval_split=None,
      hf_eval_files=None, hf_access_token=None, dataset_path=None, dataset_name=None, eval_dataset_name=None,
      train_split=None, eval_split=None, grain_train_mixture_config_path=None, grain_use_elastic_iterator=None,
      grain_packing_type=None, grain_per_worker_buffer_size=None, grain_worker_count_eval=None,
      grain_per_worker_buffer_size_eval=None, grain_num_threads=None, grain_num_threads_eval=None,
      grain_prefetch_buffer_size=None, grain_prefetch_buffer_size_eval=None, grain_data_source_max_workers=None,
      grain_shuffle_buffer_size=None, olmo_index_path=None, olmo_path_remap_from=None, olmo_path_remap_to=None,
      olmo_apply_ngram_filter=None, train_image_column=None, eval_image_column=None, default_prompt=None,
      image_placeholder=None, video_placeholder=None, max_num_images_per_example=None,
      use_tunix_gradient_accumulation=None, quantization=None, use_qwix_quantization=None,
      use_manual_quantization=None, jax_mx_gradient_mode=None, quant_cfg_path=None,
      quantization_local_shard_count=None, qwix_bwd_weight_grad_tile_count=None,
      weight_quantization_calibration_method=None, act_quantization_calibration_method=None,
      bwd_quantization_calibration_method=None, drhs_grad_quantization_calibration_method=None,
      quantize_mtp=None, quantize_router_proj=None, quantize_logits_proj=None,
      logits_proj_quant_calibration_method=None, logits_proj_bwd_quant_calibration_method=None,
      sa_qproj_bwd_bf16=None, replicate_quant_scale=None, **kwargs,
  ):
    super().__init__(
        *args, quantization=quantization, use_qwix_quantization=use_qwix_quantization,
        use_manual_quantization=use_manual_quantization, jax_mx_gradient_mode=jax_mx_gradient_mode,
        quant_cfg_path=quant_cfg_path, quantization_local_shard_count=quantization_local_shard_count,
        qwix_bwd_weight_grad_tile_count=qwix_bwd_weight_grad_tile_count,
        weight_quantization_calibration_method=weight_quantization_calibration_method,
        act_quantization_calibration_method=act_quantization_calibration_method,
        bwd_quantization_calibration_method=bwd_quantization_calibration_method,
        drhs_grad_quantization_calibration_method=drhs_grad_quantization_calibration_method,
        quantize_mtp=quantize_mtp, quantize_router_proj=quantize_router_proj,
        quantize_logits_proj=quantize_logits_proj, logits_proj_quant_calibration_method=logits_proj_quant_calibration_method,
        logits_proj_bwd_quant_calibration_method=logits_proj_bwd_quant_calibration_method,
        sa_qproj_bwd_bf16=sa_qproj_bwd_bf16, replicate_quant_scale=replicate_quant_scale, **kwargs,
    )
    self.max_length = max_length
    self.packing = packing
    self.completion_only_loss = completion_only_loss
    self.train_data_columns = train_data_columns
    self.eval_data_columns = eval_data_columns
    self.tokenize_train_data = tokenize_train_data
    self.tokenize_eval_data = tokenize_eval_data
    self.num_epoch = num_epoch
    self.enable_data_shuffling = enable_data_shuffling
    self.data_shuffle_seed = data_shuffle_seed
    self.chat_template = chat_template
    self.chat_template_path = chat_template_path
    self.formatting_func_path = formatting_func_path
    self.formatting_func_kwargs = formatting_func_kwargs
    self.max_segments_per_seq = max_segments_per_seq
    self.grain_worker_count = grain_worker_count
    self.hf_path = hf_path
    self.hf_name = hf_name
    self.hf_data_dir = hf_data_dir
    self.hf_train_files = hf_train_files
    self.hf_eval_split = hf_eval_split
    self.hf_eval_files = hf_eval_files
    self.hf_access_token = hf_access_token
    self.dataset_path = dataset_path
    self.dataset_name = dataset_name
    self.eval_dataset_name = eval_dataset_name
    self.train_split = train_split
    self.eval_split = eval_split
    self.grain_train_mixture_config_path = grain_train_mixture_config_path
    self.grain_use_elastic_iterator = grain_use_elastic_iterator
    self.grain_packing_type = grain_packing_type
    self.grain_per_worker_buffer_size = grain_per_worker_buffer_size
    self.grain_worker_count_eval = grain_worker_count_eval
    self.grain_per_worker_buffer_size_eval = grain_per_worker_buffer_size_eval
    self.grain_num_threads = grain_num_threads
    self.grain_num_threads_eval = grain_num_threads_eval
    self.grain_prefetch_buffer_size = grain_prefetch_buffer_size
    self.grain_prefetch_buffer_size_eval = grain_prefetch_buffer_size_eval
    self.grain_data_source_max_workers = grain_data_source_max_workers
    self.grain_shuffle_buffer_size = grain_shuffle_buffer_size
    self.olmo_index_path = olmo_index_path
    self.olmo_path_remap_from = olmo_path_remap_from
    self.olmo_path_remap_to = olmo_path_remap_to
    self.olmo_apply_ngram_filter = olmo_apply_ngram_filter
    self.train_image_column = train_image_column
    self.eval_image_column = eval_image_column
    self.default_prompt = default_prompt
    self.image_placeholder = image_placeholder
    self.video_placeholder = video_placeholder
    self.max_num_images_per_example = max_num_images_per_example
    self.use_tunix_gradient_accumulation = use_tunix_gradient_accumulation

  def _overrides(self):
    return super()._overrides() | {
        key: value for key, value in (
            ("max_target_length", self.max_length), ("packing", self.packing),
            ("sft_train_on_completion_only", self.completion_only_loss),
            ("train_data_columns", self.train_data_columns), ("eval_data_columns", self.eval_data_columns),
            ("tokenize_train_data", self.tokenize_train_data), ("tokenize_eval_data", self.tokenize_eval_data),
            ("num_epoch", self.num_epoch), ("enable_data_shuffling", self.enable_data_shuffling),
            ("data_shuffle_seed", self.data_shuffle_seed), ("chat_template", self.chat_template),
            ("chat_template_path", self.chat_template_path), ("formatting_func_path", self.formatting_func_path),
            ("formatting_func_kwargs", self.formatting_func_kwargs), ("max_segments_per_seq", self.max_segments_per_seq),
            ("grain_worker_count", self.grain_worker_count),
            ("hf_path", self.hf_path), ("hf_name", self.hf_name),
            ("hf_data_dir", self.hf_data_dir), ("hf_train_files", self.hf_train_files),
            ("hf_eval_split", self.hf_eval_split), ("hf_eval_files", self.hf_eval_files),
            ("hf_access_token", self.hf_access_token), ("dataset_path", self.dataset_path),
            ("dataset_name", self.dataset_name), ("eval_dataset_name", self.eval_dataset_name),
            ("train_split", self.train_split), ("eval_split", self.eval_split),
            ("grain_train_mixture_config_path", self.grain_train_mixture_config_path),
            ("grain_use_elastic_iterator", self.grain_use_elastic_iterator),
            ("grain_packing_type", self.grain_packing_type),
            ("grain_per_worker_buffer_size", self.grain_per_worker_buffer_size),
            ("grain_worker_count_eval", self.grain_worker_count_eval),
            ("grain_per_worker_buffer_size_eval", self.grain_per_worker_buffer_size_eval),
            ("grain_num_threads", self.grain_num_threads), ("grain_num_threads_eval", self.grain_num_threads_eval),
            ("grain_prefetch_buffer_size", self.grain_prefetch_buffer_size),
            ("grain_prefetch_buffer_size_eval", self.grain_prefetch_buffer_size_eval),
            ("grain_data_source_max_workers", self.grain_data_source_max_workers),
            ("grain_shuffle_buffer_size", self.grain_shuffle_buffer_size),
            ("olmo_index_path", self.olmo_index_path), ("olmo_path_remap_from", self.olmo_path_remap_from),
            ("olmo_path_remap_to", self.olmo_path_remap_to), ("olmo_apply_ngram_filter", self.olmo_apply_ngram_filter),
            ("train_image_column", self.train_image_column), ("eval_image_column", self.eval_image_column),
            ("default_prompt", self.default_prompt), ("image_placeholder", self.image_placeholder),
            ("video_placeholder", self.video_placeholder),
            ("max_num_images_per_example", self.max_num_images_per_example),
            ("use_tunix_gradient_accumulation", self.use_tunix_gradient_accumulation),
        ) if value is not None
    }


class DPOConfig(_PostTrainingArguments):
  pass


class DistillationConfig(_PostTrainingArguments):
  def __init__(self, *args, is_offline=False, offline_data_dir=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.is_offline = is_offline
    self.offline_data_dir = offline_data_dir


_PostTrainingArguments.__module__ = "lilax.trainers.post_training"
SFTConfig.__module__ = "lilax.trainers.post_training"
DPOConfig.__module__ = "lilax.trainers.post_training"
DistillationConfig.__module__ = "lilax.trainers.post_training"
