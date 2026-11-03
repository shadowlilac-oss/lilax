"""Training arguments mapped to native MaxText configuration."""

from lilax._backend.configs.runtime import resolve_initialized_config

from ._input_args import MMAP_FIELDS, TFDS_C4_FIELDS, input_field_overrides, input_overrides
from ._optimizer_args import optimizer_overrides
from ._precision_args import PRECISION_FIELDS, precision_overrides
from ._reporting_args import reporting_overrides


class TrainingArguments:
  """Override selected training settings while retaining native defaults."""

  def __init__(
      self, output_dir=None, max_steps=None, learning_rate=None, gradient_accumulation_steps=None,
      eval_steps=None, save_steps=None, per_device_train_batch_size=None, per_device_eval_batch_size=None,
      lr_schedule_type=None, warmup_steps_fraction=None, learning_rate_schedule_steps=None,
      gradient_clipping_threshold=None, adam_b1=None, adam_b2=None, adam_eps=None, adam_weight_decay=None,
      quantization=None, use_qwix_quantization=None, use_manual_quantization=None, jax_mx_gradient_mode=None,
      quant_cfg_path=None, quantization_local_shard_count=None, qwix_bwd_weight_grad_tile_count=None,
      weight_quantization_calibration_method=None, act_quantization_calibration_method=None,
      bwd_quantization_calibration_method=None, drhs_grad_quantization_calibration_method=None,
      quantize_mtp=None, quantize_router_proj=None, quantize_logits_proj=None,
      logits_proj_quant_calibration_method=None, logits_proj_bwd_quant_calibration_method=None,
      sa_qproj_bwd_bf16=None, replicate_quant_scale=None, report_to=None, launch_xprof=False, xprof_port=8791,
      eval_num_batches=None, learning_rate_final_fraction=None, wsd_decay_steps_fraction=None, wsd_decay_style=None,
      opt_type=None, adam_eps_root=None, adamw_mask=None, mu_dtype=None, muon_type=None, muon_beta=None,
      muon_weight_decay=None, muon_consistent_rms=None, muon_include_routers=None, muon_use_all_to_all=None,
      skip_step_on_spikes=None, skip_step_interval=None, skip_step_scaling_factor=None, trainable_parameters_mask=None,
      dataset_type=None, grain_file_type=None, grain_train_files=None, grain_eval_files=None,
      grain_ram_budget_mb=None, grain_index_storage_option=None, mmap_eod_id=None, blend_cache_dir=None,
      blend_index_dir=None, reset_attention_mask=None, eod_mask_loss=None, packing_max_segments_per_sample=None,
      mmap_split_sentences=None, mmap_npy_split=None,
      colocated_python_data_input=None, expansion_factor_real_data=None,
      generate_padding_batch_train=None, generate_padding_batch_eval=None,
      use_stream_chunking=None, train_shard_in_read=None, train_interleave_cycle_length=None,
  ):
    if isinstance(report_to, str):
      report_to = [] if report_to == "none" else [report_to]
    if report_to is not None and any(name not in ("trackio", "wandb", "tensorboard") for name in report_to):
      raise ValueError("report_to must contain only 'trackio', 'wandb', or 'tensorboard'.")
    self.report_to = report_to
    self.launch_xprof = launch_xprof
    self.xprof_port = xprof_port
    self.output_dir = output_dir
    self.max_steps = max_steps
    self.learning_rate = learning_rate
    self.gradient_accumulation_steps = gradient_accumulation_steps
    self.eval_steps = eval_steps
    self.eval_num_batches = eval_num_batches
    self.save_steps = save_steps
    self.per_device_train_batch_size = per_device_train_batch_size
    self.per_device_eval_batch_size = per_device_eval_batch_size
    self.lr_schedule_type = lr_schedule_type
    self.warmup_steps_fraction = warmup_steps_fraction
    self.learning_rate_schedule_steps = learning_rate_schedule_steps
    self.learning_rate_final_fraction = learning_rate_final_fraction
    self.wsd_decay_steps_fraction = wsd_decay_steps_fraction
    self.wsd_decay_style = wsd_decay_style
    self.gradient_clipping_threshold = gradient_clipping_threshold
    self.opt_type = opt_type
    self.adam_b1 = adam_b1
    self.adam_b2 = adam_b2
    self.adam_eps = adam_eps
    self.adam_eps_root = adam_eps_root
    self.adam_weight_decay = adam_weight_decay
    self.adamw_mask = adamw_mask
    self.mu_dtype = mu_dtype
    self.muon_type = muon_type
    self.muon_beta = muon_beta
    self.muon_weight_decay = muon_weight_decay
    self.muon_consistent_rms = muon_consistent_rms
    self.muon_include_routers = muon_include_routers
    self.muon_use_all_to_all = muon_use_all_to_all
    self.skip_step_on_spikes = skip_step_on_spikes
    self.skip_step_interval = skip_step_interval
    self.skip_step_scaling_factor = skip_step_scaling_factor
    self.trainable_parameters_mask = trainable_parameters_mask
    self.quantization = quantization
    self.use_qwix_quantization = use_qwix_quantization
    self.use_manual_quantization = use_manual_quantization
    self.jax_mx_gradient_mode = jax_mx_gradient_mode
    self.quant_cfg_path = quant_cfg_path
    self.quantization_local_shard_count = quantization_local_shard_count
    self.qwix_bwd_weight_grad_tile_count = qwix_bwd_weight_grad_tile_count
    self.weight_quantization_calibration_method = weight_quantization_calibration_method
    self.act_quantization_calibration_method = act_quantization_calibration_method
    self.bwd_quantization_calibration_method = bwd_quantization_calibration_method
    self.drhs_grad_quantization_calibration_method = drhs_grad_quantization_calibration_method
    self.quantize_mtp = quantize_mtp
    self.quantize_router_proj = quantize_router_proj
    self.quantize_logits_proj = quantize_logits_proj
    self.logits_proj_quant_calibration_method = logits_proj_quant_calibration_method
    self.logits_proj_bwd_quant_calibration_method = logits_proj_bwd_quant_calibration_method
    self.sa_qproj_bwd_bf16 = sa_qproj_bwd_bf16
    self.replicate_quant_scale = replicate_quant_scale
    self.dataset_type = dataset_type
    self.grain_file_type = grain_file_type
    self.grain_train_files = grain_train_files
    self.grain_eval_files = grain_eval_files
    self.grain_ram_budget_mb = grain_ram_budget_mb
    self.grain_index_storage_option = grain_index_storage_option
    self.mmap_eod_id = mmap_eod_id
    self.blend_cache_dir = blend_cache_dir
    self.blend_index_dir = blend_index_dir
    self.reset_attention_mask = reset_attention_mask
    self.eod_mask_loss = eod_mask_loss
    self.packing_max_segments_per_sample = packing_max_segments_per_sample
    self.mmap_split_sentences = mmap_split_sentences
    self.mmap_npy_split = mmap_npy_split
    self.colocated_python_data_input = colocated_python_data_input
    self.expansion_factor_real_data = expansion_factor_real_data
    self.generate_padding_batch_train = generate_padding_batch_train
    self.generate_padding_batch_eval = generate_padding_batch_eval
    self.use_stream_chunking = use_stream_chunking
    self.train_shard_in_read = train_shard_in_read
    self.train_interleave_cycle_length = train_interleave_cycle_length

  def _overrides(self):
    return {
        key: value for key, value in (
            ("base_output_directory", self.output_dir), ("steps", self.max_steps),
            ("learning_rate", self.learning_rate), ("gradient_accumulation_steps", self.gradient_accumulation_steps),
            ("eval_interval", self.eval_steps), ("eval_steps", self.eval_num_batches),
            ("checkpoint_period", self.save_steps),
            ("per_device_batch_size", self.per_device_train_batch_size),
            ("eval_per_device_batch_size", self.per_device_eval_batch_size),
            *optimizer_overrides(self),
            *input_overrides(self),
        ) if value is not None
    } | precision_overrides(self) | input_field_overrides(self) | reporting_overrides(self)

  def resolve(self, runtime, *, train_dataset_provided=False, eval_dataset_provided=False):
    overrides = runtime._overrides | self._overrides()
    if not overrides:
      return runtime.config
    return resolve_initialized_config(
        runtime.public_config, runtime.config, overrides=overrides, train_dataset_provided=train_dataset_provided,
        eval_dataset_provided=eval_dataset_provided,
    )
