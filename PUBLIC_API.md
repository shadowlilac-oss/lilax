# Public Python API

This reference lists importable declarations from the implemented working tree.
Signatures come from Python inspection. `<factory>` means a fresh per-instance value.
Upstream processing kwargs retain Transformers ownership. Backend computation and platform support
remain subject to the existing model and engine limits.

See [execution profiles](EXECUTION_PROFILE.md) and [API migration](API_MIGRATION.md).

Evaluation consumes finite supplied data without a batch cap. SyntheticDataset is training-only.
Model artifacts and resumable training checkpoints are separate operations.
The ordinary Orbax save callback validates the exact action metadata and committed native step.
Emergency and replication completion tiers remain unsupported for on_save notification.

## lilax

### AutoConfig

```python
AutoConfig()
```

```python
AutoConfig.from_pretrained(pretrained_model_name_or_path: str | pathlib.Path, *, revision: str | None = None, token: str | bool | None = None, cache_dir: str | pathlib.Path | None = None, local_files_only: bool = False, model_name: str | None = None) -> lilax.configuration.schema.MaxTextConfig
```

### MaxTextConfig

```python
MaxTextConfig(*, model_name: str, vocab_size: int, hidden_size: int, intermediate_size: int, num_hidden_layers: int, num_attention_heads: int, num_key_value_heads: int, head_dim: int, max_position_embeddings: int = 2048, enable_dropout: bool = True, dropout_rate: float = 0.0, moe_intermediate_size: int | None = None, architecture_config: lilax.configuration.model_architecture.ArchitectureConfig | None = None, attention_config: lilax.configuration.model_attention.AttentionConfig | None = None, position_config: lilax.configuration.model_attention.PositionConfig | None = None, moe_config: lilax.configuration.model_families.MoEConfig | None = None, qwen_config: lilax.configuration.domains.model_qwen.Qwen3Next | lilax.configuration.domains.model_qwen.Qwen3_8FlashNext | None = None, engram_config: lilax.configuration.domains.model_engram.Engram | None = None, ming_config: lilax.configuration.domains.model_ming.Ming | None = None, hyperconnection_config: lilax.configuration.domains.data_mhc.ManifoldConstrainedHyperConnections | None = None, vision_config: lilax.configuration.model_families.VisionConfig | None = None, vision_projector_config: lilax.configuration.domains.execution_multimodal.VisionProjector | None = None, audio_config: lilax.configuration.model_families.AudioConfig | None = None, mtp_config: lilax.configuration.domains.execution_loss.MTP | None = None, logits_config: lilax.configuration.domains.execution_loss.LogitsAndLoss | None = None, precision_config: lilax.configuration.model_precision.PrecisionConfig | None = None, quantization_config: lilax.configuration.quantization.ComputeRecipeConfig | lilax.configuration.quantization.AqtInt8Config | lilax.configuration.quantization.AqtMixedPrecisionConfig | lilax.configuration.quantization.JaxMXConfig | lilax.configuration.quantization.ServeFP8WeightConfig | lilax.configuration.quantization.QwixRecipeConfig | lilax.configuration.quantization.QwixFP8FullConfig | lilax.configuration.quantization.BatchSplitQwixConfig | lilax.configuration.quantization.BatchSplitManualConfig | lilax.configuration.domains.data_qat.QwixQATConfig | None = None, kv_cache_quantization_config: lilax.configuration.quantization.KVCacheQuantizationConfig | None = None, sparsity_config: lilax.configuration.quantization.SparsityConfig | None = None, kernel_config: lilax.configuration.model_attention.KernelConfig | None = None, layout_config: lilax.configuration.model_architecture.ModelLayoutConfig | None = None, **kwargs)
```

```python
MaxTextConfig.to_diff_dict(self)
```

```python
MaxTextConfig.to_dict(self)
```

### ModelCapabilities

```python
ModelCapabilities(model_name: str, native: bool, hf_import: bool, hf_export: bool, hf_repo_id: str | None) -> None
```

### get_model_capabilities

```python
get_model_capabilities(model_name: str) -> lilax.modeling.hub.ModelCapabilities
```

### HF_EXPORT_MODELS

Exported type or value from `None`.

### HF_IMPORT_MODELS

Exported type or value from `None`.

### HF_REPOS

Exported type or value from `None`.

### MODEL_CONFIG_NAMES

Exported type or value from `None`.

### NATIVE_MODELS

Exported type or value from `None`.

### __version__

Exported type or value from `None`.

### AutoProcessor

```python
AutoProcessor()
```

```python
AutoProcessor.from_pretrained(pretrained_model_name_or_path, *, revision=None, token=None, cache_dir=None, local_files_only=False, tokenize: bool | None = None, use_chat_template: bool | None = None, use_truncation: bool | None = None, image_column: str | list[str] | None = None, image_placeholder: str | None = None, video_placeholder: str | None = None, max_num_images_per_example: int | None = None, audio_placeholder: str | None = None, use_audio_in_video: bool | None = None, video_max_grid_t: int | None = None, video_max_grid_h: int | None = None, video_max_grid_w: int | None = None, **kwargs)
```

### AutoTokenizer

```python
AutoTokenizer()
```

```python
AutoTokenizer.from_pretrained(pretrained_model_name_or_path, *, revision=None, token=None, cache_dir=None, local_files_only=False, tokenize: bool | None = None, use_chat_template: bool | None = None, use_truncation: bool | None = None, **kwargs)
```

### AutoModelForCausalLM

```python
AutoModelForCausalLM()
```

```python
AutoModelForCausalLM.from_config(config, *, quantization_config=None, seed=0)
```

```python
AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, *, config=None, revision=None, token=None, cache_dir=None, local_files_only=False, quantization_config=None, fuse_qkv=None, fuse_mlp=None, precision_config=None)
```

### MaxTextModel

```python
MaxTextModel(native_model, runtime, *, construction_config=None, checkpoint=None, model_options=None, quantization_config=None, config=None)
```

```python
MaxTextModel.prepare_for_full_state_restore(self, qat_config, *, runtime)
```

```python
MaxTextModel.materialize(self, *, runtime)
```

```python
MaxTextModel.__call__(self, input_ids, *, position_ids=None, attention_mask=None, segment_ids=None, inputs_embeds=None, images=None, image_masks=None, videos=None, video_masks=None, video_grid_thw=None, audio_values=None, decoder_target_tokens=None, decoder_target_mask=None, enable_dropout=True)
```

```python
MaxTextModel.generate(self, input_ids, generation_config=None, *, attention_mask=None, max_new_tokens=None, do_sample=None, top_k=None, top_p=None, temperature=None, rng=None, position_ids=None, mrope_deltas=None, images=None, image_masks=None, videos=None, video_masks=None, video_grid_thw=None, audio_values=None, audio_masks=None)
```

```python
MaxTextModel.generate_batch(self, inputs, generation_config=None, *, continuous_batching_config=None, rng=None)
```

```python
MaxTextModel.save_pretrained(self, save_directory, *, format='maxtext', export_config=None)
```

### GenerationConfig

```python
GenerationConfig(*, max_new_tokens: int | None = None, do_sample: bool = False, top_k: int = 0, top_p: float = -1.0, temperature: float = 1.0, eos_token_id: int | list[int] | None = None) -> None
```

```python
GenerationConfig.sampling_options(self)
```

### Trainer

```python
Trainer(model=None, *, args, data_collator=None, train_dataset=None, eval_dataset=None, processing_class=None, callbacks=None, optimizers=(None, None), model_init=None)
```

```python
Trainer.train(self, resume_from_checkpoint=None, *, checkpoint_step=None)
```

```python
Trainer.evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval')
```

```python
Trainer.save_state(self)
```

```python
Trainer.save_model(self, output_dir=None)
```

### TrainingArguments

```python
TrainingArguments(*, output_dir: str, run_name: str = '', max_steps: int = 150001, num_train_epochs: int = 1, per_device_train_batch_size: int | float = 12, per_device_eval_batch_size: int | float = 0.0, gradient_accumulation_steps: int = 1, learning_rate: float = 3e-05, optim: Literal['adamw', 'adam_pax', 'sgd', 'muon'] = 'adamw', adam_beta1: float = 0.9, adam_beta2: float = 0.95, adam_epsilon: float = 1e-08, weight_decay: float = 0.1, max_grad_norm: float = 1.0, lr_scheduler_type: Literal['cosine', 'wsd'] = 'cosine', warmup_ratio: float = 0.1, lr_scheduler_config: lilax.training.optimizers.LRSchedulerConfig | None = None, optimizer_config: lilax.training.optimizers.OptimizerConfig | None = None, logging_steps: int = 100, eval_steps: int = -1, save_steps: int = 10000, eval_on_start: bool = False, checkpoint_config: lilax.checkpoints.policy.TrainingCheckpointConfig | None = None, report_to: str | list[str] = 'none', logging_dir: str | None = None, parallelism_config: lilax.distributed.ParallelismConfig | None = None, jax_distributed_initialization_timeout: int = 300, diloco_config: lilax.distributed.DiLoCoConfig | None = None, gradient_checkpointing: bool = True, remat_config: lilax.training.remat.RematConfig | None = None, objective_config: lilax.training.objectives.ObjectiveConfig | None = None, batch_rampup_config: lilax.training.objectives.BatchRampupConfig | None = None) -> None
```

```python
TrainingArguments.validate_reporting(cls, value)
```

### SFTConfig

```python
SFTConfig(*, output_dir: str, run_name: str = '', max_steps: int = 150001, num_train_epochs: int = 1, per_device_train_batch_size: int | float = 12, per_device_eval_batch_size: int | float = 0.0, gradient_accumulation_steps: int = 1, learning_rate: float = 2e-05, optim: Literal['adamw', 'adam_pax', 'sgd', 'muon'] = 'adamw', adam_beta1: float = 0.9, adam_beta2: float = 0.95, adam_epsilon: float = 1e-08, weight_decay: float = 0.1, max_grad_norm: float = 1.0, lr_scheduler_type: Literal['cosine', 'wsd'] = 'cosine', warmup_ratio: float = 0.1, lr_scheduler_config: lilax.training.optimizers.LRSchedulerConfig | None = None, optimizer_config: lilax.training.optimizers.OptimizerConfig | None = None, logging_steps: int = 100, eval_steps: int = -1, save_steps: int = 10000, eval_on_start: bool = False, checkpoint_config: lilax.checkpoints.policy.TrainingCheckpointConfig | None = None, report_to: str | list[str] = 'none', logging_dir: str | None = None, parallelism_config: lilax.distributed.ParallelismConfig | None = None, jax_distributed_initialization_timeout: int = 300, diloco_config: lilax.distributed.DiLoCoConfig | None = None, gradient_checkpointing: bool = True, remat_config: lilax.training.remat.RematConfig | None = None, objective_config: lilax.training.objectives.ObjectiveConfig | None = None, batch_rampup_config: lilax.training.objectives.BatchRampupConfig | None = None, max_length: int | None = None, packing: bool = True, completion_only_loss: bool = True, dataset_text_field: str = 'text', xprof_dir: str | None = None) -> None
```

### SFTTrainer

```python
SFTTrainer(model, *, args: lilax.trainers.post_training.sft.SFTConfig, formatting_func=None, data_collator=None, train_dataset=None, eval_dataset=None, processing_class=None, callbacks=None, optimizers=(None, None))
```

```python
SFTTrainer.train(self, resume_from_checkpoint=None, *, checkpoint_step=None)
```

```python
SFTTrainer.evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval')
```

```python
SFTTrainer.save_model(self, output_dir=None)
```

```python
SFTTrainer.save_state(self)
```

### DPOConfig

```python
DPOConfig(*, output_dir: str, run_name: str = '', max_steps: int = 150001, num_train_epochs: int = 1, per_device_train_batch_size: int | float = 12, per_device_eval_batch_size: int | float = 0.0, gradient_accumulation_steps: int = 1, learning_rate: float = 3e-05, optim: Literal['adamw', 'adam_pax', 'sgd', 'muon'] = 'adamw', adam_beta1: float = 0.9, adam_beta2: float = 0.95, adam_epsilon: float = 1e-08, weight_decay: float = 0.1, max_grad_norm: float = 1.0, lr_scheduler_type: Literal['cosine', 'wsd'] = 'cosine', warmup_ratio: float = 0.1, lr_scheduler_config: lilax.training.optimizers.LRSchedulerConfig | None = None, optimizer_config: lilax.training.optimizers.OptimizerConfig | None = None, logging_steps: int = 100, eval_steps: int = -1, save_steps: int = 10000, eval_on_start: bool = False, checkpoint_config: lilax.checkpoints.policy.TrainingCheckpointConfig | None = None, report_to: str | list[str] = 'none', logging_dir: str | None = None, parallelism_config: lilax.distributed.ParallelismConfig | None = None, jax_distributed_initialization_timeout: int = 300, diloco_config: lilax.distributed.DiLoCoConfig | None = None, gradient_checkpointing: bool = True, remat_config: lilax.training.remat.RematConfig | None = None, objective_config: lilax.training.objectives.ObjectiveConfig | None = None, batch_rampup_config: lilax.training.objectives.BatchRampupConfig | None = None, algorithm: Literal['dpo', 'orpo'] = 'dpo', beta: float = 0.1, label_smoothing: float = 0.0, orpo_lambda: float = 0.1, max_prompt_length: int | None = None) -> None
```

### DPOTrainer

```python
DPOTrainer(model, *, args: lilax.trainers.post_training.dpo.DPOConfig, data_collator=None, train_dataset=None, eval_dataset=None, processing_class=None, callbacks=None, optimizers=(None, None))
```

### DistillationConfig

```python
DistillationConfig(*, output_dir: str, run_name: str = '', max_steps: int = 150001, num_train_epochs: int = 1, per_device_train_batch_size: int | float = 12, per_device_eval_batch_size: int | float = 0.0, gradient_accumulation_steps: int = 1, learning_rate: float = 3e-05, optim: Literal['adamw', 'adam_pax', 'sgd', 'muon'] = 'adamw', adam_beta1: float = 0.9, adam_beta2: float = 0.95, adam_epsilon: float = 1e-08, weight_decay: float = 0.1, max_grad_norm: float = 1.0, lr_scheduler_type: Literal['cosine', 'wsd'] = 'cosine', warmup_ratio: float = 0.1, lr_scheduler_config: lilax.training.optimizers.LRSchedulerConfig | None = None, optimizer_config: lilax.training.optimizers.OptimizerConfig | None = None, logging_steps: int = 100, eval_steps: int = -1, save_steps: int = 10000, eval_on_start: bool = False, checkpoint_config: lilax.checkpoints.policy.TrainingCheckpointConfig | None = None, report_to: str | list[str] = 'none', logging_dir: str | None = None, parallelism_config: lilax.distributed.ParallelismConfig | None = None, jax_distributed_initialization_timeout: int = 300, diloco_config: lilax.distributed.DiLoCoConfig | None = None, gradient_checkpointing: bool = True, remat_config: lilax.training.remat.RematConfig | None = None, objective_config: lilax.training.objectives.ObjectiveConfig | None = None, batch_rampup_config: lilax.training.objectives.BatchRampupConfig | None = None, alpha: float = 0.5, temperature: float = 1.0, beta: float = 0.0, feature_loss_type: Literal['cosine', 'l2'] = 'cosine', layer_indices: list[int] | None = None, alpha_end: float | None = None, alpha_schedule: Literal['constant', 'linear', 'cosine'] = 'constant', temperature_end: float | None = None, temperature_schedule: Literal['constant', 'linear', 'cosine'] = 'constant', beta_end: float | None = None, beta_schedule: Literal['constant', 'linear', 'cosine'] = 'constant', learn_to_init: lilax.training.distillation.LearnToInitConfig | None = None) -> None
```

### DistillationTrainer

```python
DistillationTrainer(model, teacher_model, *, args: lilax.trainers.post_training.distillation.DistillationConfig, data_collator=None, train_dataset=None, eval_dataset=None, processing_class=None, callbacks=None, optimizers=(None, None))
```

```python
DistillationTrainer.generate_teacher_logits(self, *, dataset=None, top_k=128, optional_keys=('inputs_position', 'inputs_segmentation', 'targets_segmentation', 'targets'), gcs_upload_path=None, local_tmp_dir='/tmp', steps_per_file=50, upload_workers=4, serialization_workers=16)
```

```python
DistillationTrainer.train(self, resume_from_checkpoint=None, *, checkpoint_step=None)
```

```python
DistillationTrainer.evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval')
```

```python
DistillationTrainer.save_model(self, output_dir=None)
```

```python
DistillationTrainer.save_state(self)
```

### GRPOConfig

```python
GRPOConfig(*, output_dir: str, run_name: str = '', max_steps: NoneType = None, num_train_epochs: int = 1, per_device_train_batch_size: int | float = 12, per_device_eval_batch_size: int | float = 0.0, gradient_accumulation_steps: NoneType = None, learning_rate: float = 3e-05, optim: Literal['adamw', 'adam_pax', 'sgd', 'muon'] = 'adamw', adam_beta1: float = 0.9, adam_beta2: float = 0.95, adam_epsilon: float = 1e-08, weight_decay: float = 0.1, max_grad_norm: float = 1.0, lr_scheduler_type: Literal['cosine', 'wsd'] = 'cosine', warmup_ratio: float = 0.1, lr_scheduler_config: lilax.training.optimizers.LRSchedulerConfig | None = None, optimizer_config: lilax.training.optimizers.OptimizerConfig | None = None, logging_steps: int = 100, eval_steps: int = -1, save_steps: int = 10000, eval_on_start: bool = False, checkpoint_config: lilax.checkpoints.policy.TrainingCheckpointConfig | None = None, report_to: str | list[str] = 'none', logging_dir: str | None = None, parallelism_config: lilax.distributed.ParallelismConfig | None = None, jax_distributed_initialization_timeout: int = 300, diloco_config: lilax.distributed.DiLoCoConfig | None = None, gradient_checkpointing: bool = True, remat_config: lilax.training.remat.RematConfig | None = None, objective_config: lilax.training.objectives.ObjectiveConfig | None = None, batch_rampup_config: lilax.training.objectives.BatchRampupConfig | None = None, num_generations: int = 2, num_iterations: int = 1, beta: float = 0.08, epsilon: float = 0.2, epsilon_high: float | None = None, loss_type: Literal['grpo', 'gspo-token'] = 'grpo', loss_aggregation: Literal['token-mean', 'sequence-mean', 'sequence-mean-token-mean'] = 'sequence-mean-token-mean', use_rollout_logps: bool = True, reward_num_workers: int = 0, reward_worker_timeout_seconds: float = 180.0, rewards: lilax.rollout.rewards.MathRewardsConfig | None = None, batch_size: int = 1, num_batches: int = 4, train_micro_batch_size: int = -1, rollout_micro_batch_size: int = -1, max_seq_token_per_tpu: int = 0, evaluation_config: lilax.training.grpo.GRPOEvaluationConfig | None = None) -> None
```

### GRPOTrainer

```python
GRPOTrainer(model, reward_funcs=None, *, args: lilax.trainers.post_training.grpo_config.GRPOConfig, train_dataset=None, eval_dataset=None, processing_class=None, callbacks=None, optimizers=(None, None), ref_model=None, environment_factory=None, rollout_config=None, agentic_config=None)
```

```python
GRPOTrainer.train(self, resume_from_checkpoint=None, *, checkpoint_step=None)
```

```python
GRPOTrainer.evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval')
```

```python
GRPOTrainer.generate(self, prompts, num_passes=1)
```

```python
GRPOTrainer.save_model(self, output_dir=None)
```

```python
GRPOTrainer.save_state(self)
```

## lilax.configuration

`AutoConfig` is reexported from `lilax.configuration.loading`.

`MaxTextConfig` is reexported from `lilax.configuration.schema`.

### QwixQATConfig

```python
QwixQATConfig(*, rules: tuple[lilax.configuration.domains.data_qat.QwixQATRule, ...], provider: Literal['odml', 'qt'] = 'odml', disable_per_channel_weights: bool = False, fixed_range_for_inputs: tuple[float, float] | None = None, fixed_range_for_outputs: tuple[float, float] | None = None, strict: bool = True) -> None
```

### QwixQATRule

```python
QwixQATRule(*, module_path: str = '.*', op_names: tuple[str, ...] = (), weight_qtype: str | None = None, act_qtype: str | None = None, act_static_scale: bool | None = None, weight_calibration_method: str = 'absmax', act_calibration_method: str | None = None, act_batch_axes: tuple[int, ...] = (0,), tile_size: int | float | None = None) -> None
```

### ArchitectureConfig

```python
ArchitectureConfig(*, decoder_block: lilax.configuration.domain_types.DecoderBlockType = <DecoderBlockType.LLAMA2: 'llama2'>, dense_init_scale: float = 1.0, attention_output_dim: int = -1, global_head_dim: int = 0, mlp_activations: list[str] = <factory>, mlp_activations_limit: float = -1.0, normalization_layer_epsilon: float = 1e-05, fused_qkv: bool = False, attention_bias: bool = False, fused_mlp: bool = False, qk_norm_with_scale: bool = True, v_norm_with_scale: bool = True) -> None
```

### ModelLayoutConfig

```python
ModelLayoutConfig(*, inhomogeneous_layer_cycle_interval: int = 1) -> None
```

### ModelScalingConfig

```python
ModelScalingConfig(*, global_parameter_scale: int = 1, base_emb_dim: int = 2048, base_num_query_heads: int = 16, base_num_kv_heads: int = 16, base_mlp_dim: int = 7168, base_num_decoder_layers: int = 16, base_moe_mlp_dim: int = -1) -> None
```

```python
ModelScalingConfig.apply(self, config)
```

### AttentionConfig

```python
AttentionConfig(*, general: lilax.configuration.model_attention.AttentionParameters | None = None, moba: lilax.configuration.domains.model_attention_variants.MoBa | None = None, mla: lilax.configuration.domains.model_attention_variants.MlaAttention | None = None, compressed: lilax.configuration.domains.model_attention_variants.CompressedAttention | None = None, indexer: lilax.configuration.domains.model_attention_variants.AttentionIndexer | None = None, llama4: lilax.configuration.domains.model_attention_variants.Llama4Attention | None = None) -> None
```

### AttentionParameters

```python
AttentionParameters(*, attention_type: Literal['global', 'local_sliding', 'chunk', 'mla', 'full', 'compressed', 'block_diffusion'] = 'global', share_kv_projections: bool = False, global_num_kv_heads: int = 0, hidden_size_per_layer_input: Annotated[int, Ge(ge=0)] = 0, vocab_size_per_layer_input: Annotated[int, Ge(ge=0)] = 0, num_kv_shared_layers: Annotated[int, Ge(ge=0)] = 0, use_double_wide_mlp: bool = False, attention_sink: bool = False, float32_qk_product: bool = False, float32_logits: bool = False, sliding_window_size: Annotated[int, Ge(ge=0)] = 0, chunk_attn_window_size: Annotated[int, Ge(ge=0)] = 0, causal_block_size: Annotated[int, Gt(gt=0)] = 32, attn_logits_soft_cap: Optional[Annotated[float, Ge(ge=0)]] = None, use_post_attn_norm: bool = False, use_post_ffw_norm: bool = False, use_qk_clip: bool = False, qk_clip_threshold: float = 100.0, mla_qk_head_chunk_size: Annotated[int, Ge(ge=0)] = 0, csa_qk_head_chunk_size: Annotated[int, Ge(ge=0)] = 0) -> None
```

### AttentionIndexerConfig

```python
AttentionIndexerConfig(*, use_indexer: bool = False, indexer_head_dim: Annotated[int, Ge(ge=0)] = 128, indexer_n_heads: Annotated[int, Ge(ge=0)] = 64, indexer_topk: Annotated[int, Ge(ge=0)] = 2048, indexer_sparse_training: bool = False, indexer_loss_scaling_factor: float = 0.0, indexer_use_approx_top_k: bool = False, indexer_approx_top_k_recall: float = 0.95, indexer_mask_exact_topk: bool = True) -> None
```

### CompressedAttentionConfig

```python
CompressedAttentionConfig(*, o_lora_rank: Annotated[int, Ge(ge=0)] = 0, o_groups: Annotated[int, Ge(ge=0)] = 0, compress_ratios: list[int] = <factory>, compressed_rope_max_timescale: int = 160000, use_csa_streamindex_kernel: bool = False) -> None
```

### KernelConfig

```python
KernelConfig(*, attention: str = 'autoselected', use_ragged_attention: bool = False, enable_padding_causal_mask: bool = True, use_tokamax_splash: bool = False, use_jax_splash: bool = False, splash: lilax.configuration.model_attention.SplashAttentionConfig | None = None, moe: lilax.configuration.domains.model_moe.MoEKernels | None = None) -> None
```

### Llama4AttentionConfig

```python
Llama4AttentionConfig(*, use_qk_norm: bool = False, temperature_tuning: bool = False) -> None
```

### MlaAttentionConfig

```python
MlaAttentionConfig(*, mla_naive_kvcache: bool = True, q_lora_rank: Annotated[int, Ge(ge=0)] = 0, kv_lora_rank: Annotated[int, Ge(ge=0)] = 512, qk_nope_head_dim: Annotated[int, Ge(ge=0)] = 128, qk_rope_head_dim: Annotated[int, Ge(ge=0)] = 64, v_head_dim: Annotated[int, Ge(ge=0)] = 128, use_sliced_mla_proj: bool = False, use_mla_absorbed_mqa: bool = False) -> None
```

### MoBaConfig

```python
MoBaConfig(*, moba: bool = False, moba_chunk_size: int = 1024, moba_topk: int = 8) -> None
```

### MoEKernelsConfig

```python
MoEKernelsConfig(*, megablox: bool = True, sparse_matmul: bool = True, wi_tile_fwd_batch_seq: int = 512, wi_tile_fwd_embed_dim: int = 1024, wi_tile_fwd_mlp_dim: int = 1024, eval_wi_tile_fwd_batch_seq: int = 512, eval_wi_tile_fwd_embed_dim: int = 1024, eval_wi_tile_fwd_mlp_dim: int = 1024, wi_tile_dlhs_batch_seq: int = 512, wi_tile_dlhs_embed_dim: int = 1024, wi_tile_dlhs_mlp_dim: int = 1024, wi_tile_drhs_batch_seq: int = 512, wi_tile_drhs_embed_dim: int = 1024, wi_tile_drhs_mlp_dim: int = 1024, wo_tile_fwd_batch_seq: int = 512, wo_tile_fwd_embed_dim: int = 1024, wo_tile_fwd_mlp_dim: int = 1024, eval_wo_tile_fwd_batch_seq: int = 512, eval_wo_tile_fwd_embed_dim: int = 1024, eval_wo_tile_fwd_mlp_dim: int = 1024, wo_tile_dlhs_batch_seq: int = 512, wo_tile_dlhs_embed_dim: int = 1024, wo_tile_dlhs_mlp_dim: int = 1024, wo_tile_drhs_batch_seq: int = 512, wo_tile_drhs_embed_dim: int = 1024, wo_tile_drhs_mlp_dim: int = 1024, merge_gating_gmm: bool = False, num_moe_emb_chunks: int = 0, use_tokamax_gmm: bool = False, use_gmm_v2: bool = False, use_te_grouped_gemm: bool = False, use_gmm_v2_heuristic_tiling: bool = False) -> None
```

### PositionConfig

```python
PositionConfig(*, embedding: lilax.configuration.domains.data_position.PositionalEmbedding | None = None, rope: lilax.configuration.domains.data_position.Rope | None = None, yarn: lilax.configuration.domains.data_position.YarnRope | None = None, use_mrope: bool = False, mrope_section: list[int] = <factory>, position_id_per_seconds: int = 25) -> None
```

### PositionalEmbeddingConfig

```python
PositionalEmbeddingConfig(*, use_iota_embed: bool = False, use_untrainable_positional_embedding: bool = False, trainable_position_size: int = -1, nope_layer_interval: int = -1) -> None
```

### RopeConfig

```python
RopeConfig(*, rope_type: lilax.configuration.domain_types.RopeType = <RopeType.DEFAULT: 'default'>, rope_use_scale: bool = True, rope_min_timescale: int = 1, rope_max_timescale: int = 10000, rope_linear_scaling_factor: float = 1.0, local_rope_max_timescale: int = -1, global_rope_max_timescale: int = -1, global_rope_proportion: float = 0.25, local_rope_proportion: float = 1.0) -> None
```

### SplashAttentionConfig

```python
SplashAttentionConfig(*, experimental_sa_quant_q_fp8: bool = False, experimental_sa_quant_k_fp8: bool = False, sa_ignore_segment_ids: bool = False) -> None
```

### YarnRopeConfig

```python
YarnRopeConfig(*, max_position_embeddings: int = 163840, original_max_position_embeddings: int = 4096, rope_factor: int = 40, beta_fast: int = 32, beta_slow: int = 1, mscale: float = 1.0, rope_interleave: bool = True, rope_truncate: bool = True, rope_attention_scaling: bool = False, rope_pairwise: bool = False, rope_direct_position_freqs: bool = False, rope_dot_pairwise: bool = False) -> None
```

```python
YarnRopeConfig.validate_rope_pairwise(self) -> 'YarnRope'
```

### AudioConfig

```python
AudioConfig(*, d_model_for_audio: int = 256, encoder_attention_heads_for_audio: int = 4, encoder_ffn_dim_for_audio: int = 512, encoder_layers_for_audio: int = 2, attention_dropout_for_audio: float = 0.0, activation_dropout_for_audio: float = 0.0, activation_function_for_audio: str = 'gelu', num_mel_bins_for_audio: int = 128, max_source_positions_for_audio: int = 1500, scale_embedding_for_audio: bool = True, n_window_for_audio: int = 50, n_window_infer_for_audio: int = 800, conv_chunksize_for_audio: int = 500, downsample_hidden_size_for_audio: int = 256, output_dim_for_audio: int = 512, num_conv_layers_for_audio: int = 3, max_timescale_for_audio: float = 10000.0, max_sample_len_for_audio: int = 10000, use_audio: bool = False, freeze_audio_encoder_params: bool = True) -> None
```

### DeepSeekMoEConfig

```python
DeepSeekMoEConfig(*, first_num_dense_layers: Annotated[int, Ge(ge=0)] = 0, shared_experts: Annotated[int, Ge(ge=0)] = 0, routed_scaling_factor: float = 1.0, routed_score_func: str = '', routed_bias: bool = False, routed_bias_update_rate: float = 0.0, defer_small_all_reduces: bool = False, log_moe_bias_norms: bool = False, log_step_diagnostics: bool = False, mlp_bias: bool = False, n_routing_groups: int = -1, first_num_hash_layers: int = 0, topk_routing_group: int = -1, use_batch_split_schedule: bool = False, batch_split_factor: int = 1, use_lineage: bool = False, lineage_quantization: Literal['none', 'fp8_full', 'fp8_full_splash_qk', 'fp8_full_splash_qk_e3', 'fp8_full_combine_per_hop', 'fp8_full_splash_qk_e3_combine_per_hop'] = 'none', lineage_expert_permutation: Optional[Literal['lpt']] = None, lineage_diag_band_skip: bool = True, lineage_dq_reduction_steps: int = 1) -> None
```

### EngramConfig

```python
EngramConfig(*, engram_layers: list[int] = <factory>, engram_num_heads: int = 8, engram_head_dim: int = 1280, engram_vocab_bases: list[int] = <factory>, engram_max_ngram_size: int = 3, engram_kernel_size: int = 4, engram_seed: int = 0) -> None
```

### HyperconnectionConfig

```python
HyperconnectionConfig(*, mhc_expansion_rate: Annotated[int, Gt(gt=0)] = 1, sinkhorn_iterations: Annotated[int, Gt(gt=0)] = 20, enable_mhc_lite: bool = False, use_mhc_pallas_kernel: bool = False, mhc_pallas_kernel_fwd_block_size: int = 256, mhc_pallas_kernel_bwd_block_size: int = 128, mhc_pallas_kernel_bwd_feature_block_size: int = 1024, te_mhc: bool = False) -> None
```

```python
HyperconnectionConfig.validate_mhc_kernel(self) -> 'ManifoldConstrainedHyperConnections'
```

### LogitsConfig

```python
LogitsConfig(*, logits_via_embedding: bool = False, normalize_embedding_logits: bool = True, logits_dot_in_fp32: bool = False, cast_logits_to_fp32: bool = True, tie_lm_head_cotangents: bool = False, lm_head_weight_grad_in_kernel_order: bool | None = None, lm_head_fsdp_as_vocab_tp: bool = False, final_logits_soft_cap: Optional[Annotated[float, Ge(ge=0)]] = None, z_loss_multiplier: float = 0.0, num_vocab_tiling: int = 1, vocab_tiling_ag_once: bool = False) -> None
```

### MingConfig

```python
MingConfig(*, kda_lower_bound: float = -5.0, ming_audio_chunk_size: int = 3000, ming_norm_audio_embeds: bool = True) -> None
```

### MoEConfig

```python
MoEConfig(*, general: lilax.configuration.model_families.MoEParameters | None = None, deepseek: lilax.configuration.domains.model_moe.DeepSeekMoE | None = None) -> None
```

### MoEParameters

```python
MoEParameters(*, num_experts: Annotated[int, Gt(gt=0)] = 1, num_experts_per_tok: Annotated[int, Gt(gt=0)] = 1, capacity_factor: float = -1.0, ragged_buffer_factor: float = -1.0, eval_ragged_buffer_factor: float = -1.0, moe_dropless_fallback: Optional[Literal['step', 'layer']] = None, retry_dropless_first_steps: int = 0, first_phase_ragged_buffer_factor: float = 0.0, log_required_ragged_buffer_factor: bool = False, warmup_programs_in_init: bool = False, warmup_skip_train_dropless: bool = False, num_moe_token_chunks: Annotated[int, Gt(gt=0)] = 1, moe_accumulate_chunk_wgrad: bool = False, moe_accumulate_wi_dlhs: bool = False, moe_flat_fsdp_weights: bool = False, moe_gmm_v2_dlhs_transpose_rhs: bool = False, moe_bwd_prequant_before_unsort: bool = False, moe_combine_bwd_direct_qarray: bool = False, moe_3d_dlhs_partial_sum: bool = False, moe_chunk_barrier: bool = False, moe_log_max_load_ratio: bool = False, moe_expert_input_dim: int = -1, padded_base_moe_mlp_dim: int | None = None, load_balance_loss_weight: Annotated[float, Ge(ge=0)] = 0.0, moe_use_megatron_seq_aux_loss: bool = False, use_custom_sort_vjp: bool = True, router_topk_matmul_vjp: bool = False, moe_topk_before_ep_all_gather: bool = False, use_ring_of_experts: bool = False, te_moe_block: bool = False, te_moe_disable_cudnn_gmm_fusion: bool = False, te_moe_quantize_before_fsdp_all_gather: bool = False, te_ep_overflow_check_every_n_steps: Annotated[int, Gt(gt=0)] = 20, te_gmm_quantization: lilax.configuration.domains.model_moe.TEGroupedGemmQuantizationType | None = <TEGroupedGemmQuantizationType.EMPTY: ''>, moe_quantize_token_all_gather: bool = False, moe_dispatch_no_expert_sharding: bool = False, use_ragged_sort: bool = False, moe_quantize_combine_bwd_method: str | None = '', ragged_sort_use_single_sparsecore: bool = False, moe_tc_ragged_sort: bool = False, moe_tc_ragged_gather_block_size: int = 1024, moe_tc_ragged_reduce_block_size: int = 896, moe_tc_ragged_mask_padding: bool = True, moe_tc_ragged_weights_on_activation: bool = False, moe_tc_ragged_flatten_block_size: int = 0, moe_tc_ragged_3d_gmm: bool = False, moe_tc_ragged_3d_dispatch: bool = False, moe_tc_window_routing: bool = False, moe_use_direct_token_gather: bool = False, use_gather_mosaic_kernel: bool = False, ragged_gather_fallback: bool = False, ragged_gather_reduce_fallback: bool = False, ragged_gather_cost_estimate_flops: int = -1, ragged_gather_reduce_cost_estimate_flops: int = -1, ragged_gather_cost_estimate_bytes_accessed: int = -1, ragged_gather_reduce_cost_estimate_bytes_accessed: int = -1, moe_pin_sparse_core_all_gathers: bool = False, moe_pin_sparse_core_ep_all_gathers: bool = False, moe_pin_sparse_core_fsdp_all_gathers: bool = False, moe_pin_sparse_core_fsdp_all_gathers_fwd_only: bool = False, moe_fsdp_all_gather_sparse_core_id: int = 0, moe_ep_all_gather_sparse_core_id: int = 1, use_random_routing: bool = False, interleave_moe_layer_step: int = 1, moe_fsdp_use_two_stage_all_gather: bool = False, shard_exp_on_fsdp: bool = False, shard_embed_moe_on_fsdp: bool = False, use_2d_fsdp_sharding: bool = False, norm_topk_prob: bool = False, return_routed_experts: bool = False, float32_weight_sum: bool = True, float32_gate_logits: bool = False, prefuse_moe_weights: bool = False, fuse_expert_scales: bool = False, act_quantization_calibration_method: str = 'absmax') -> None
```

### MTPConfig

```python
MTPConfig(*, mtp_num_layers: Annotated[int, Ge(ge=0)] = 0, mtp_loss_scaling_factor: Annotated[float, Ge(ge=0)] = 0.1, mtp_eval_target_module: Annotated[int, Ge(ge=0)] = 0, mtp_reuse_input_embedding: bool = False, mtp_norm_main_hidden_state: bool = False) -> None
```

### QwenConfig

Exported type or value from `types`.

### Qwen3NextConfig

```python
Qwen3NextConfig(*, gdn_conv_kernel_dim: int = 4, gdn_key_head_dim: int = 128, gdn_value_head_dim: int = 128, gdn_num_key_heads: int = 16, gdn_num_value_heads: int = 32, gdn_chunk_size: int = 64, use_qk_norm_in_gdn: bool = True, gdn_mamba_block_size: int = 0, partial_rotary_factor: float = 1.0) -> None
```

### Qwen3_8FlashNextConfig

```python
Qwen3_8FlashNextConfig(*, hc_count: int = 4, hc_lowrank: int = 320, index_n_heads: int = 4, index_kv_heads: int = 1, index_head_dim: int = 128, index_select_block_size: int = 4, index_select_budget: int = 2048, gdn_output_gate_activation: str = 'sigmoid', gdn_chunk_inverse: Literal['solve', 'block_doubling'] = 'block_doubling', qsa_indexer_mask: Literal['gather', 'reshape'] = 'gather', qsa_attention_kernel: Literal['dense', 'flash'] = 'dense', use_ple_layer: bool = False, ple_layer_id: int = 1, ple_embed_dim: int = 2560, ple_conv_kernel_size: int = 4, ngram_size: int = 3, heads_per_ngram: int = 8, ngram_vocab_size_base: int = 20000000, make_ngram_vocab_size_divisible_by: int = 128, split_ngram_parts: int = 128, ple_seed: int = 1234, ple_eos_token_id: int = 248044) -> None
```

### VisionConfig

```python
VisionConfig(*, hidden_size_for_vit: int = 1408, intermediate_size_for_vit: int = 5632, num_attention_heads_for_vit: int = 16, num_channels_for_vit: int = 3, tile_size_for_vit: int = 336, patch_size_for_vit: int = 14, conv_stride_for_vit: int = 14, num_hidden_layers_for_vit: int = 34, rope_theta_for_vit: int = 10000, vision_output_dim_for_vit: int = 4096, spatial_merge_size_for_vit: int = 2, out_hidden_size_for_vit: int = 512, temporal_patch_size_for_vit: int = 2, num_position_embeddings_for_vit: int = 1024, deepstack_visual_indexes_for_vit: list[int] = [], vision_output_length: int = -1, use_multimodal: bool = False, attention_for_vit: str = 'dot_product', vision_encoder_block: lilax.configuration.domain_types.VisionEncoderBlockType = <VisionEncoderBlockType.NONE: 'none'>, freeze_vision_encoder_params: bool = True, image_size_for_vit: int | list[int] | None = 896, posemb_type_for_vit: str = 'learn') -> None
```

### VisionProjectorConfig

```python
VisionProjectorConfig(*, projector_input_dim_for_vit: int = 4096, projector_output_dim_for_vit: int = 4096, pixel_shuffle_ratio_for_vit: float = 0.5, projector_dropout_for_vit: float = 0.0, vision_projector_type: str = 'default', vision_connector_num_layers: int = 2, vision_connector_hidden_size: int = 0, vision_connector_activation: str = 'gelu', vision_connector_use_bias: bool = True) -> None
```

### PrecisionConfig

```python
PrecisionConfig(*, dtype: lilax.configuration.domain_types.DType = <DType.BFLOAT16: 'bfloat16'>, grad_dtype: lilax.configuration.domain_types.DType = <DType.FLOAT32: 'float32'>, weight_dtype: lilax.configuration.domain_types.DType = <DType.FLOAT32: 'float32'>, matmul_precision: lilax.configuration.domain_types.MatmulPrecision = <MatmulPrecision.DEFAULT: 'default'>, activations_in_float32: bool = False, dtype_mm: str = 'float32', unquantized_modules: list[str] = <factory>, weight_block_size: int | list[int] | None = None) -> None
```

### AqtInt8Config

```python
AqtInt8Config(*, quantization_local_shard_count: int = -1) -> None
```

### AqtMixedPrecisionConfig

```python
AqtMixedPrecisionConfig(*, quant_cfg_path: str = '') -> None
```

### BatchSplitManualConfig

```python
BatchSplitManualConfig(*, use_qwix_quantization: bool = False) -> None
```

### BatchSplitQwixConfig

```python
BatchSplitQwixConfig(*, weight_quantization_calibration_method: str = 'absmax', act_quantization_calibration_method: str = 'absmax', bwd_quantization_calibration_method: str = 'absmax') -> None
```

### ComputeRecipeConfig

```python
ComputeRecipeConfig(*, quantization: Literal[<QuantizationType.NONE: ''>, <QuantizationType.FP8: 'fp8'>, <QuantizationType.NANOO_FP8: 'nanoo_fp8'>, <QuantizationType.TE_NO_QUANT: 'te_no_quant'>, <QuantizationType.TE_FP8_DS: 'te_fp8_delayedscaling'>, <QuantizationType.TE_FP8_CS: 'te_fp8_currentscaling'>, <QuantizationType.TE_MXFP8: 'te_mxfp8'>, <QuantizationType.TE_NVFP4: 'te_nvfp4'>, <QuantizationType.TE_NVFP4_NO_RHT: 'te_nvfp4_no_rht'>] = <QuantizationType.NONE: ''>) -> None
```

### JaxMXConfig

```python
JaxMXConfig(*, quantization: Literal[<QuantizationType.JAX_MXFP8: 'jax_mxfp8'>, <QuantizationType.JAX_MXFP4: 'jax_mxfp4'>], jax_mx_gradient_mode: Literal['', 'mxfp8', 'mxfp4'] = '') -> None
```

### KVCacheQuantizationConfig

```python
KVCacheQuantizationConfig(*, quantize_kvcache: bool = False, kv_quant_axis: lilax.configuration.domain_types.KvQuantAxis = <KvQuantAxis.HEADS_AND_DKV: 'heads_and_dkv'>, kv_quant_dtype: Literal['int8', 'int4'] = 'int8') -> None
```

### QuantizationRecipe

Exported type or value from `types`.

### QwixFP8FullConfig

```python
QwixFP8FullConfig(*, quantize_mtp: bool = False, quantize_router_proj: bool = True, quantize_logits_proj: bool = False, logits_proj_quant_calibration_method: str = '', logits_proj_bwd_quant_calibration_method: str = '', sa_qproj_bwd_bf16: bool = False, weight_quantization_calibration_method: str = 'absmax', act_quantization_calibration_method: str = 'absmax', bwd_quantization_calibration_method: str = 'absmax', drhs_grad_quantization_calibration_method: str | None = None) -> None
```

### QwixRecipeConfig

```python
QwixRecipeConfig(*, quantization: Literal[<QuantizationType.NONE: ''>, <QuantizationType.INT4: 'int4'>, <QuantizationType.INT8: 'int8'>, <QuantizationType.FP4: 'fp4'>, <QuantizationType.FP8_E5M2: 'fp8_e5m2'>, <QuantizationType.FP8_E4M3: 'fp8_e4m3'>, <QuantizationType.FP8: 'fp8'>, <QuantizationType.FP8_GPU: 'fp8_gpu'>, <QuantizationType.FP8_NANO_V2: 'fp8_nanoo'>] = <QuantizationType.NONE: ''>, quantize_router_proj: bool = True, qwix_bwd_weight_grad_tile_count: Annotated[int, Gt(gt=0)] = 1) -> None
```

### ServeFP8WeightConfig

```python
ServeFP8WeightConfig(*, act_quantization_calibration_method: str = 'absmax') -> None
```

### SparsityConfig

```python
SparsityConfig(*, weight_sparsity_n: int | None = None, weight_sparsity_m: int | None = None, weight_sparsity_update_step: int = 10, weight_sparsity_start_step: int = 50) -> None
```

### DType

`BFLOAT16 = 'bfloat16'`  
`FLOAT32 = 'float32'`  
`FLOAT16 = 'float16'`  
`FLOAT8_E4M3FN = 'float8_e4m3fn'`  
`FLOAT8_E5M2 = 'float8_e5m2'`  

### MatmulPrecision

`DEFAULT = 'default'`  
`HIGH = 'high'`  
`HIGHEST = 'highest'`  
`BFLOAT16 = 'bfloat16'`  
`FLOAT32 = 'float32'`  

### QuantizationType

`NONE = ''`  
`FP4 = 'fp4'`  
`INT4 = 'int4'`  
`INT8 = 'int8'`  
`INTMP = 'intmp'`  
`FP8_E5M2 = 'fp8_e5m2'`  
`FP8_E4M3 = 'fp8_e4m3'`  
`FP8 = 'fp8'`  
`NANOO_FP8 = 'nanoo_fp8'`  
`FP8_NANO_V2 = 'fp8_nanoo'`  
`FP8_GPU = 'fp8_gpu'`  
`FP8_FULL = 'fp8_full'`  
`TE_NO_QUANT = 'te_no_quant'`  
`TE_FP8_DS = 'te_fp8_delayedscaling'`  
`TE_FP8_CS = 'te_fp8_currentscaling'`  
`TE_MXFP8 = 'te_mxfp8'`  
`JAX_MXFP8 = 'jax_mxfp8'`  
`JAX_MXFP4 = 'jax_mxfp4'`  
`TE_NVFP4 = 'te_nvfp4'`  
`TE_NVFP4_NO_RHT = 'te_nvfp4_no_rht'`  
`SERVE_FP8_WEIGHT = 'serve_fp8_weight'`  

### KvQuantAxis

`NONE = ''`  
`DKV = 'dkv'`  
`HEADS_AND_DKV = 'heads_and_dkv'`  

### DecoderBlockType

`DEFAULT = 'default'`  
`LLAMA2 = 'llama2'`  
`MISTRAL = 'mistral'`  
`MIXTRAL = 'mixtral'`  
`DEEPSEEK = 'deepseek'`  
`GEMMA = 'gemma'`  
`GEMMA2 = 'gemma2'`  
`GEMMA3 = 'gemma3'`  
`GEMMA4 = 'gemma4'`  
`GEMMA4_SMALL = 'gemma4_small'`  
`QWEN2 = 'qwen2'`  
`QWEN3 = 'qwen3'`  
`QWEN3_MOE = 'qwen3_moe'`  
`QWEN3_CUSTOM_MOE = 'qwen3_custom_moe'`  
`QWEN3_NEXT = 'qwen3_next'`  
`QWEN3_5 = 'qwen3_5'`  
`GPT3 = 'gpt3'`  
`GPT_OSS = 'gpt_oss'`  
`SIMPLE = 'simple'`  
`SIMPLE_MLP = 'simple_mlp'`  
`LLAMA4 = 'llama4'`  
`OLMO3 = 'olmo3'`  
`DEEPSEEK4 = 'deepseek4'`  
`ENVY = 'envy'`  
`WEAVER = 'weaver'`  
`MING = 'ming'`  
`HY4 = 'hy4'`  

### RopeType

`DEFAULT = 'default'`  
`LLAMA3_1 = 'llama3.1'`  
`YARN = 'yarn'`  

### ProfilerType

`NONE = ''`  
`XPLANE = 'xplane'`  
`NSYS = 'nsys'`  

### VisionEncoderBlockType

`NONE = 'none'`  
`GEMMA3 = 'gemma3'`  
`LLAMA4 = 'llama4'`  
`QWEN3_OMNI = 'qwen3_omni'`  
`GEMMA4 = 'gemma4'`  
`QWEN3_5 = 'qwen3_5'`  
`QWEN3_VL = 'qwen3_vl'`  

### TEGroupedGemmQuantizationType

`EMPTY = ''`  
`TE_NO_QUANT = 'te_no_quant'`  
`TE_MXFP8 = 'te_mxfp8'`  

## lilax.data

### Dataset

```python
Dataset(arrow_table: datasets.table.Table, info: Optional[datasets.info.DatasetInfo] = None, split: Optional[datasets.splits.NamedSplit] = None, indices_table: Optional[datasets.table.Table] = None, fingerprint: Optional[str] = None)
```

### IterableDataset

```python
IterableDataset(ex_iterable: datasets.iterable_dataset._BaseExamplesIterable, info: Optional[datasets.info.DatasetInfo] = None, split: Optional[datasets.splits.NamedSplit] = None, formatting: Optional[datasets.iterable_dataset.FormattingConfig] = None, distributed: Optional[datasets.iterable_dataset.DistributedConfig] = None, token_per_repo_id: Optional[dict[str, Union[str, bool, NoneType]]] = None)
```

### load_dataset

```python
load_dataset(path: str, name: Optional[str] = None, data_dir: Optional[str] = None, data_files: Union[str, collections.abc.Sequence[str], collections.abc.Mapping[str, Union[str, collections.abc.Sequence[str]]], NoneType] = None, split: Union[str, datasets.splits.Split, list[str], list[datasets.splits.Split], NoneType] = None, cache_dir: Optional[str] = None, features: Optional[datasets.features.features.Features] = None, download_config: Optional[datasets.download.download_config.DownloadConfig] = None, download_mode: Union[datasets.download.download_manager.DownloadMode, str, NoneType] = None, verification_mode: Union[datasets.utils.info_utils.VerificationMode, str, NoneType] = None, keep_in_memory: Optional[bool] = None, save_infos: bool = False, revision: Union[datasets.utils.version.Version, str, NoneType] = None, token: Union[bool, str, NoneType] = None, streaming: bool = False, num_proc: Optional[int] = None, storage_options: Optional[dict] = None, **config_kwargs) -> Union[datasets.dataset_dict.DatasetDict, datasets.arrow_dataset.Dataset, datasets.dataset_dict.IterableDatasetDict, datasets.iterable_dataset.IterableDataset]
```

### SyntheticDataset

```python
SyntheticDataset(*, reuse_example_batch: int = 0, max_segments_per_seq: int = -1) -> None
```

### TeacherLogitsDataset

```python
TeacherLogitsDataset(files: str) -> None
```

### RLDataOptions

```python
RLDataOptions(*, train_fraction: Annotated[float, Gt(gt=0.0), Le(le=1.0)] = 1.0, test_batch_start_index: int = 0, processor: Optional[Callable[..., dict]] = None) -> None
```

### prepare_rl_dataset

```python
prepare_rl_dataset(dataset, *, processing_class, options=None)
```

### train_tokenizer_from_files

```python
train_tokenizer_from_files(files: str, *, file_type: Literal['parquet', 'arrayrecord', 'tfrecord'] = 'parquet', data_column: str = 'text', vocab_size: int = 32768, max_corpus_chars: int = 10000000, assets_path: str = '/home/shadowlilac/maxtext/lilax/src/lilax/_backend/assets', vocab_model_name: str = 'tokenizer')
```

## lilax.processing

`AutoTokenizer` is reexported from `lilax.processing`.

`AutoProcessor` is reexported from `lilax.processing`.

### processing_overrides

```python
processing_overrides(processing_class)
```

## lilax.training

### BatchRampupConfig

```python
BatchRampupConfig(*, enable_rampup_batch_size: bool = False, per_device_batch_size_start: float = 4.0, per_device_batch_size_increment: float = 2.0, global_rampup_samples: int = 500) -> None
```

### ObjectiveConfig

```python
ObjectiveConfig(*, training_objective: Literal['causal_lm', 'block_diffusion'] = 'causal_lm', block_diffusion_mask_id: int = -1, block_diffusion_min_noise: float = 0.001, block_diffusion_logit_alignment: Literal['same_position', 'shifted'] = 'same_position', block_diffusion_canvas_policy: Literal['all_masked', 'seed_and_mask'] = 'all_masked') -> None
```

### LRSchedulerConfig

```python
LRSchedulerConfig(*, schedule_steps: int = -1, final_fraction: float = 0.1, wsd_decay_steps_fraction: float = 0.1, wsd_decay_style: Literal['linear', 'cosine'] = 'linear') -> None
```

### MuonConfig

```python
MuonConfig(*, implementation: Literal['optax_muon', 'maxtext_muon'] = 'optax_muon', beta: float = 0.95, weight_decay: float = 0.0, consistent_rms: float | None = None, include_routers: bool = True, use_all_to_all: bool = False) -> None
```

### OptimizerConfig

```python
OptimizerConfig(*, adam_eps_root: float = 0.0, adamw_mask: list[str] | None = None, mu_dtype: str = '', muon: lilax.training.optimizers.MuonConfig | None = None, spike_skip: lilax.training.optimizers.SpikeSkipConfig | None = None, trainable_parameters: list[str] | None = None) -> None
```

### SpikeSkipConfig

```python
SpikeSkipConfig(*, enabled: bool = False, interval: int = 128, scaling_factor: float = 6.0) -> None
```

### RematConfig

```python
RematConfig(*, remat_policy: str = 'full', remat_policy_for_vit: str = 'minimal', optimizer_memory_host_offload: bool = False, parameter_memory_host_offload: bool = False) -> None
```

### LearnToInitConfig

```python
LearnToInitConfig(*, enabled: bool = False, use_general_linear_map: bool = False, weights_copy_map: dict[str, str] | None = None, student_weights_share_map: dict[str, str] | None = None, attention_module_name: str | None = None, layer_indices: list[int] | None = None, student_params_to_update: list[str] | None = None) -> None
```

### GRPOEvaluationConfig

```python
GRPOEvaluationConfig(*, eval_batch_size: int = -1, sampling_strategy: str = 'greedy', generation_configs: dict[str, lilax.generation.configuration.GenerationConfig] | None = None, num_eval_passes: int = 1, response_selection: Literal['correct', 'incorrect'] = 'incorrect', return_responses: bool = False, mode: Literal['pass', 'maj', 'pass_at_1'] = 'pass') -> None
```

## lilax.distributed

### MeshConfig

```python
MeshConfig(*, shard_mode: lilax.configuration.domains.execution_mesh.ShardMode = <ShardMode.AUTO: 'auto'>, context_parallel_strategy: str = 'all_gather') -> None
```

### ShardingConfig

```python
ShardingConfig(*, shard_optimizer_over_data: bool = False, save_fsdp_gathered_weights: bool = False, prefetch_fsdp_weights: bool = False, prefetch_fsdp_defer_grad_reduce_scatter: bool = False) -> None
```

### ParallelismConfig

```python
ParallelismConfig(*, mesh: lilax.distributed.MeshConfig | None = None, sharding: lilax.distributed.ShardingConfig | None = None) -> None
```

### DiLoCoConfig

```python
DiLoCoConfig(*, enable_diloco: bool = False, diloco_sync_period: int = 36, diloco_outer_lr: float = 0.3, diloco_outer_momentum: float = 0.9) -> None
```

### ShardMode

`AUTO = 'auto'`  
`EXPLICIT = 'explicit'`  

## lilax.callbacks

### TrainerCallback

```python
TrainerCallback()
```

### TrainerControl

```python
TrainerControl(should_training_stop: bool = False, should_epoch_stop: bool = False, should_save: bool = False, should_evaluate: bool = False, should_log: bool = False) -> None
```

### TrainerState

```python
TrainerState(epoch: float = 0, global_step: int = 0, max_steps: int = 0, logging_steps: int = 500, eval_steps: int = 500, save_steps: int = 500, train_batch_size: int | None = None, num_train_epochs: int = 0, num_input_tokens_seen: int = 0, total_flos: float = 0, log_history: list[dict[str, float]] = None, best_metric: float | None = None, best_global_step: int | None = None, best_model_checkpoint: str | None = None, is_local_process_zero: bool = True, is_world_process_zero: bool = True, is_hyper_param_search: bool = False, trial_name: str | None = None, trial_params: dict[str, str | float | int | bool] | None = None, stateful_callbacks: list['TrainerCallback'] | None = None) -> None
```

### TensorBoardCallback

```python
TensorBoardCallback(*, use_vertex_tensorboard=False, project='', region='')
```

```python
TensorBoardCallback.on_train_begin(self, args, state, control, **kwargs)
```

```python
TensorBoardCallback.on_log(self, args, state, control, logs=None, **kwargs)
```

```python
TensorBoardCallback.on_evaluate(self, args, state, control, metrics=None, **kwargs)
```

```python
TensorBoardCallback.on_train_end(self, args, state, control, **kwargs)
```

### TrackioCallback

```python
TrackioCallback(*, project='maxtext', name='')
```

```python
TrackioCallback.on_train_begin(self, args, state, control, **kwargs)
```

```python
TrackioCallback.on_log(self, args, state, control, logs=None, **kwargs)
```

```python
TrackioCallback.on_evaluate(self, args, state, control, metrics=None, **kwargs)
```

```python
TrackioCallback.on_train_end(self, args, state, control, **kwargs)
```

### WandbCallback

```python
WandbCallback(*, project='maxtext', name='')
```

```python
WandbCallback.on_train_begin(self, args, state, control, **kwargs)
```

```python
WandbCallback.on_log(self, args, state, control, logs=None, **kwargs)
```

```python
WandbCallback.on_evaluate(self, args, state, control, metrics=None, **kwargs)
```

```python
WandbCallback.on_train_end(self, args, state, control, **kwargs)
```

### NonFiniteLossCallback

```python
NonFiniteLossCallback(*, abort_on_nan_loss=True, abort_on_inf_loss=True)
```

```python
NonFiniteLossCallback.on_step_end(self, args, state, control, loss=None, **kwargs)
```

### TargetEvalLossCallback

```python
TargetEvalLossCallback(target_eval_loss=0.0)
```

```python
TargetEvalLossCallback.on_evaluate(self, args, state, control, metrics=None, metric_key_prefix='eval', **kwargs)
```

### ProfilerCallback

```python
ProfilerCallback(config: lilax.profiling.ProfilingConfig)
```

### RLProfilerCallback

```python
RLProfilerCallback(*, start_invocation=2, num_invocations=1, timeout_secs=60.0)
```

### MetricsCallback

```python
MetricsCallback(*, metrics_file=None, gcs_metrics=False, save_config_to_gcs=False)
```

```python
MetricsCallback.on_init_end(self, args, state, control, **kwargs)
```

```python
MetricsCallback.on_train_begin(self, args, state, control, **kwargs)
```

```python
MetricsCallback.on_log(self, args, state, control, logs=None, **kwargs)
```

```python
MetricsCallback.on_evaluate(self, args, state, control, metrics=None, **kwargs)
```

### MLPerfCallback

```python
MLPerfCallback(*, mllog_file='')
```

## lilax.checkpoints

### TrainingCheckpointConfig

```python
TrainingCheckpointConfig(*, enabled: bool = True, async_checkpointing: bool = True, max_num_checkpoints_to_keep: int | None = None, save_checkpoint_on_start: bool = True, save_checkpoint_on_completion: bool = True, enable_continuous_checkpointing: bool = False, enable_autocheckpoint: bool = False, save_only_lora_params: bool | None = None, restore_only_lora_params: bool | None = None) -> None
```

### HuggingFaceExportConfig

```python
HuggingFaceExportConfig(*, hf_model_path: str | pathlib.Path | None = None, revision: str | None = None, token: str | bool | None = None, cache_dir: str | pathlib.Path | None = None, local_files_only: bool = False, override_model_architecture: bool = False) -> None
```

### CheckpointConversionConfig

```python
CheckpointConversionConfig(*, conversion_output_dir: str | pathlib.Path | None = None, lazy_load_tensors: bool = True, eager_load_method: str = 'safetensors', save_dtype: str = 'bfloat16', simulated_cpu_devices_count: int = 16) -> None
```

### convert_checkpoint

```python
convert_checkpoint(source, destination, *, fused_qkv=None, fused_mlp=None, config: lilax.checkpoints.importing.CheckpointConversionConfig | None = None)
```

## lilax.generation

### ContinuousBatchingConfig

```python
ContinuousBatchingConfig(*, enable_batch_prefill: bool = False, min_decode_steps: int = 10, prefill_lengths: Union[list[int], Literal['auto']] = 'auto', batch_prefill_max_batch_size: int = 16, debug: bool = False) -> None
```

`GenerationConfig` is reexported from `lilax.generation.configuration`.

### CompletionOutput

```python
CompletionOutput(index: str, token_ids: numpy.ndarray, logprobs: numpy.ndarray, prompt_length: int) -> None
```

### log_prob_of_chosen_token

```python
log_prob_of_chosen_token(logits, chosen_index)
```

### prompt_logprobs_from_packed_prefill

```python
prompt_logprobs_from_packed_prefill(logits, input_tokens, decoder_positions, decoder_segment_ids, true_lengths)
```

### prompt_logprobs_from_prefill

```python
prompt_logprobs_from_prefill(logits, input_tokens, true_length)
```

### sampling

```python
sampling(logits, rng, algorithm, topk=0, nucleus_topp=0, temperature=1.0)
```

## lilax.rollout

### AgentBinding

```python
AgentBinding(*, agent_class: type, environment_class: type, agent_kwargs: dict[str, typing.Any] | None = None, environment_kwargs: dict[str, typing.Any] | None = None) -> None
```

### AgenticRolloutConfig

```python
AgenticRolloutConfig(*, enabled: bool = False, max_concurrency: int = 256, off_policy_steps: int = 0, system_prompt: str = '', agent: lilax.rollout.agentic.AgentBinding | None = None) -> None
```

### RolloutClusterConfig

```python
RolloutClusterConfig(*, trainer_devices_fraction: Annotated[float, Gt(gt=0.0), Le(le=1.0)] = 0.5, sampler_devices_fraction: Annotated[float, Gt(gt=0.0), Le(le=1.0)] = 0.5, chips_per_vm: int = 4, use_pathways: bool = True, num_trainer_slices: int = -1, num_samplers_slices: int = -1, rollout_data_parallelism: int = -1, rollout_tensor_parallelism: int = -1, rollout_expert_parallelism: int = 1, inference_replicas: int = 1, inference_devices_per_replica: int = 4, inference_rollouts: int = 1, use_pathways_reshard: bool = True, gc_collect_after_weight_sync: bool = True) -> None
```

### RolloutConfig

```python
RolloutConfig(*, generation_config: lilax.generation.configuration.GenerationConfig | None = None, max_prompt_length: int | None = None, engine: Optional[Literal['maxtext', 'vllm']] = None, reshard_chunk_size: int | None = None, cluster: lilax.rollout.configuration.RolloutClusterConfig | None = None, vllm: lilax.rollout.configuration.VLLMRolloutConfig | None = None, weight_sync: lilax.rollout.configuration.WeightSyncConfig | None = None) -> None
```

### VLLMRolloutConfig

```python
VLLMRolloutConfig(*, kv_cache_buffer: Annotated[int, Gt(gt=0)] = 256, hbm_utilization_vllm: Annotated[float, Gt(gt=0.0), Le(le=1.0)] = 0.72, swap_space_vllm_gb: Annotated[int, Ge(ge=0)] = 2, enable_dp_attention: bool = False, enable_expert_parallel: bool = False, async_scheduling: bool = False, max_num_batched_tokens: int | None = None, max_num_seqs: int | None = None, vllm_block_size: Annotated[int | None, Gt(gt=0)] = None, stop_strings: list[str] | None = None, additional_config: dict[str, typing.Any] = <factory>, hf_overrides: dict[str, typing.Any] = <factory>, vllm_hf_config_path: str = '') -> None
```

### WeightSyncConfig

```python
WeightSyncConfig(*, free_kv_cache_during_weight_sync: bool = True, use_standalone_converter: bool = False, use_weight_converter: bool = True, use_raiden_ffi: bool | None = None, rollout_backend: Literal['maxtext', 'vllm_torchax'] = 'maxtext', rollout_weight_dtype: str = '', weight_sync_debug: bool = False, log_weight_sync_time: bool = False, kv_tp_size: Annotated[int, Ge(ge=1)] = 1, moe_mlp_tp_size: Annotated[int, Ge(ge=1)] = 1, vllm_load_format: str = 'dummy', debug_converter: bool = False, gcs_debug_path: str = '') -> None
```

### MathRewardsConfig

```python
MathRewardsConfig(*, exact_answer: float = 5.0, exact_format_match: float = 3.0, white_space_format_match: float = 1.5, partial_format_match: float = 0.5, ratio_guess_to_answer_high: float = 0.5, ratio_guess_to_answer_low: float = 0.25, incorrect_format: float = -0.5, incorrect_answer: float = -1.0, math_verify_timeout: int = 300, math_verify_num_procs: int | None = None, special_tokens: lilax.rollout.rewards.RewardSpecialTokens | None = None) -> None
```

### RewardSpecialTokens

```python
RewardSpecialTokens(*, reasoning_start_token: str = '<reasoning>', reasoning_end_token: str = '</reasoning>', reasoning_start_token_in_prompt: bool = False, solution_start_token: str = '<answer>', solution_end_token: str = '</answer>') -> None
```

## lilax.serving

### EvaluationServerConfig

```python
EvaluationServerConfig(*, max_model_len: int, host: str = 'localhost', port: int = 8000, tensor_parallel_size: int = 4, expert_parallel_size: int = 1, data_parallel_size: int = 1, max_num_batched_tokens: int | None = None, max_num_seqs: int | None = None, hbm_memory_utilization: float = 0.3, concurrency: int | None = None, enable_chat_api: bool = False) -> None
```

### InferenceCacheConfig

```python
InferenceCacheConfig(*, stack_prefill_result_cache: bool = False, prefill_cache_axis_order: str = '1,2,0,3', ar_cache_axis_order: str = '1,2,0,3', compute_axis_order: str = '0,1,2,3', reshape_q: bool = False, load_from_prefill_dir: bool = False, prefill_cache_dir: str = '') -> None
```

### JetStreamConfig

```python
JetStreamConfig(*, inference_server: str = 'MaxtextInterleavedServer', prefill_slice: str = 'v5e-16', generate_slice: str = 'v5e-16', use_chunked_prefill: bool = False, prefill_chunk_size: int = 256, enable_prefix_caching: bool = False, prefix_caching_hbm_byte: int = 10000000000, prefix_caching_dram_byte: int = 100000000000, enable_model_warmup: bool = False, lora_input_adapters_path: str = '', multi_sampling: bool = False, prometheus_port: int = 0, parallelism_config: lilax.distributed.ParallelismConfig | None = None, jax_distributed_initialization_timeout: int = 300, cache_config: lilax.serving.configuration.InferenceCacheConfig | None = None) -> None
```

### VLLMConfig

```python
VLLMConfig(*, hbm_utilization_vllm: float = 0.72, enable_dp_attention: bool = False, async_scheduling: bool = False, max_num_batched_tokens: int | None = None, max_num_seqs: int | None = None, vllm_hf_config_path: str = '', hf_overrides: dict[str, typing.Any] = <factory>, additional_config: dict[str, typing.Any] = <factory>, parallelism_config: lilax.distributed.ParallelismConfig | None = None, jax_distributed_initialization_timeout: int = 300) -> None
```

### ServingEndpoint

```python
ServingEndpoint(model, *, config: lilax.serving.configuration.EvaluationServerConfig, processing_class=None)
```

### serve

```python
serve(model, *, server_config: lilax.serving.configuration.JetStreamConfig | None = None) -> None
```

### start_server

```python
start_server(model, *, server_config: lilax.serving.configuration.JetStreamConfig | None = None) -> jetstream.core.server_lib.JetStreamServer
```

### create_vllm_llm

```python
create_vllm_llm(model, *, serving_config: lilax.serving.configuration.VLLMConfig | None = None)
```

## lilax.evaluation

### DatasetEvalConfig

```python
DatasetEvalConfig(*, benchmark: str, num_samples: int | None = None, max_tokens: int = 1024, temperature: float = 0.0, concurrency: int = 64, request_timeout: int = 600, max_retries: int = 3, skip_warmup: bool = False) -> None
```

### EvalOutputConfig

```python
EvalOutputConfig(*, results_path: str, gcs_results_path: str | None = None) -> None
```

### HarnessEvalConfig

```python
HarnessEvalConfig(*, backend: Literal['evalchemy', 'lm_eval'] = 'evalchemy', tasks: tuple[str, ...] | list[str] = ('mmlu',), num_fewshot: int = 0, num_samples: int | float | None = None, apply_chat_template: bool = False, fewshot_as_multiturn: bool = False, generation_config: lilax.generation.configuration.GenerationConfig | None = None, skip_warmup: bool = False) -> None
```

### SimpleEvalsConfig

```python
SimpleEvalsConfig(*, tasks: tuple[typing.Literal['mmlu', 'gpqa', 'gsm8k', 'drop', 'mgsm', 'mgsm_en', 'aime2024', 'aime2025'], ...] = ('mmlu',), num_samples: int | None = None, n_repeats: int | None = None, max_tokens: int = 2048, temperature: float = 0.5, system_message: str | None = 'You are a helpful assistant.', reasoning_effort: str | None = None, log_debug_info: bool = False, continue_on_request_error: bool = False, skip_warmup: bool = False) -> None
```

### evaluate

```python
evaluate(model, *, config: lilax.evaluation.configuration.DatasetEvalConfig | lilax.evaluation.configuration.HarnessEvalConfig | lilax.evaluation.configuration.SimpleEvalsConfig, output: lilax.evaluation.configuration.EvalOutputConfig, processing_class=None, eval_dataset=None, compute_metrics=None, token=None)
```

## lilax.evaluation.benchmarks

### evaluate_mmlu

```python
evaluate_mmlu(model, *, processing_class, eval_dataset, prompt_template='The following are multiple choice questions (with answers) about {subject}.\n\n{question}\n{choices}\nAnswer:')
```

### evaluate_multimodal

```python
evaluate_multimodal(model, *, processing_class, eval_dataset, num_examples=-1, tmp_results_file='mm_eval_results.csv', remove_tmp_results=True, image_resize=-1)
```

### evaluate_omni

```python
evaluate_omni(model, *, processing_class, eval_dataset, num_examples=-1, tmp_results_file='omni_eval_results.csv', remove_tmp_results=True, image_resize=-1)
```

## lilax.evaluation.metrics

### score_rouge

```python
score_rouge(responses: list[str], references: list[str], use_stemmer: bool = True) -> dict
```

## lilax.profiling

### ProfilingConfig

```python
ProfilingConfig(*, profiler: lilax.configuration.domain_types.ProfilerType = <ProfilerType.NONE: ''>, skip_first_n_steps_for_profiler: int = 1, profiler_steps: int = 5, profile_cleanly: bool = True, profile_periodically_period: int = -1, hide_profiler_step_metric: bool = False, enable_continuous_profiling: bool = False) -> None
```

### profile_m3

```python
profile_m3(trainer) -> float
```

## lilax.openenv

### OpenEnvAgent

```python
OpenEnvAgent(system_prompt: str, action_from_response: collections.abc.Callable[[str], typing.Any], observation_to_text: collections.abc.Callable[[typing.Any], str], status_mapper: collections.abc.Callable[[typing.Any, float, dict[str, typing.Any]], tunix.rl.agentic.agents.agent_types.TrajectoryStatus])
```

```python
OpenEnvAgent.update_from_model(self, response: str, **kwargs) -> tunix.rl.agentic.agents.agent_types.Action
```

```python
OpenEnvAgent.update_from_env(self, observation: Any, reward: float, done: bool, info: dict[str, typing.Any] | None = None, **kwargs) -> None
```

### OpenEnvEnvironment

```python
OpenEnvEnvironment(task, *, group_id, pair_index, client_factory, reset_kwargs, reward_from_result, max_steps)
```

```python
OpenEnvEnvironment.reset(self)
```

```python
OpenEnvEnvironment.step(self, action)
```

```python
OpenEnvEnvironment.close(self)
```

### InitialObservation

```python
InitialObservation(task: dict[str, typing.Any], observation: Any) -> None
```

### TerminalStatus

`SUCCEEDED = 1`  
`RUNNING = 2`  
`MAX_STEPS_REACHED = 3`  
`MAX_CONTEXT_LIMIT_REACHED = 4`  
`TIMEOUT = 5`  
`ENV_TIMEOUT = 6`  
`FAILED = 7`  
`CANCELLED = 8`  

## lilax.peft

### LoraConfig

```python
LoraConfig(*, r: int, lora_alpha: float, target_modules: str = '', lora_weight_qtype: str | None = None, lora_tile_size: int | None = None) -> None
```

### PeftModel

```python
PeftModel(native_model, runtime, adapter_config=None, *, base_model=None, peft_config=None, adapter_source=None)
```

```python
PeftModel.prepare_for_full_state_restore(self, qat_config, *, runtime)
```

```python
PeftModel.materialize(self, *, runtime)
```

```python
PeftModel.save_pretrained(self, save_directory, *, format='huggingface')
```

```python
PeftModel.from_pretrained(cls, model, model_id, peft_config: lilax.peft.LoraConfig, *, format='maxtext', revision=None, token=None)
```

### get_peft_model

```python
get_peft_model(model, peft_config: lilax.peft.LoraConfig) -> lilax.peft.PeftModel
```

## lilax.trainers

`Trainer` is reexported from `lilax.trainers.pretraining.trainer`.

`TrainingArguments` is reexported from `lilax.trainers.arguments.training_args`.

`SFTConfig` is reexported from `lilax.trainers.post_training.sft`.

`SFTTrainer` is reexported from `lilax.trainers.post_training.sft`.

`DPOConfig` is reexported from `lilax.trainers.post_training.dpo`.

`DPOTrainer` is reexported from `lilax.trainers.post_training.dpo`.

`DistillationConfig` is reexported from `lilax.trainers.post_training.distillation`.

`DistillationTrainer` is reexported from `lilax.trainers.post_training.distillation`.

`GRPOConfig` is reexported from `lilax.trainers.post_training.grpo_config`.

`GRPOTrainer` is reexported from `lilax.trainers.post_training.reinforcement`.

## lilax.benchmarks.cluster

### ClusterConfig

```python
ClusterConfig(*, cluster_name: str, project: str, device_type: str, zone: str = '', location: str | None = None, compute_type: str | None = None, topology: str | None = None) -> None
```

### PathwaysConfig

```python
PathwaysConfig(*, server_image: str | None = None, proxy_server_image: str | None = None, runner_image: str | None = None, colocated_python_sidecar_image: str | None = None, server_flags: str = '', proxy_flags: str = '', worker_flags: str = '', headless: bool = False) -> None
```

### WorkloadConfig

```python
WorkloadConfig(*, model: lilax._backend.benchmarks.cluster_runner.MaxTextModel | None, base_output_directory: str, device_type: str | None = None, num_slices: int | None = None, base_docker_image: str | None = None, num_steps: int = 20, max_restarts: int = 0, priority: str = 'medium', run_name: str | None = None, compute_type: str | None = None, topology: str | None = None, mounts: list[str] = <factory>, pathways_config: lilax._backend.benchmarks.cluster_runner.PathwaysConfig | None = None, hlo_dump: bool = False, skip_validation: bool = False, libtpu_type: str = 'maxtext-docker', libtpu_nightly_version: str = '', generate_metrics_and_upload_to_big_query: bool = False, db_project: str = '', db_dataset: str = '', disruption_configs: list | None = None) -> None
```

### BenchmarkModel

```python
BenchmarkModel(model_name: str, model_type: str, tuning_params: dict, xla_flags: str, pathways_tuning_params: dict | None = None, pathways_xla_flag_options: dict | None = None) -> None
```

### run_cluster_benchmarks

```python
run_cluster_benchmarks(cluster: lilax._backend.benchmarks.cluster_runner.ClusterConfig, workloads: list[lilax._backend.benchmarks.cluster_runner.WorkloadConfig], *, user: str | None = None, disruption_manager=None, wait: bool = False, timeout: str = '-1s') -> list[tuple[str, int]]
```
