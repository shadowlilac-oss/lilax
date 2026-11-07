# Public API migration ledger

This ledger records the agreed argument removals and relocations implemented by the public API update. It distinguishes public removal from backend dependency deletion. Exact YAML section and field names are documented in [execution profiles](EXECUTION_PROFILE.md). The original owner paths below identify the removed Python arguments; they are not new YAML names.

## Reading the paths and destinations

Paths below name the original proposed public owner, not a new YAML path. `TrainingArguments` paths also apply to inherited `SFTConfig`, `DPOConfig`, `DistillationConfig`, and `GRPOConfig` fields. Example locations use stable section titles, not line numbers shifted by deletion. Nested objects are enumerated by leaf once. Removing a container includes the listed leaves, not an unrecorded capability deletion.

| Destination code | Destination category; see EXECUTION_PROFILE.md for exact keys |
|---|---|
| topology | Launcher execution profile, physical mesh/ICI/DCN allocation |
| sharding | Launcher execution profile, logical-axis rules and custom sharding overrides |
| pipeline | Launcher execution profile, detailed pipeline schedule/overlap |
| init | Launcher execution profile, process/controller/device initialization |
| storage | Launcher execution profile, checkpoint format/chunking/I/O/replica storage |
| recovery | Launcher execution profile, emergency storage and deployment/elastic recovery |
| transport | Launcher execution profile, DiLoCo fragmentation/bucketing/network tuning |
| compiler | Launcher execution profile, cache/offline topology/XLA/backend compilation |
| diagnostics | Launcher execution profile, backend dumps/checkify/JAX logs/cloud profiling |
| execution | Launcher execution profile, backend inflight/warmup/implementation tuning |
| tensors | Launcher execution profile, per-tensor rematerialization/offload placement |
| kernels | Launcher execution profile, kernel block sizes/layouts/physical tuning |

PUBLIC_API.md lists the implemented importable signatures. The tables below retain exact old public names and paths without duplicating long values. Original source defaults remain in their linked domain/backend files. A shared nested type's leaves are enumerated once under that original type; removed edges identify every enclosing public owner. Public TrainingArguments/algorithm leaves nested in internalized job specs survive at Trainer.args.

## Training checkpoint storage

Original example location is ALL-ARGS section 1, `SFTTrainer.args=SFTConfig(checkpoint_config=TrainingCheckpointConfig(...))`. Container arguments `TrainingCheckpointConfig.storage` and `.emergency` are relocated with every leaf below. The checkpoint-policy controls, retention, asynchronous saving, resume selection, and output paths stay public.

| Original public owner/path and exact argument | Destination | Reason |
|---|---|---|
| `TrainingCheckpointConfig.enable_single_replica_ckpt_restoring` | storage | Replica restoration layout |
| `TrainingCheckpointConfig.checkpoint_todelete_subdir` | storage | Storage deletion staging |
| `TrainingCheckpointConfig.checkpoint_todelete_full_path` | storage | Storage deletion staging |
| `TrainingCheckpointConfig.colocated_python_checkpointing` | storage | Checkpoint worker placement |
| `TrainingCheckpointConfig.enable_checkpoint_cloud_logger` | diagnostics | Cloud-specific checkpoint logging |
| `TrainingCheckpointConfig.storage.checkpoint_storage_target_data_file_size_bytes` | storage | Storage chunk size |
| `TrainingCheckpointConfig.storage.checkpoint_storage_use_ocdbt` | storage | Storage format |
| `TrainingCheckpointConfig.storage.checkpoint_storage_use_zarr3` | storage | Storage format |
| `TrainingCheckpointConfig.storage.checkpoint_storage_concurrent_gb` | storage | I/O concurrency |
| `TrainingCheckpointConfig.storage.checkpoint_storage_device_host_concurrent_gb` | storage | Device/host I/O concurrency |
| `TrainingCheckpointConfig.emergency.enable_multi_tier_checkpointing` | recovery | Deployment storage tiers |
| `TrainingCheckpointConfig.emergency.local_checkpoint_directory` | recovery | Local tier storage |
| `TrainingCheckpointConfig.emergency.local_checkpoint_period` | recovery | Local tier replication schedule |
| `TrainingCheckpointConfig.emergency.multi_tier_checkpointing_backup_interval_minutes` | recovery | Tier backup schedule |
| `TrainingCheckpointConfig.emergency.multi_tier_checkpointing_backup_interval_steps` | recovery | Tier backup schedule |
| `TrainingCheckpointConfig.emergency.mtc_data_parallelism` | recovery | Storage replication topology |
| `TrainingCheckpointConfig.emergency.enable_emergency_checkpoint` | recovery | Deployment recovery storage |
| `TrainingCheckpointConfig.emergency.use_replicator_service` | recovery | Deployment replication service |
| `TrainingCheckpointConfig.emergency.replicator_backup_interval_minutes` | recovery | Replication schedule |

## Parallelism physical allocation and backend sharding

Original example location is section 1, `SFTConfig.parallelism_config`. `ParallelismConfig.ici`, `.dcn`, and `.pipeline` containers are relocated. `ParallelismConfig`, `MeshConfig` and `ShardingConfig` themselves remain public for the retained common strategy/FSDP controls. High-level strategy is not equated to physical ICI/DCN placement.

| Original public owner/path and exact argument | Destination | Reason |
|---|---|---|
| `ParallelismConfig.mesh.num_slices` | topology | Physical slices |
| `ParallelismConfig.mesh.mesh_axes` | topology | Physical mesh axes |
| `ParallelismConfig.mesh.context_parallel_load_balance` | topology | Backend CP allocation tuning |
| `ParallelismConfig.mesh.context_parallel_reorder_strategy` | topology | Backend CP reorder tuning |
| `ParallelismConfig.mesh.custom_mesh` | topology | Physical mesh override |
| `ParallelismConfig.mesh.custom_mesh_and_rule` | sharding | Custom mesh/rules |
| `ParallelismConfig.mesh.custom_mesh_and_rule_for_eval` | sharding | Eval mesh/rules |
| `ParallelismConfig.mesh.allow_split_physical_axes` | topology | Physical axis splitting |
| `ParallelismConfig.mesh.optimize_mesh_for_tpu_v6e` | topology | Device-specific mesh tuning |
| `ParallelismConfig.ici.ici_diloco_parallelism` | topology | ICI allocation |
| `ParallelismConfig.ici.ici_data_parallelism` | topology | ICI allocation |
| `ParallelismConfig.ici.ici_fsdp_parallelism` | topology | ICI allocation |
| `ParallelismConfig.ici.ici_fsdp_transpose_parallelism` | topology | ICI allocation |
| `ParallelismConfig.ici.ici_sequence_parallelism` | topology | ICI allocation |
| `ParallelismConfig.ici.ici_context_parallelism` | topology | ICI allocation |
| `ParallelismConfig.ici.ici_context_usp_ulysses_parallelism` | topology | ICI allocation |
| `ParallelismConfig.ici.ici_context_autoregressive_parallelism` | topology | ICI allocation |
| `ParallelismConfig.ici.ici_tensor_parallelism` | topology | ICI allocation |
| `ParallelismConfig.ici.ici_tensor_sequence_parallelism` | topology | ICI allocation |
| `ParallelismConfig.ici.ici_autoregressive_parallelism` | topology | ICI allocation |
| `ParallelismConfig.ici.ici_pipeline_parallelism` | topology | ICI allocation |
| `ParallelismConfig.ici.ici_expert_parallelism` | topology | ICI allocation |
| `ParallelismConfig.dcn.dcn_diloco_parallelism` | topology | DCN allocation |
| `ParallelismConfig.dcn.dcn_data_parallelism` | topology | DCN allocation |
| `ParallelismConfig.dcn.dcn_fsdp_parallelism` | topology | DCN allocation |
| `ParallelismConfig.dcn.dcn_fsdp_transpose_parallelism` | topology | DCN allocation |
| `ParallelismConfig.dcn.dcn_sequence_parallelism` | topology | DCN allocation |
| `ParallelismConfig.dcn.dcn_context_parallelism` | topology | DCN allocation |
| `ParallelismConfig.dcn.dcn_context_usp_ulysses_parallelism` | topology | DCN allocation |
| `ParallelismConfig.dcn.dcn_context_autoregressive_parallelism` | topology | DCN allocation |
| `ParallelismConfig.dcn.dcn_tensor_parallelism` | topology | DCN allocation |
| `ParallelismConfig.dcn.dcn_tensor_sequence_parallelism` | topology | DCN allocation |
| `ParallelismConfig.dcn.dcn_pipeline_parallelism` | topology | DCN allocation |
| `ParallelismConfig.dcn.dcn_expert_parallelism` | topology | DCN allocation |
| `ParallelismConfig.dcn.dcn_autoregressive_parallelism` | topology | DCN allocation |
| `ParallelismConfig.pipeline.pipeline_fsdp_ag_per_repeat` | pipeline | Schedule/all-gather tuning |
| `ParallelismConfig.pipeline.pipeline_bsw_no_carry` | pipeline | Backend schedule tuning |
| `ParallelismConfig.pipeline.pipeline_bsw_phase_split` | pipeline | Backend schedule tuning |
| `ParallelismConfig.pipeline.pipeline_bsw_in_compute_dtype` | pipeline | Pipeline buffer policy, not model compute dtype |
| `ParallelismConfig.pipeline.num_layers_per_pipeline_stage` | pipeline | Stage placement |
| `ParallelismConfig.pipeline.num_pipeline_repeats` | pipeline | Schedule repeats |
| `ParallelismConfig.pipeline.pipeline_parallel_layers` | pipeline | Stage placement |
| `ParallelismConfig.pipeline.num_pipeline_microbatches` | pipeline | Pipeline schedule, not train batch size |
| `ParallelismConfig.pipeline.pipeline_delay_activation_forwarding` | pipeline | Overlap timing |
| `ParallelismConfig.pipeline.pipeline_fsdp_ag_once` | pipeline | Schedule/all-gather tuning |
| `ParallelismConfig.pipeline.scan_pipeline_iterations` | pipeline | Backend scan implementation |
| `ParallelismConfig.pipeline.scan_pipeline_repeats` | pipeline | Backend scan implementation |
| `ParallelismConfig.pipeline.scan_layers_per_stage` | pipeline | Backend scan implementation |
| `ParallelismConfig.pipeline.set_remat_policy_on_pipeline_iterations` | tensors | Pipeline remat boundary |
| `ParallelismConfig.pipeline.set_remat_policy_on_layers_per_stage` | tensors | Pipeline remat boundary |
| `ParallelismConfig.sharding.logical_axis_rules` | sharding | Full ordered logical rule list, not separate public arguments per tuple |
| `ParallelismConfig.sharding.logical_axis_rules_for_eval` | sharding | Eval logical rules |
| `ParallelismConfig.sharding.data_sharding` | sharding | Physical data mapping |
| `ParallelismConfig.sharding.context_sharding` | sharding | Physical CP axis |
| `ParallelismConfig.sharding.ulysses_context_sharding` | sharding | Physical CP axis |
| `ParallelismConfig.sharding.input_data_sharding_logical_axes` | sharding | Logical batch mapping |
| `ParallelismConfig.sharding.sharding_tolerance` | sharding | Backend validation tolerance |
| `ParallelismConfig.sharding.check_vma` | diagnostics | Backend mesh assertion |
| `ParallelismConfig.sharding.dense_fsdp_use_two_stage_all_gather` | sharding | Backend collective implementation |
| `ParallelismConfig.sharding.dense_weight_grad_in_kernel_order` | sharding | Kernel buffer layout |

## Initialization and elastic recovery

Original example location is section 1, `SFTConfig.distributed_config` and `.elastic_config`, and section 5, `elastic_checkpoint_strategy`. The `TrainingArguments.elastic_config` container also appeared as explicit `None` in section 2's causal/model-init/DPO/ORPO/distillation/GRPO configs, section 4's adapter config, section 6's environment GRPO config, and section 8's job configs. Every leaf is listed once here.

| Original public owner/path and exact argument | Destination | Reason |
|---|---|---|
| `DistributedConfig.hardware` | init | Launch backend/device choice |
| `DistributedConfig.skip_jax_distributed_system` | init | Process initialization bypass |
| `DistributedConfig.enable_single_controller` | init | Controller arrangement |
| `DistributedConfig.subslice_shape` | init | Physical subslice shape |
| `ElasticConfig.elastic_enabled` | recovery | Deployment recovery strategy |
| `ElasticConfig.elastic_backup_kind` | recovery | Deployment backup strategy |
| `ElasticConfig.elastic_snapshot_interval` | recovery | Recovery snapshot schedule |
| `ElasticConfig.elastic_timeout_seconds` | recovery | Recovery timeout, not ordinary distributed timeout |
| `ElasticConfig.elastic_max_retries` | recovery | Deployment retries |
| `ElasticConfig.elastic_min_slice_count` | recovery | Elastic topology |

## DiLoCo transport details

Original example locations are section 1's `SFTConfig.diloco_config`, section 5's `streaming_diloco_strategy`, and section 8's `PretrainingJobSpec.args.diloco_config`. The `DiLoCoConfig.bandwidth` container is relocated with all leaves below. `enable_diloco`, synchronization period, outer learning rate and momentum stay visible as algorithm controls.

| Original public owner/path and exact argument | Destination | Reason |
|---|---|---|
| `DiLoCoConfig.enable_streaming_diloco` | transport | Transport strategy |
| `DiLoCoConfig.num_diloco_fragments` | transport | Fragmentation |
| `DiLoCoConfig.use_sequential_layers` | transport | Transfer order |
| `DiLoCoConfig.diloco_bucketize_non_scanned` | transport | Bucketing |
| `DiLoCoConfig.num_communication_overlapping_steps` | transport | Communication overlap |
| `DiLoCoConfig.communication_overlapping_alpha` | transport | Overlap policy |
| `DiLoCoConfig.bandwidth.dcn_bandwidth_limit` | transport | Network limit |
| `DiLoCoConfig.bandwidth.dcn_bandwidth_burst` | transport | Network burst |
| `DiLoCoConfig.bandwidth.dcn_bandwidth_latency` | transport | Network latency tuning |
| `DiLoCoConfig.bandwidth.dcn_bandwidth_interface` | transport | Network interface |

## Compiler, execution, and per-tensor recomputation

Original locations are section 1's `SFTConfig.compilation_config`, `.execution_config`, and `.remat_config`, plus section 5's `compile_train_step`, `compile_engine`, and `is_oom` config constructors. `TrainingArguments.execution_config` also appeared as `None` in section 2's trainer configs, section 4's adapter config, section 6's environment config, and section 8's job configs. `CompilationConfig.hlo_dump` and `TrainExecutionConfig.diagnostics` containers are relocated with all leaves below.

| Original public owner/path and exact argument | Destination | Reason |
|---|---|---|
| `CompilationConfig.compiled_trainstep_file` | compiler | Offline compiled artifact |
| `CompilationConfig.compile_topology` | compiler | Offline target topology |
| `CompilationConfig.compile_topology_num_slices` | compiler | Offline slices |
| `CompilationConfig.internal_compile_num_devices` | compiler | Compiler target device count |
| `CompilationConfig.compile_xla_flags` | compiler | XLA flags |
| `CompilationConfig.jax_cache_dir` | compiler | Cache storage |
| `CompilationConfig.shardy` | compiler | Backend partitioner |
| `CompilationConfig.prng_impl` | compiler | Backend PRNG implementation, unchanged when profile resolves it |
| `CompilationConfig.remove_size_one_mesh_axis_from_type` | compiler | Compiler mesh representation |
| `CompilationConfig.hlo_dump.dump_hlo` | diagnostics | Backend IR dump |
| `CompilationConfig.hlo_dump.dump_step` | diagnostics | Dump timing |
| `CompilationConfig.hlo_dump.dump_hlo_local_dir` | diagnostics | Dump storage |
| `CompilationConfig.hlo_dump.dump_hlo_delete_local_after` | diagnostics | Dump cleanup |
| `CompilationConfig.hlo_dump.dump_hlo_gcs_dir` | diagnostics | Dump cloud storage |
| `CompilationConfig.hlo_dump.dump_hlo_module_name` | diagnostics | Dump module filter |
| `CompilationConfig.hlo_dump.dump_hlo_local_module_name` | diagnostics | Dump module filter |
| `CompilationConfig.hlo_dump.dump_hlo_xla_flags` | diagnostics | Dump flags |
| `CompilationConfig.hlo_dump.dump_hlo_upload_all` | diagnostics | Dump upload policy |
| `CompilationConfig.hlo_dump.dump_jaxpr` | diagnostics | Backend IR dump |
| `CompilationConfig.hlo_dump.dump_jaxpr_local_dir` | diagnostics | Dump storage |
| `CompilationConfig.hlo_dump.dump_jaxpr_delete_local_after` | diagnostics | Dump cleanup |
| `CompilationConfig.hlo_dump.dump_jaxpr_gcs_dir` | diagnostics | Dump cloud storage |
| `TrainExecutionConfig.use_tunix_gradient_accumulation` | execution | Implementation selector, public accumulation steps retained |
| `TrainExecutionConfig.max_inflight_computations` | execution | Backend queue limit |
| `TrainExecutionConfig.block_state_before_run_start` | execution | Backend warmup barrier |
| `TrainExecutionConfig.warm_input_reshard_before_run_start` | execution | Backend warmup |
| `TrainExecutionConfig.warm_eval_input_reshard_before_run_start` | execution | Backend warmup |
| `TrainExecutionConfig.diagnostics.constant_bound_config` | diagnostics | Backend bound assertions |
| `TrainExecutionConfig.diagnostics.jax_debug_log_modules` | diagnostics | JAX module logs |
| `TrainExecutionConfig.diagnostics.max_checkify` | diagnostics | JAX checkify machinery, nonfinite loss callback retained |
| `RematConfig.decoder_layer_input` | tensors | Per-tensor placement |
| `RematConfig.indexer_cutoff_threshold` | tensors | Per-tensor placement |
| `RematConfig.context` | tensors | Per-tensor placement |
| `RematConfig.mlpwi` | tensors | Per-tensor placement |
| `RematConfig.mlpwi_0` | tensors | Per-tensor placement |
| `RematConfig.mlpwi_1` | tensors | Per-tensor placement |
| `RematConfig.moe_x_sorted` | tensors | Per-tensor placement |
| `RematConfig.moe_routing_maps` | tensors | Per-tensor placement |
| `RematConfig.moe_tc_routing_checkpoint` | tensors | Per-tensor placement |
| `RematConfig.mlpwo` | tensors | Per-tensor placement |
| `RematConfig.te_moe_dispatch_remat_location` | tensors | Per-tensor placement |
| `RematConfig.moe_mlpwi_0` | tensors | Per-tensor placement |
| `RematConfig.moe_mlpwi_1` | tensors | Per-tensor placement |
| `RematConfig.moe_mlpwo` | tensors | Per-tensor placement |
| `RematConfig.query_proj` | tensors | Per-tensor placement |
| `RematConfig.key_proj` | tensors | Per-tensor placement |
| `RematConfig.value_proj` | tensors | Per-tensor placement |
| `RematConfig.kv_proj` | tensors | Per-tensor placement |
| `RematConfig.query_wa_proj` | tensors | Per-tensor placement |
| `RematConfig.kv_wa_proj` | tensors | Per-tensor placement |
| `RematConfig.qkv_proj` | tensors | Per-tensor placement |
| `RematConfig.out_proj` | tensors | Per-tensor placement |
| `RematConfig.mla_q` | tensors | Per-tensor placement |
| `RematConfig.mla_kv` | tensors | Per-tensor placement |
| `RematConfig.attention_out` | tensors | Per-tensor placement |
| `RematConfig.engram` | tensors | Per-tensor placement |

## Model kernel tuning and quantization transport

Original example location is section 4's `reference_config=MaxTextConfig(...)` and its pretrained quantization alternative. These are removals from the example's public constructor view, not deletions from legacy backend source. The model architecture, compute precision, quantization recipes/calibration/exclusions, attention implementation selection, masking, and FP8 query/key controls stay visible. Section 9's unchanged typed source-family inventory is reference evidence, not an execution-profile-free training call.

| Original public owner/path and exact argument | Destination | Reason |
|---|---|---|
| `MaxTextConfig.kernel_config.ragged_block_size` | kernels | Kernel block size |
| `MaxTextConfig.kernel_config.force_q_layout` | kernels | Kernel layout |
| `MaxTextConfig.layout_config.scan_layers` | execution | Backend scan implementation |
| `MaxTextConfig.layout_config.param_scan_axis` | execution | Backend parameter scan layout |
| `MaxTextConfig.kernel_config.splash.sa_block_q` | kernels | Kernel block size |
| `MaxTextConfig.kernel_config.splash.sa_block_kv` | kernels | Kernel block size |
| `MaxTextConfig.kernel_config.splash.sa_block_kv_compute` | kernels | Kernel block size |
| `MaxTextConfig.kernel_config.splash.eval_sa_block_q` | kernels | Eval block size |
| `MaxTextConfig.kernel_config.splash.eval_sa_block_kv` | kernels | Eval block size |
| `MaxTextConfig.kernel_config.splash.eval_sa_block_kv_compute` | kernels | Eval block size |
| `MaxTextConfig.kernel_config.splash.sa_block_q_dkv` | kernels | Backward block size |
| `MaxTextConfig.kernel_config.splash.sa_block_kv_dkv` | kernels | Backward block size |
| `MaxTextConfig.kernel_config.splash.sa_block_kv_dkv_compute` | kernels | Backward block size |
| `MaxTextConfig.kernel_config.splash.sa_block_q_dq` | kernels | Backward block size |
| `MaxTextConfig.kernel_config.splash.sa_block_kv_dq` | kernels | Backward block size |
| `MaxTextConfig.kernel_config.splash.sa_use_fused_bwd_kernel` | kernels | Backend backward fusion |
| `MaxTextConfig.kernel_config.splash.sa_bwd_dkv_megacore` | kernels | Device-specific kernel schedule |
| `MaxTextConfig.kernel_config.splash.sa_q_layout` | kernels | Tensor layout |
| `MaxTextConfig.kernel_config.splash.sa_k_layout` | kernels | Tensor layout |
| `MaxTextConfig.kernel_config.splash.sa_v_layout` | kernels | Tensor layout |
| `MaxTextConfig.kernel_config.splash.eval_sa_q_layout` | kernels | Eval tensor layout |
| `MaxTextConfig.kernel_config.splash.eval_sa_k_layout` | kernels | Eval tensor layout |
| `MaxTextConfig.kernel_config.splash.eval_sa_v_layout` | kernels | Eval tensor layout |
| `MaxTextConfig.kernel_config.splash.use_splash_scheduler` | kernels | Kernel schedule |
| `MaxTextConfig.kernel_config.splash.ring_scan_unroll` | kernels | Backend scan unroll |
| `MaxTextConfig.kernel_config.splash.sa_fuse_reciprocal` | kernels | Backend fusion, existing math preserved |
| `MaxTextConfig.kernel_config.splash.sa_use_base2_exp` | kernels | Backend exponential implementation, existing math preserved |
| `MaxTextConfig.kernel_config.splash.sa_qk_diag_skip` | kernels | Kernel diagonal schedule |
| `MaxTextConfig.kernel_config.splash.sa_sv_diag_skip` | kernels | Kernel diagonal schedule |
| `MaxTextConfig.kernel_config.splash.sa_qk_diag_grid` | kernels | Kernel grid |
| `MaxTextConfig.kernel_config.splash.local_sa_block_q` | kernels | Local-attention block size |
| `MaxTextConfig.kernel_config.splash.local_sa_block_kv` | kernels | Local-attention block size |
| `MaxTextConfig.kernel_config.splash.local_sa_block_kv_compute` | kernels | Local-attention block size |
| `MaxTextConfig.kernel_config.splash.eval_local_sa_block_q` | kernels | Eval local-attention block size |
| `MaxTextConfig.kernel_config.splash.eval_local_sa_block_kv` | kernels | Eval local-attention block size |
| `MaxTextConfig.kernel_config.splash.eval_local_sa_block_kv_compute` | kernels | Eval local-attention block size |
| `MaxTextConfig.kernel_config.splash.local_sa_block_q_dkv` | kernels | Local backward block size |
| `MaxTextConfig.kernel_config.splash.local_sa_block_kv_dkv` | kernels | Local backward block size |
| `MaxTextConfig.kernel_config.splash.local_sa_block_kv_dkv_compute` | kernels | Local backward block size |
| `MaxTextConfig.kernel_config.splash.local_sa_block_q_dq` | kernels | Local backward block size |
| `MaxTextConfig.kernel_config.splash.local_sa_block_kv_dq` | kernels | Local backward block size |
| `MaxTextConfig.kernel_config.splash.local_sa_use_fused_bwd_kernel` | kernels | Local backward fusion |
| `MaxTextConfig.kernel_config.splash.local_sa_q_layout` | kernels | Local tensor layout |
| `MaxTextConfig.kernel_config.splash.local_sa_k_layout` | kernels | Local tensor layout |
| `MaxTextConfig.kernel_config.splash.local_sa_v_layout` | kernels | Local tensor layout |
| `MaxTextConfig.kernel_config.splash.eval_local_sa_q_layout` | kernels | Eval local tensor layout |
| `MaxTextConfig.kernel_config.splash.eval_local_sa_k_layout` | kernels | Eval local tensor layout |
| `MaxTextConfig.kernel_config.splash.eval_local_sa_v_layout` | kernels | Eval local tensor layout |
| `MaxTextConfig.kernel_config.splash.local_use_splash_scheduler` | kernels | Local schedule |
| `MaxTextConfig.kernel_config.splash.local_sa_fuse_reciprocal` | kernels | Local fusion, existing math preserved |
| `MaxTextConfig.kernel_config.splash.local_sa_use_base2_exp` | kernels | Local exponential implementation, existing math preserved |
| `MaxTextConfig.kernel_config.splash.sa_dkv_reuse_scaled_q` | kernels | Kernel intermediate reuse |
| `MaxTextConfig.kernel_config.splash.sa_q_prescaled_log2e` | kernels | Kernel input representation |
| `MaxTextConfig.kernel_config.splash.use_max_logit_estimate` | kernels | Kernel cost estimate |
| `MaxTextConfig.kernel_config.splash.cost_estimate_flops_fwd` | kernels | Kernel cost estimate |
| `MaxTextConfig.kernel_config.splash.cost_estimate_flops_bwd` | kernels | Kernel cost estimate |
| `MaxTextConfig.kernel_config.splash.dq_reduction_steps` | kernels | Backend reduction schedule |
| `QuantizationConfig.replicate_quant_scale` | sharding | Quantization scale placement, not format/calibration |
| `QuantizationConfig.te_comm_gemm_overlap` | transport | Collective/GEMM overlap, not TE compute recipe |

## Callback backend diagnostics

Original example location is section 5's `callback_choices`, passed at a trainer constructor's `callbacks` argument. Whole `GoodputCallback.config`, `ManagedDiagnosticsCallback.config`, and `GcpMonitoringCallback.config` objects are relocated with every leaf below. The `XProfCallback` deployment-server constructor is also removed. Ordinary logging, nonfinite-loss guards, target-loss behavior, basic profiler scheduling, MLPerf logging, callback lifecycle and event ownership remain public.

| Original public owner/path and exact argument | Destination | Reason |
|---|---|---|
| `MetricsCallback.record_internal_nn_metrics` | diagnostics | Internal backend metrics |
| `MetricsCallback.prometheus_port` | diagnostics | Metrics server deployment |
| `ProfilerCallback.config.upload_all_profiler_results` | diagnostics | Profile upload policy |
| `ProfilerCallback.config.enable_jax_profiler` | diagnostics | Backend profiler server |
| `ProfilerCallback.config.jax_profiler_port` | diagnostics | Backend profiler port |
| `ProfilerCallback.config.enable_tpu_profiling_options` | diagnostics | TPU profiler machinery |
| `ProfilerCallback.config.tpu_num_chips_to_profile_per_task` | diagnostics | Profiler physical allocation |
| `ProfilerCallback.config.tpu_num_sparse_cores_to_trace` | diagnostics | Sparse-core tracing |
| `ProfilerCallback.config.tpu_num_sparse_core_tiles_to_trace` | diagnostics | Sparse-core tracing |
| `ProfilerCallback.config.xprof_tpu_power_trace_level` | diagnostics | Power tracing |
| `ProfilerCallback.config.xprof_e2e_enable_fw_throttle_event` | diagnostics | Firmware tracing |
| `ProfilerCallback.config.xprof_e2e_enable_fw_power_level_event` | diagnostics | Firmware tracing |
| `ProfilerCallback.config.xprof_e2e_enable_fw_thermal_event` | diagnostics | Firmware tracing |
| `ProfilerCallback.config.profile_power_events` | diagnostics | Power tracing |
| `GoodputCallback.config.enable_goodput_recording` | diagnostics | Deployment goodput instrumentation |
| `GoodputCallback.config.goodput_job_name` | diagnostics | Deployment identity |
| `GoodputCallback.config.monitor_goodput` | diagnostics | Deployment monitoring |
| `GoodputCallback.config.goodput_upload_interval_seconds` | diagnostics | Cloud upload schedule |
| `GoodputCallback.config.enable_pathways_goodput` | diagnostics | Pathways instrumentation |
| `GoodputCallback.config.monitor_step_time_deviation` | diagnostics | Deployment monitoring |
| `GoodputCallback.config.step_deviation_interval_seconds` | diagnostics | Monitoring schedule |
| `GoodputCallback.config.enable_gcp_goodput_metrics` | diagnostics | Cloud metrics |
| `GoodputCallback.config.enable_gcp_step_deviation_metrics` | diagnostics | Cloud metrics |
| `GoodputCallback.config.rolling_windows_seconds` | diagnostics | Monitoring windows |
| `GoodputCallback.config.enable_rolling_window_goodput` | diagnostics | Deployment monitoring |
| `ManagedDiagnosticsCallback.config.managed_mldiagnostics` | diagnostics | Managed diagnostics service |
| `ManagedDiagnosticsCallback.config.managed_mldiagnostics_on_demand_profiling` | diagnostics | Managed profile scheduling |
| `ManagedDiagnosticsCallback.config.managed_mldiagnostics_run_group` | diagnostics | Cloud identity |
| `ManagedDiagnosticsCallback.config.managed_mldiagnostics_region` | diagnostics | Cloud region |
| `ManagedDiagnosticsCallback.config.managed_mldiagnostics_storage_path` | diagnostics | Cloud storage |
| `ManagedDiagnosticsCallback.config.managed_mldiagnostics_metric_only_run` | diagnostics | Managed service mode |
| `GcpMonitoringCallback.config.report_heartbeat_metric_for_gcp_monitoring` | diagnostics | Deployment heartbeat |
| `GcpMonitoringCallback.config.heartbeat_reporting_interval_in_seconds` | diagnostics | Heartbeat schedule |
| `GcpMonitoringCallback.config.report_performance_metric_for_gcp_monitoring` | diagnostics | Cloud performance metrics |
| `XProfCallback.launch_server` | diagnostics | Server deployment |
| `XProfCallback.port` | diagnostics | Server port |
| `RLProfilerCallback.max_num_hosts` | diagnostics | Physical profiler allocation |

The same profile-only JAX profiler server controls also appeared in section 7's separate JetStream constructor alternatives. Ordinary serving/cache/batching/model/evaluation arguments remain public.

| Original public owner/path and exact argument | Example location | Destination | Reason |
|---|---|---|---|
| `JetStreamConfig.enable_jax_profiler` | Section 7, both JetStream alternatives | diagnostics | Backend profiler server |
| `JetStreamConfig.jax_profiler_port` | Section 7, both JetStream alternatives | diagnostics | Backend profiler port |

## Retained controls and status

Compute/gradient/weight dtype, matmul precision, supported FP8/NVFP4 recipes, calibration/exclusions, gradient checkpointing/basic remat/host offload, optimizer/algorithm controls, common parallelism/FSDP/timeout, checkpoint policy, PEFT, generation and serving remain public. Jobs and compiler internals do not remain public merely because this earlier phase retained them. The later rows below supersede that earlier disposition. No supported common compile option is deliberately removed; the former internal_compile flag was never generic compile enablement.

Backend implementations remain available internally. `gradient_checkpointing` maps to the existing remat policy. Exact execution-profile sections and source field names are documented in EXECUTION_PROFILE.md. Public model configuration composes the focused model domains listed in PUBLIC_API.md.

## Later agreed wrapper and processing corrections

Locations refer to the original accepted design's common/data declarations and examples. Public removal does not delete backend implementations. Processing settings now belong to [AutoTokenizer and AutoProcessor](src/lilax/processing.py).

| Exact original public symbol/path | Original example location | Disposition/destination | Reason |
|---|---|---|---|
| `TrainingArguments.evaluation_config` | 1/2, inherited algorithm args | Removed public wrapper edge; direct eval_on_start | No evaluation config bag |
| `EvaluationConfig` | 1, nested SFT args | Removed public container | Unnecessary wrapper |
| `EvaluationConfig.start_step` | 1 | Removed numeric threshold; direct TrainingArguments.eval_on_start=False | HF-named boolean, not an equivalent arbitrary start-step threshold |
| `EvaluationConfig.num_batches` | 1 | Removed public knob; supplied eval dataset controls extent | No hardcoded eval batch cap |
| `GRPOEvaluationConfig.num_test_batches` | 2/6 | Removed public cap; supplied eval dataset | Same no-cap decision; other specialized RL eval options survive |
| `TrainingArguments.distributed_config` | 1/2 | Removed edge; timeout direct on TrainingArguments | Single-field wrapper after profile split |
| `DistributedConfig` | 1 | Removed public container | Other leaves already profile-only above |
| `DistributedConfig.jax_distributed_initialization_timeout` | 1 | TrainingArguments.jax_distributed_initialization_timeout | Retained public, flattened |
| `JetStreamConfig.distributed_config` | 7 | Removed edge; direct timeout | Same removed wrapper |
| `VLLMConfig.distributed_config` | 7 | Removed edge; direct timeout | Same removed wrapper |
| `TrainingArguments.compilation_config` | 1/2 | Removed public edge | No generic compile bag |
| `CompilationConfig` | 1/5 | Internal/profile-only container | No surviving generic enable switch |
| `CompilationConfig.internal_compile` | 1/5 | Internal offline compiler topology-discovery mechanic | train_compile.get_topology_mesh uses topology discovery, not enable-JIT |
| `prepare_dataset` | 2/3 | Internal-only trainer preparation | Trainer receives processing_class |
| `prepare_dataset.dataset` | 2/3 | Trainer train_dataset/eval_dataset | Direct dataset ownership |
| `prepare_dataset.processing_class` | 2/3 | Trainer.processing_class | Single processing-class selection |
| `prepare_dataset.options` | 2/3 | Removed edge | No redundant processing config |
| `prepare_dataset.formatting_func` | 2/3 | SFTTrainer.formatting_func or dataset transform | Formatting is dataset/trainer-owned |
| `TextProcessingOptions` | 2/3/8 | Removed public container | Settings assigned below |
| `TextProcessingOptions.columns` | 2/3/8 | Dataset selection/transform, SFTConfig.dataset_text_field | Not tokenizer vocabulary or tokenization |
| `TextProcessingOptions.tokenize` | 2/3/8 | Lilax AutoTokenizer/AutoProcessor.from_pretrained.tokenize | Custom tokenizer/processor setting retained |
| `TextProcessingOptions.use_chat_template` | 2/3/8 | Lilax AutoTokenizer/AutoProcessor.from_pretrained.use_chat_template | HF tokenizer/processor owns template application |
| `TextProcessingOptions.use_truncation` | 2/3/8 | Lilax AutoTokenizer/AutoProcessor.from_pretrained.use_truncation | Tokenizer/processor setting retained |
| `TextProcessingOptions.default_prompt` | 2/3/8 | Intentionally removed public knob; application records | No hidden default prompt |
| `TextProcessingOptions.chat_template_path` | 2/3/8 | Intentionally removed public knob; HF tokenizer chat_template | No competing file selector |
| `TextProcessingOptions.data_template_path` | 2/3/8 | Intentionally removed public knob; dataset records/transform | No hidden data-template file |
| `TextProcessingOptions.multimodal` | 3/8 | Removed edge; Lilax AutoProcessor owns leaves below | No redundant processor config |
| `MultimodalProcessingOptions` | 3 | Removed public container | Fields move directly to AutoProcessor |
| `TextProcessingOptions.multimodal.image_column` | 3 | AutoProcessor.from_pretrained.image_column | Processor-owned |
| `TextProcessingOptions.multimodal.image_placeholder` | 3 | AutoProcessor.from_pretrained.image_placeholder | Processor-owned |
| `TextProcessingOptions.multimodal.video_placeholder` | 3 | AutoProcessor.from_pretrained.video_placeholder | Processor-owned |
| `TextProcessingOptions.multimodal.max_num_images_per_example` | 3 | AutoProcessor.from_pretrained.max_num_images_per_example | Processor-owned |
| `RLDataOptions.data_template_path` | 2/3 | Intentionally removed public knob; dataset transform | Same template-path removal |

## OLMo and Grain public removal

The user's explicit removal supersedes the earlier retained-OLMo/Grain disposition. Every row below is removed from the public API. Backend implementations and dependencies remain internal. Ordinary datasets use HF datasets; the HF pipeline retains tokenization/chunking, packing, masks and multimodal preprocessing through its existing transforms.

Locations name the pre-removal document sections. API.md data declarations and training-args.md data declarations contain the original finite signatures/defaults. ALL-ARGS section 3 supplied the nested constructor example. Backend evidence remains [data_datasets.py](src/lilax/_backend/configs/inert_datasets.py), [olmo_grain_data_processing.py](src/lilax/_backend/input_pipeline/olmo/olmo_grain_data_processing.py) and [olmo_data_grain.py](src/lilax/_backend/input_pipeline/olmo/olmo_data_grain.py). These evidence pointers do not restore public exports or assert backend deletion.

| Exact removed symbol/owner path | Original declaration/example location | Explicit disposition and reason |
|---|---|---|
| `load_olmo_dataset` | API data arguments; training-args data arguments; ALL-ARGS 3 call; EXAMPLES 2 source-reference prose | Public function removed by explicit user decision; no public replacement |
| `load_olmo_dataset.index_path` | API/training-args signature; ALL-ARGS 3 loader call | Public index selector removed with function |
| `load_olmo_dataset.path_remap_from` | API/training-args signature; ALL-ARGS 3 loader call | Public source-path remapping arg removed |
| `load_olmo_dataset.path_remap_to` | API/training-args signature; ALL-ARGS 3 loader call | Public destination-path remapping arg removed |
| `load_olmo_dataset.apply_ngram_filter` | API/training-args signature; ALL-ARGS 3 loader call | Public specialized filtering arg removed |
| `load_olmo_dataset.options` | API/training-args signature; ALL-ARGS 3 nested GrainOptions call | Public config edge removed; leaves enumerated once below |
| `OLMoDataset` | API shared imports, DatasetLike and descriptor prose; load_olmo_dataset return annotation | Public descriptor/type removed; no undeclared constructor args invented |
| `DatasetLike` union member `OLMoDataset` | API shared types; training-args shared-type prose | Removed from accepted public dataset union, including downstream EvalDataset composition |
| `load_olmo_dataset` return type `OLMoDataset` | API data signature; training-args returned DatasetLike; ALL-ARGS 3 binding | Removed return contract with the removed function/type |
| `GrainOptions` | API/training-args data declarations; ALL-ARGS 3 import/nested constructor | Entire public options container removed; no surviving OLMo exception |
| `GrainOptions.worker_count` | API/training-args GrainOptions declaration; ALL-ARGS 3 nested options | Public Grain worker-count arg removed |
| `GrainOptions.per_worker_buffer_size` | API/training-args GrainOptions declaration; ALL-ARGS 3 nested options | Public Grain worker-buffer arg removed |
| `GrainOptions.ram_budget_mb` | API/training-args GrainOptions declaration; ALL-ARGS 3 nested options | Public Grain memory-budget arg removed |
| `GrainOptions.num_threads` | API/training-args GrainOptions declaration; ALL-ARGS 3 nested options | Public Grain thread arg removed |
| `GrainOptions.prefetch_buffer_size` | API/training-args GrainOptions declaration; ALL-ARGS 3 nested options | Public Grain prefetch arg removed |
| `GrainOptions.data_source_max_workers` | API/training-args GrainOptions declaration; ALL-ARGS 3 nested options | Public Grain source-worker arg removed |
| `GrainOptions.index_storage_option` | API/training-args GrainOptions declaration; ALL-ARGS 3 nested options and alternatives prose | Public Grain index-storage arg and value alternatives removed |
| `GrainOptions.shuffle_buffer_size` | API/training-args GrainOptions declaration; ALL-ARGS 3 nested options | Public Grain shuffle-buffer arg removed; ordinary HF shuffle remains |
| `GrainOptions.elastic` | API/training-args GrainOptions declaration; ALL-ARGS 3 nested options | Public Grain elastic-input arg removed |
| `GrainOptions.packing_type` | API/training-args GrainOptions declaration; ALL-ARGS 3 nested options and alternatives prose | Public Grain packing arg and value alternatives removed; trainer/HF data responsibilities unchanged |
| `GrainOptions.max_segments_per_seq` | API/training-args GrainOptions declaration; ALL-ARGS 3 nested options | Public Grain leaf removed; distinct surviving SyntheticDataset.max_segments_per_seq is unchanged |
| `lilax.data.load_olmo_dataset` import/export | ALL-ARGS 3 from lilax.data import | Removed proposed public loader import/export |
| `lilax.data.GrainOptions` import/export | ALL-ARGS 3 from lilax.data import | Removed proposed public options import/export |
| `lilax.data.OLMoDataset` import/export | API shared imports | Removed proposed public descriptor import/export |
| `olmo_train` binding and `load_olmo_dataset` example call | ALL-ARGS 3 source alternatives; EXAMPLES 2 source-reference prose; ALL-ARGS 10 coverage row | Entire specialized example removed, not replaced with a renamed native alternative |

## Native data and tokenizer public exports

Original locations are ALL-ARGS section 3, EXAMPLES sections 2/12 and API/training data declarations. Existing native implementations remain required by internal pipelines. HF datasets/tokenizers are the ordinary user workflow. No new native public mode or full feature-preservation proof is claimed. The formerly retained OLMo/Grain exception is superseded by the explicit public removals immediately above, including every GrainOptions leaf. Original field evidence is in the linked source domains in training-args.md and FIELDS.md.

| Exact original public symbol/path | Disposition/destination | Reason |
|---|---|---|
| `load_grain_dataset` | Public export removed; backend internal-only | Ordinary HF dataset workflow |
| `load_grain_dataset.files` | Internal Grain loader | Native input selector |
| `load_grain_dataset.file_type` | Internal Grain loader | Native format selector |
| `load_grain_dataset.mixture_config_path` | Internal Grain loader | Native mixture implementation |
| `load_grain_dataset.options` | Internal-only loader edge; public GrainOptions container also removed | No public native loader or retained OLMo exception |
| `load_grain_dataset.mmap` | Internal loader edge | Native format implementation |
| `load_grain_dataset.input_options` | Internal loader edge | Backend data execution |
| `MMapOptions` | Internal-only container | No remaining public MMap loader |
| `MMapOptions.eod_id` | Internal MMap pipeline | Existing vocabulary behavior retained internally |
| `MMapOptions.blend_cache_dir` | Internal MMap pipeline | Native cache |
| `MMapOptions.blend_index_dir` | Internal MMap pipeline | Native index |
| `MMapOptions.reset_attention_mask` | Internal MMap pipeline | No algorithm rewrite |
| `MMapOptions.eod_mask_loss` | Internal MMap pipeline | No loss rewrite |
| `MMapOptions.packing_max_segments_per_sample` | Internal MMap pipeline | Native packing |
| `MMapOptions.split_sentences` | Internal MMap pipeline | Native sentence splitting |
| `MMapOptions.npy_split` | Internal MMap pipeline | Native split selector |
| `InputPipelineOptions` | Internal-only container | Backend input execution |
| `InputPipelineOptions.colocated_python_data_input` | `execution.colocated_python_data_input` | Worker placement |
| `InputPipelineOptions.expansion_factor_real_data` | Internal input pipeline | Existing expansion behavior |
| `InputPipelineOptions.generate_padding_batch` | Internal input pipeline | Existing padding behavior |
| `load_tfds_dataset` | Public export removed; backend internal-only | Ordinary HF dataset workflow |
| `load_tfds_dataset.name` | Internal TFDS loader | Native source selector |
| `load_tfds_dataset.data_dir` | Internal TFDS loader | Native storage selector |
| `load_tfds_dataset.split` | Internal TFDS loader | Native split selector |
| `load_tfds_dataset.options` | Internal InputPipelineOptions edge | Leaves enumerated above |
| `load_tfds_dataset.c4` | Internal C4Options edge | Leaves enumerated below |
| `C4Options` | Internal-only container | No remaining public TFDS loader |
| `C4Options.use_stream_chunking` | Internal C4 pipeline | Existing chunking retained internally |
| `C4Options.train_shard_in_read` | Internal C4 pipeline | Existing read-sharding behavior |
| `C4Options.train_interleave_cycle_length` | Internal C4 pipeline | Existing interleave behavior |
| `C4Options.eval_cache_prefill_in_background` | Internal C4 pipeline | Existing eval prefetch |
| `SentencePieceTokenizer` | Public export removed; backend internal-only | No native public tokenizer alternative |
| `SentencePieceTokenizer.vocab_file` | Internal tokenizer implementation | Existing assets remain usable internally |
| `SentencePieceTokenizer.add_bos_token` | Internal tokenizer implementation | Preserve native special-token behavior |
| `SentencePieceTokenizer.add_eos_token` | Internal tokenizer implementation | Preserve native special-token behavior |
| `SentencePieceTokenizer.encode` | Internal tokenizer method | No public native method |
| `SentencePieceTokenizer.encode.text` | Internal tokenizer method | Existing encoder input |
| `SentencePieceTokenizer.decode` | Internal tokenizer method | No public native method |
| `SentencePieceTokenizer.decode.token_ids` | Internal tokenizer method | Existing decoder input |
| `TikTokenTokenizer` | Public export removed; backend internal-only | No native public tokenizer alternative |
| `TikTokenTokenizer.vocab_file` | Internal tokenizer implementation | Existing assets retained internally |
| `TikTokenTokenizer.add_bos_token` | Internal tokenizer implementation | Existing BOS behavior |
| `TikTokenTokenizer.add_eos_token` | Internal tokenizer implementation | Existing EOS behavior |
| `TikTokenTokenizer.encode` | Internal tokenizer method | No public native method |
| `TikTokenTokenizer.encode.text` | Internal tokenizer method | Existing encoder input |
| `TikTokenTokenizer.encode.allowed_special` | Internal tokenizer method | Existing special-token policy |
| `TikTokenTokenizer.encode.disallowed_special` | Internal tokenizer method | Existing special-token policy |
| `TikTokenTokenizer.decode` | Internal tokenizer method | No public native method |
| `TikTokenTokenizer.decode.token_ids` | Internal tokenizer method | Existing decoder input |

## Quantization owner split and loader internals

Original location is ALL-ARGS section 4, API model declarations and model-args.md. Source evidence is [model_quantization.py](src/lilax/configuration/domains/model_quantization.py), [qat_application.py](src/lilax/modeling/qat_application.py), [qwix_model.py](src/lilax/_backend/layers/quantization/qwix_model.py) and [models.py](src/lilax/modeling/models.py). Both graph-recipe and explicit QAT paths can transform the model with qwix.quantize_model. Source mutual exclusion remains. The rejected intermediate ComputeRecipeConfig/QwixRecipeConfig inheritance is superseded by independent focused types; exact intermediate paths are recorded below. These are public ownership corrections, not backend deletions, numerical changes, profile relocations or implemented adapters.

| Exact old symbol/owner path | New public owner or internal disposition | Reason |
|---|---|---|
| `QuantizationConfig` | Removed universal public container | Separate compute, Qwix rules, KV cache and sparsity |
| `MaxTextConfig.quantization_config` union member `QuantizationConfig` | QuantizationRecipe union in FIELDS.md | One focused recipe-or-QAT entry |
| `AutoModelForCausalLM.from_pretrained.quantization_config` union member `QuantizationConfig` | QuantizationRecipe union in FIELDS.md | Loader argument itself survives |
| `AutoModelForCausalLM.from_config.quantization_config` union member `QuantizationConfig` | QuantizationRecipe union in FIELDS.md | From-scratch reference only |
| `QuantizationConfig.quantization` | ComputeRecipeConfig.quantization, QwixRecipeConfig.quantization, JaxMXConfig.quantization; fixed recipe identity on other focused types | Existing dispatch separated; supported FP8/NVFP4 retained |
| `QuantizationConfig.jax_mx_gradient_mode` | JaxMXConfig.jax_mx_gradient_mode | JAX MX only |
| `QuantizationConfig.unquantized_modules` | PrecisionConfig.unquantized_modules | Stored dtype/native get_quant exclusions, not generic Qwix rule filtering |
| `QuantizationConfig.weight_block_size` | PrecisionConfig.weight_block_size | Stored-weight scaling metadata |
| `QuantizationConfig.quant_cfg_path` | AqtMixedPrecisionConfig.quant_cfg_path | intmp JSON only |
| `QuantizationConfig.quantize_mtp` | QwixFP8FullConfig.quantize_mtp | Existing fp8_full/MTP restriction |
| `QuantizationConfig.quantize_router_proj` | QwixRecipeConfig.quantize_router_proj, QwixFP8FullConfig.quantize_router_proj | Actual graph-Qwix rules only |
| `QuantizationConfig.quantize_logits_proj` | QwixFP8FullConfig.quantize_logits_proj | Existing fp8_full/logits/tiling restriction |
| `QuantizationConfig.logits_proj_quant_calibration_method` | QwixFP8FullConfig.logits_proj_quant_calibration_method | FP8_FULL logits rule only |
| `QuantizationConfig.logits_proj_bwd_quant_calibration_method` | QwixFP8FullConfig.logits_proj_bwd_quant_calibration_method | FP8_FULL logits backward rule only |
| `QuantizationConfig.sa_qproj_bwd_bf16` | QwixFP8FullConfig.sa_qproj_bwd_bf16 | FP8_FULL query backward rule only |
| `QuantizationConfig.quantization_local_shard_count` | AqtInt8Config.quantization_local_shard_count | AQT-int8 numerical scale semantics |
| `QuantizationConfig.qwix_bwd_weight_grad_tile_count` | QwixRecipeConfig.qwix_bwd_weight_grad_tile_count | Uniform Qwix only; FP8_FULL ignores it |
| `QuantizationConfig.use_qwix_quantization` | Global selector removed; graph recipe types and BatchSplitQwixConfig fix True; BatchSplitManualConfig.use_qwix_quantization retains existing GMM selection | No competing graph-transform selectors; manual GMM is not graph interception |
| `QuantizationConfig.use_manual_quantization` | Public boolean removed; BatchSplitQwixConfig fixes False, BatchSplitManualConfig fixes True | Existing batch-split numerical dispatch, not automatic schedule selection |
| `QuantizationConfig.weight_quantization_calibration_method` | QwixFP8FullConfig.weight_quantization_calibration_method, BatchSplitQwixConfig.weight_quantization_calibration_method | Actual recipe consumers only; no separate expert weight-calibration consumer was found |
| `QuantizationConfig.act_quantization_calibration_method` | QwixFP8FullConfig.act_quantization_calibration_method, BatchSplitQwixConfig.act_quantization_calibration_method, ServeFP8WeightConfig.act_quantization_calibration_method; shared MoEConfig.general field | Explicit recipe and MoE calibration values must agree |
| `QuantizationConfig.bwd_quantization_calibration_method` | QwixFP8FullConfig.bwd_quantization_calibration_method, BatchSplitQwixConfig.bwd_quantization_calibration_method | Actual Qwix full/batch consumers only |
| `QuantizationConfig.drhs_grad_quantization_calibration_method` | QwixFP8FullConfig.drhs_grad_quantization_calibration_method | FP8_FULL cotangent-arm scale semantics |
| `QuantizationConfig.quantize_kvcache` | MaxTextConfig.kv_cache_quantization_config.quantize_kvcache | Independent cache behavior |
| `QuantizationConfig.kv_quant_axis` | MaxTextConfig.kv_cache_quantization_config.kv_quant_axis | Independent cache behavior |
| `QuantizationConfig.kv_quant_dtype` | MaxTextConfig.kv_cache_quantization_config.kv_quant_dtype | Independent cache format |
| `QuantizationConfig.weight_sparsity_n` | MaxTextConfig.sparsity_config.weight_sparsity_n | Sparsity is not QAT/compute dtype |
| `QuantizationConfig.weight_sparsity_m` | MaxTextConfig.sparsity_config.weight_sparsity_m | Existing N:M sparsity |
| `QuantizationConfig.weight_sparsity_update_step` | MaxTextConfig.sparsity_config.weight_sparsity_update_step | Existing update schedule |
| `QuantizationConfig.weight_sparsity_start_step` | MaxTextConfig.sparsity_config.weight_sparsity_start_step | Existing start schedule |

### Superseded intermediate recipe ownership

Original locations for every row are FIELDS.md focused quantization declarations and ALL-ARGS section 4's two pretrained recipe calls. API.md/model-args.md also exposed the old union. Each cell names both exact owner paths when the field was inherited. No row relocates a numerical control to YAML. Implemented destinations are inspectable in PUBLIC_API.md. Defaults remain in the [source declarations](src/lilax/configuration/domains/model_quantization.py); applicability follows [configure.py](src/lilax/_backend/layers/quantization/configure.py), [qwix_rules.py](src/lilax/_backend/layers/quantization/qwix_rules.py) and [GMM rule selection](src/lilax/_backend/kernels/megablox/quantization_rule.py).

| Exact removed intermediate path or relationship | Current disposition/destination | Reason |
|---|---|---|
| `QwixRecipeConfig(ComputeRecipeConfig)` inheritance | Removed public inheritance; independent focused types | No inherited cross-provider controls |
| `ComputeRecipeConfig.quantization` universal enum annotation | Selector narrowed to NONE, FP8, NANOO_FP8 and TE recipes | AQT int8/intmp, JAX MX, serving, Qwix and batch-split select through their focused types |
| `QwixRecipeConfig.quantization` universal enum annotation | Uniform Qwix selectors only; FP8_FULL selected by QwixFP8FullConfig | Actual rule/provider dispatch, not all source enum members |
| `ComputeRecipeConfig.jax_mx_gradient_mode`, `QwixRecipeConfig.jax_mx_gradient_mode` | JaxMXConfig.jax_mx_gradient_mode | Not consumed by Qwix or other providers |
| `ComputeRecipeConfig.unquantized_modules`, `QwixRecipeConfig.unquantized_modules` | PrecisionConfig.unquantized_modules | Stored dtype/native quant selection, not Qwix interception exclusions |
| `ComputeRecipeConfig.weight_block_size`, `QwixRecipeConfig.weight_block_size` | PrecisionConfig.weight_block_size | Weight metadata independent of compute recipe |
| `ComputeRecipeConfig.quant_cfg_path`, `QwixRecipeConfig.quant_cfg_path` | AqtMixedPrecisionConfig.quant_cfg_path | Only intmp reads the JSON |
| `ComputeRecipeConfig.quantize_mtp`, `QwixRecipeConfig.quantize_mtp` | QwixFP8FullConfig.quantize_mtp | FP8_FULL-only rule |
| `ComputeRecipeConfig.quantize_router_proj` | QwixRecipeConfig.quantize_router_proj, QwixFP8FullConfig.quantize_router_proj | Qwix-only rule; QwixRecipeConfig field survives with its default |
| `ComputeRecipeConfig.quantize_logits_proj`, `QwixRecipeConfig.quantize_logits_proj` | QwixFP8FullConfig.quantize_logits_proj | FP8_FULL-only rule |
| `ComputeRecipeConfig.logits_proj_quant_calibration_method`, `QwixRecipeConfig.logits_proj_quant_calibration_method` | QwixFP8FullConfig.logits_proj_quant_calibration_method | FP8_FULL logits calibration only |
| `ComputeRecipeConfig.logits_proj_bwd_quant_calibration_method`, `QwixRecipeConfig.logits_proj_bwd_quant_calibration_method` | QwixFP8FullConfig.logits_proj_bwd_quant_calibration_method | FP8_FULL logits backward calibration only |
| `ComputeRecipeConfig.sa_qproj_bwd_bf16`, `QwixRecipeConfig.sa_qproj_bwd_bf16` | QwixFP8FullConfig.sa_qproj_bwd_bf16 | FP8_FULL query-gradient rule only |
| `ComputeRecipeConfig.quantization_local_shard_count`, `QwixRecipeConfig.quantization_local_shard_count` | AqtInt8Config.quantization_local_shard_count | Only AQT int8 consumes it |
| `ComputeRecipeConfig.qwix_bwd_weight_grad_tile_count` | QwixRecipeConfig.qwix_bwd_weight_grad_tile_count | Uniform Qwix only; surviving Qwix field retains default 1 |
| `ComputeRecipeConfig.use_manual_quantization`, `QwixRecipeConfig.use_manual_quantization` | Boolean removed; BatchSplitManualConfig/BatchSplitQwixConfig select True/False respectively | No manual-switch argument on graph or ordinary compute recipes |
| `ComputeRecipeConfig.weight_quantization_calibration_method`, `QwixRecipeConfig.weight_quantization_calibration_method` | QwixFP8FullConfig.weight_quantization_calibration_method, BatchSplitQwixConfig.weight_quantization_calibration_method | Uniform Qwix does not consume it |
| `ComputeRecipeConfig.act_quantization_calibration_method`, `QwixRecipeConfig.act_quantization_calibration_method` | QwixFP8FullConfig.act_quantization_calibration_method, BatchSplitQwixConfig.act_quantization_calibration_method, ServeFP8WeightConfig.act_quantization_calibration_method; shared MoEConfig.general field | Explicit recipe and MoE values must agree |
| `ComputeRecipeConfig.bwd_quantization_calibration_method`, `QwixRecipeConfig.bwd_quantization_calibration_method` | QwixFP8FullConfig.bwd_quantization_calibration_method, BatchSplitQwixConfig.bwd_quantization_calibration_method | Only actual consuming paths |
| `ComputeRecipeConfig.drhs_grad_quantization_calibration_method`, `QwixRecipeConfig.drhs_grad_quantization_calibration_method` | QwixFP8FullConfig.drhs_grad_quantization_calibration_method | FP8_FULL cotangent-arm override only |
| Transient `ComputeRecipeConfig.quantization` values `QuantizationType.AQT_FP8`, `QuantizationType.AQT_FP8_FULL` | Removed unsupported declaration spellings, not a backend capability deletion | Neither enum member exists; these names are not supported recipe selectors |

The old global `QuantizationConfig.use_qwix_quantization` removal does not remove BatchSplitManualConfig's source-relevant GMM lookup choice. Its False default is preserved. No new graph transformation is implied. Shared expert-MoE activation calibration is owned by `MoEConfig.general.act_quantization_calibration_method` and the selected consumed-field recipe. Explicit values must agree. Existing GMM calibration validation remains unchanged. The earlier retained GrainOptions/load_olmo_dataset disposition is superseded by the explicit public-removal table above.
| `checkpoint_config.precision_config` example mutation | AutoModelForCausalLM.from_pretrained.precision_config | Direct loader ownership; config persistence field remains |
| `AutoModelForCausalLM.from_pretrained.runtime_config` | Internal model runtime | Earlier focused draft contradicted ordinary loader |
| `AutoModelForCausalLM.from_pretrained.construction_config` | Internal construction | No public lifecycle bag |
| `AutoModelForCausalLM.from_pretrained.conversion_config` | Standalone convert_checkpoint config only | Conversion staging is not ordinary loader config |
| `AutoModelForCausalLM.from_pretrained.rng_key` | Internal loader initialization | No new public pretrained RNG contract |
| `AutoModelForCausalLM.from_config.runtime_config` | Internal runtime | Separate from-scratch seed remains public |
| `AutoModelForCausalLM.from_config.construction_config` | Internal construction | No lifecycle bag |
| `AutoModelForCausalLM.from_config.rngs` | Public seed argument replaces draft NNX generator | Matches integrated API declaration, not numerical rewrite |
| `ModelConstructionConfig` | Internal-only record | Old focused draft contradicted ordinary loader |
| `ModelConstructionConfig.model_mode` | Internal construction | Runtime mode |
| `ModelConstructionConfig.quant_mode_str` | Internal construction | Internal train/serve dispatch |

The earlier replicate_quant_scale and te_comm_gemm_overlap rows remain profile relocations. QwixQATConfig and every QwixQATRule leaf remain public, unchanged. fuse_qkv/fuse_mlp are direct loader selections and do not remove architecture metadata fields. Supported fusion must pass projection_layout.py's existing model/quantization checks. No kernel or math change is claimed.

## Internal compiler, estimator and profiler operations

PEFT's old `format="hf"` and `format="native"` values on `PeftModel.save_pretrained` and `PeftModel.from_pretrained` are replaced by the agreed `"huggingface"` and `"maxtext"` values. The variadic raw-native `PeftModel.__call__(*args, **kwargs)` override is removed; PeftModel inherits the finite canonical MaxTextModel forward arguments. `LoraConfig` retains its fields as keyword-only configuration arguments. Adapter tensor/conversion/restore behavior is unchanged by those interface edits.

The standalone `lilax.trainers.post_training.experimental_grpo.train_grpo` entrypoint is removed. Its computation is selected by `GRPOTrainer` with `RolloutConfig(engine="maxtext")`.

| Removed argument | Destination | Reason |
|---|---|---|
| `train_grpo.config` | GRPOTrainer model, GRPOConfig and execution profile | No public whole-run configuration workflow |
| `train_grpo.inference_config` | RolloutConfig cluster/generation settings and internal inference projection | Preserve native prefix allocation and source-backed sampler setup |
| `train_grpo.report_to` | GRPOConfig.report_to and callbacks | One reporting owner |
| `train_grpo.launch_xprof` | Execution profile xprof.launch_server | Deployment setting |
| `train_grpo.xprof_port` | Execution profile xprof.port | Deployment setting |

Original location is ALL-ARGS section 5, EXAMPLES section 11, FIELDS compilation declarations and extension-args.md. Evidence is [train_compile.py](src/lilax/_backend/trainers/pre_train/train_compile.py), [estimator.py](src/lilax/_backend/trainers/pre_train/estimator.py) and [compilation.py](src/lilax/_backend/trainers/setup/compilation.py). Internalizing these does not schedule their execution automatically.

| Exact removed symbol/argument path | Disposition/destination | Reason |
|---|---|---|
| `compile_train_step` | Internal compiler operation | Not ordinary training API |
| `compile_train_step.trainer` | Internal compiler operation | Existing train compilation input |
| `compile_train_step.config` | Internal compiler/profile config | CompilationConfig removed publicly; its leaves already enumerated |
| `compile_engine` | Internal compiler operation | Existing DiLoCo restriction preserved |
| `compile_engine.trainer` | Internal compiler operation | Existing engine input |
| `compile_engine.config` | Internal compiler/profile config | No public compiler bag |
| `is_oom` | Internal capacity-search operation | Not ordinary training API |
| `is_oom.trainer` | Internal capacity search | Existing input |
| `is_oom.config` | Internal compiler/profile config | Existing compile/OOM probe |
| `estimate` | Internal expensive estimator | No automatic train search |
| `estimate.trainer` | Internal estimator | Existing batch/remat candidate input |
| `estimate.write_estimator_result` | Internal estimator | Existing result output policy |
| `stage_xprof_logdir` | Internal profile staging helper | No public staging lifecycle |
| `stage_xprof_logdir.logdir` | Direct SFTConfig.xprof_dir | Trainer owns requested profile path |
| `stage_xprof_logdir.step` | Internal trainer step | No manual staging step |

## Internal serialized job records

Original locations are ALL-ARGS section 8, EXAMPLES section 14 and FIELDS job declarations. These records are internal-only, not another public training schema. ModelLoadSpec, HFDataSpec and TokenizerLoadSpec shared leaves are enumerated once below and apply at every nested train/eval/student/teacher occurrence. The args edges embed still-public TrainingArguments/DistillationConfig and their focused children; those numerical leaves survive visibly in ALL-ARGS sections 1/2, so they are not removed public settings. TextProcessingOptions leaves are enumerated in the processing table above. No record's removal means its backend pipeline or job implementation was deleted.

| Exact removed public record/field | Internal destination | Reason |
|---|---|---|
| `ModelLoadSpec` | Serialized model job payload | Ordinary code calls pretrained loader directly |
| `ModelLoadSpec.pretrained_model_name_or_path` | Serialized model job payload | Existing checkpoint selector |
| `ModelLoadSpec.config` | Serialized model job payload | Model config remains public on loader |
| `ModelLoadSpec.revision` | Serialized model job payload | Loader option remains public |
| `ModelLoadSpec.token` | Serialized model job payload | Loader option remains public |
| `ModelLoadSpec.cache_dir` | Serialized model job payload | Loader option remains public |
| `ModelLoadSpec.local_files_only` | Serialized model job payload | Loader option remains public |
| `ModelLoadSpec.quantization_config` | Serialized model job payload | Focused loader union remains public |
| `TokenizerLoadSpec` | Serialized tokenizer job payload | Direct Lilax AutoTokenizer |
| `TokenizerLoadSpec.pretrained_model_name_or_path` | Serialized tokenizer job payload | Public wrapper loader retains selector |
| `TokenizerLoadSpec.revision` | Serialized tokenizer job payload | Direct loader option retained |
| `TokenizerLoadSpec.token` | Serialized tokenizer job payload | Direct loader option retained |
| `TokenizerLoadSpec.cache_dir` | Serialized tokenizer job payload | Direct loader option retained |
| `TokenizerLoadSpec.local_files_only` | Serialized tokenizer job payload | Direct loader option retained |
| `HFDataSpec` | Serialized dataset job payload | Direct datasets.load_dataset |
| `HFDataSpec.path` | Serialized dataset job payload | Upstream loader retains selector |
| `HFDataSpec.name` | Serialized dataset job payload | Upstream loader retains subset |
| `HFDataSpec.data_dir` | Serialized dataset job payload | Upstream loader retains directory |
| `HFDataSpec.data_files` | Serialized dataset job payload | Upstream loader retains file selection |
| `HFDataSpec.split` | Serialized dataset job payload | Upstream loader retains split |
| `HFDataSpec.revision` | Serialized dataset job payload | Upstream loader retains revision |
| `HFDataSpec.token` | Serialized dataset job payload | Upstream loader retains credential |
| `HFDataSpec.streaming` | Serialized dataset job payload | Upstream streaming remains public |
| `DatasetSourceSpec` | Internal serialized source union | No separate native public mode |
| `PretrainingJobSpec` | Internal serialized trainer payload | Direct Trainer lifecycle |
| `PretrainingJobSpec.model` | Internal ModelLoadSpec edge | Shared leaves above |
| `PretrainingJobSpec.args` | Internal serialized args edge | Every common nested setting survives at Trainer.args |
| `PretrainingJobSpec.train_dataset` | Internal HFDataSpec/source edge | Shared leaves above; direct train_dataset survives |
| `PretrainingJobSpec.processing_class` | Internal TokenizerLoadSpec edge | Direct Trainer.processing_class survives |
| `PretrainingJobSpec.eval_dataset` | Internal HFDataSpec/source edge | Direct eval_dataset survives |
| `PretrainingJobSpec.processing_options` | Removed serialized processing edge | Reassigned/removed TextProcessingOptions leaves above |
| `PretrainingJobSpec.resume_from_checkpoint` | Internal serialized lifecycle field | Trainer.train resume remains public |
| `DistillationJobSpec` | Internal serialized trainer payload | Direct DistillationTrainer lifecycle |
| `DistillationJobSpec.model` | Internal ModelLoadSpec edge | Shared leaves above |
| `DistillationJobSpec.teacher_model` | Internal ModelLoadSpec edge | Direct teacher_model survives |
| `DistillationJobSpec.args` | Internal serialized algorithm args edge | All common/algorithm/focused leaves survive at Trainer.args |
| `DistillationJobSpec.train_dataset` | Internal HFDataSpec/source edge | Direct train_dataset survives |
| `DistillationJobSpec.processing_class` | Internal TokenizerLoadSpec edge | Single direct processing_class survives |
| `DistillationJobSpec.eval_dataset` | Internal HFDataSpec/source edge | Direct eval_dataset survives |
| `DistillationJobSpec.processing_options` | Removed serialized processing edge | Processing leaves accounted above |
| `DistillationJobSpec.resume_from_checkpoint` | Internal serialized lifecycle field | Public train resume selection survives |
| `DistillationJob` | Internal job infrastructure | No ordinary public job lifecycle |
| `DistillationJob.training` | Internal DistillationJobSpec edge | Nested fields enumerated above |
| `DistillationJob.cluster` | Internal launcher/deployment | Cluster selection; profile exact fields in EXECUTION_PROFILE.md |
| `DistillationJob.project` | Internal launcher/deployment | Cloud project; profile exact fields in EXECUTION_PROFILE.md |
| `DistillationJob.zone` | Internal launcher/deployment | Placement; profile exact fields in EXECUTION_PROFILE.md |
| `DistillationJob.device_type` | Internal launcher/deployment | Device selection; profile exact fields in EXECUTION_PROFILE.md |
| `DistillationJob.workload` | Internal job implementation | Workload identity |
| `DistillationJob.image` | Internal launcher/deployment | Runtime image; profile exact fields in EXECUTION_PROFILE.md |
| `DistillationJob.staging_path` | Internal job implementation | Generated serialized payload path |
| `DistillationJob.staging_gcs` | Internal job implementation | Generated payload upload path |
| `DistillationJob.config_path` | Internal job implementation | Earlier focused draft spelling of staging_path |
| `DistillationJob.config_gcs` | Internal job implementation | Earlier focused draft spelling of staging_gcs |
| `DistillationJob.libtpu_init_args` | Internal launcher/compiler | Init flags; profile exact fields in EXECUTION_PROFILE.md |
| `DistillationJob.priority` | Internal launcher/deployment | Scheduler priority; profile exact fields in EXECUTION_PROFILE.md |
| `DistillationJob.num_slices` | Internal launcher/deployment | Physical topology; profile exact fields in EXECUTION_PROFILE.md |
| `DistillationJob.dataset_bucket` | Internal staging implementation | Existing remote data staging |
| `DistillationJob.dataset_subpath` | Internal staging implementation | Existing remote data staging |
| `DistillationJob.use_gcsfuse` | Internal launcher/storage | Storage mount; profile exact fields in EXECUTION_PROFILE.md |
| `DistillationJob.tokenizer_gcs` | Internal staging implementation | Existing tokenizer asset staging |
| `DistillationJob.tokenizer_local` | Internal staging implementation | Existing tokenizer asset staging |
| `DistillationJob.hf_cache_dir` | Internal launcher/storage | Worker cache; profile exact fields in EXECUTION_PROFILE.md |
| `ResumeResult` | Internal job lifecycle result | No public retry loop |
| `ResumeResult.completed` | Internal job lifecycle result | Existing completion status |
| `ResumeResult.attempts` | Internal job lifecycle result | Existing retry count |
| `ResumeResult.checkpoint_step` | Internal job lifecycle result | Existing progress status |
| `ResumeResult.terminal_state` | Internal job lifecycle result | Existing terminal status |

## Removed ordinary launch, provisioning and lifecycle calls

Every row below originally appeared in ALL-ARGS section 8, EXAMPLES section 14, or extension-args.md's launch/provisioning/jobs section. Existing [trainers/jobs/diloco.py](src/lilax/_backend/launch/jobs/diloco.py), launch and provisioning implementations remain backend/infrastructure concerns. The destination is internal launcher/infrastructure ownership, not a new implemented profile schema. These private provisioning operations are not execution-profile commands. DiLoCo's algorithm settings remain public at SFTConfig.diloco_config and run through Trainer; wizard selection remains unchanged. Ordinary training does not call any operation below automatically.

| Exact removed public function/argument | Destination/reason |
|---|---|
| `submit_diloco_job` | Internal launcher job submission, not training lifecycle |
| `submit_diloco_job.training` | Internal serialized job payload, leaves above |
| `submit_diloco_job.project_id` | Deployment project |
| `submit_diloco_job.cluster_name` | Deployment cluster |
| `submit_diloco_job.location` | Deployment placement |
| `submit_diloco_job.docker_image` | Deployment image |
| `submit_diloco_job.tpu_type` | Deployment device |
| `submit_diloco_job.num_slices` | Deployment topology |
| `submit_diloco_job.workload_name` | Job identity |
| `submit_diloco_job.xla_flags` | Launcher compiler flags; source constant unchanged |
| `submit_diloco_job.priority` | Scheduler policy |
| `submit_diloco_job.max_restarts` | Infrastructure retry policy |
| `DEFAULT_DILOCO_JOB_XLA_FLAGS` | Internal source constant, not ordinary public args |
| `stage` | Internal job staging operation |
| `stage.job` | Internal job record, leaves above |
| `submit` | Internal job submission operation |
| `submit.job` | Internal job record |
| `submit.stage_config` | Internal serialized payload staging policy |
| `submit.steps` | Internal job launch target; direct Trainer max_steps survives |
| `submit.checkpoint_period` | Internal job recipe override; common save_steps survives |
| `monitor` | Internal job lifecycle operation |
| `monitor.job` | Internal job record |
| `monitor.log_path` | Infrastructure log output |
| `resume_until_done` | Internal retry-to-target operation, no automatic retries on train |
| `resume_until_done.job` | Internal job record |
| `resume_until_done.steps` | Infrastructure target |
| `resume_until_done.max_retries` | Infrastructure retry policy |
| `resume_until_done.checkpoint_period` | Existing job checkpoint policy |
| `delete` | Internal explicit resource deletion, never automatic training cleanup |
| `delete.job` | Internal deletion target |
| `launch` | Internal jaxcoord boundary, not public training call |
| `launch.script` | Launcher program selection |
| `launch.script_args` | Launcher program arguments |
| `launch.num_nodes` | Deployment topology |
| `launch.node_rank` | Deployment process identity |
| `launch.main_process_ip` | Deployment network |
| `launch.main_process_port` | Deployment network |
| `launch.gpu_ids` | Deployment devices |
| `run_gpu_multi_process` | Internal process launcher |
| `run_gpu_multi_process.command` | Launcher command |
| `run_multihost` | Internal existing-host launcher |
| `run_multihost.tpu_prefix` | Deployment host selector |
| `run_multihost.command` | Launcher command |
| `run_multihost.project` | Deployment project |
| `run_multihost.zone` | Deployment placement |
| `run_multihost.script_dir` | Launcher staging |
| `run_multihost.run_name` | Launcher identity |
| `run_multihost.use_existing_folder` | Launcher staging policy |
| `run_multihost.internal_ip` | Deployment network |
| `run_multihost.scp_timeout_secs` | Launcher transfer timeout |
| `run_multihost_job` | Internal provisioning/job launcher |
| `run_multihost_job.command` | Launcher command |
| `run_multihost_job.bucket_name` | Launcher storage |
| `run_multihost_job.tpu_type` | Deployment device |
| `run_multihost_job.version` | Deployment image version |
| `run_multihost_job.num_slices` | Deployment topology |
| `run_multihost_job.script_dir` | Launcher staging |
| `run_multihost_job.bucket_dir` | Launcher staging |
| `run_multihost_job.project` | Deployment project |
| `run_multihost_job.zone` | Deployment placement |
| `run_multihost_job.run_name` | Launcher identity |
| `run_multihost_job.cqr_extra_args` | Deployment reservation policy |
| `run_multihost_job.enable_autocheckpoint` | Infrastructure recovery; common checkpoint option remains public |
| `setup` | Internal isolated-environment provisioning |
| `setup.python` | Selected isolated interpreter |
| `setup.source_project` | Provisioned project |
| `setup.device` | Device-specific dependency selection |
| `setup.mode` | Provisioning mode |
| `setup.workflow` | Provisioned workload dependencies |
| `setup.with_tf` | Provisioned optional dependency |
| `setup.jax_version` | Provisioned dependency version |
| `setup.libtpu_version` | Provisioned dependency version |
| `setup.libtpu_gcs_path` | Provisioned dependency source |
| `setup.vllm_adapter` | Provisioned optional dependency |
| `setup.max_attempts` | Provisioning retry policy |
| `setup.retry_delay` | Provisioning retry policy |
| `setup.allow_system_changes` | Provisioning safety policy |
| `setup.grain_wheels` | Provisioned dependency source |
| `install_overlay` | Internal dependency overlay installation |
| `install_overlay.python` | Isolated target interpreter |
| `install_overlay.source_project` | Overlay source |
| `install_overlay.workflow` | Overlay dependency selection |
| `install_overlay.vllm_adapter` | Overlay dependency selection |
| `create_python_312_environment` | Internal isolated-environment creation |
| `create_python_312_environment.bootstrap_python` | Provisioning bootstrap interpreter |
| `create_python_312_environment.environment_path` | Provisioning destination |
| `provision_host` | Internal no-argument host administration |
| `run_preflight` | Internal no-argument infrastructure preflight |

## Removed container exports and corrected draft aliases

These rows supplement the leaf tables, not replace them. Original locations are the pre-correction ALL-ARGS sections and focused declarations named below. Profile destinations use the codes defined above and have exact fields in EXECUTION_PROFILE.md. No internal dependency is deleted. Evidence remains the original [checkpoint domains](src/lilax/configuration/domains/model_checkpointing.py), [monitoring domains](src/lilax/configuration/domains/execution_monitoring.py), the original-owner leaf tables above and the corrected API.md/FIELDS.md contracts.

| Exact original symbol/owner path | Original example/declaration location | Disposition/destination | Reason |
|---|---|---|---|
| `OrbaxStorageConfig` | 1, TrainingCheckpointConfig.storage; API checkpoint declarations | Profile/internal-only storage record | All five leaves enumerated under TrainingCheckpointConfig.storage above |
| `TrainingCheckpointConfig.storage` | 1, nested checkpoint constructor | Removed public edge; storage | Storage tuning is not save policy |
| `EmergencyCheckpointConfig` | 1, TrainingCheckpointConfig.emergency; API checkpoint declarations | Profile/internal-only recovery record | All nine leaves enumerated under TrainingCheckpointConfig.emergency above |
| `TrainingCheckpointConfig.emergency` | 1, nested checkpoint constructor | Removed public edge; recovery | Deployment storage tiers |
| `ParallelismConfig.ici` | 1, SFTConfig.parallelism_config | Removed public edge; topology | All ICI leaves enumerated above |
| `ParallelismConfig.dcn` | 1, SFTConfig.parallelism_config | Removed public edge; topology | All DCN leaves enumerated above |
| `ParallelismConfig.pipeline` | 1, SFTConfig.parallelism_config | Removed public edge; pipeline | All pipeline leaves enumerated above |
| `ElasticConfig` | 1/5, elastic strategy; FIELDS execution | Profile/internal-only recovery record | All six leaves enumerated above |
| `TrainingArguments.elastic_config` | 1/2/4/6/8, inherited trainer args | Removed public edge; recovery | No public deployment recovery bag |
| `DiLoCoConfig.bandwidth` | 1/5/8, DiLoCo strategy | Removed public edge; transport | All four leaves enumerated above |
| `CompilationConfig.hlo_dump` | 1/5, compiler config | Removed public edge; diagnostics | Every HLO/JAXPR leaf enumerated above |
| `TrainExecutionConfig` | 1/2/4/6/8, trainer args; FIELDS execution | Profile/internal-only execution record | All leaves enumerated above |
| `TrainingArguments.execution_config` | 1/2/4/6/8, inherited trainer args | Removed public edge; execution | No backend lifecycle bag |
| `TrainExecutionConfig.diagnostics` | 1, execution config | Removed public edge; diagnostics | Three diagnostic leaves enumerated above |
| `DiagnosticsConfig` | FIELDS execution, TrainExecutionConfig.diagnostics | Profile/internal-only diagnostics record | Same three nested leaves, not additional controls |
| `TrainingArguments.accelerator_config` | training-args common args draft | Removed redundant public composition edge | Common parallelism/remat/timeout/DiLoCo fields remain direct focused owners; no second accelerator bag |
| `GoodputCallback` | 5, callback_choices; API callbacks | Internal-only deployment instrumentation | No ordinary training callback export |
| `GoodputCallback.config` | 5, callback_choices | Removed public edge; diagnostics | All eleven leaves enumerated above |
| `GoodputConfig` | FIELDS observability | Profile/internal-only diagnostics record | Same eleven GoodputCallback.config leaves |
| `ManagedDiagnosticsCallback` | 5, callback_choices; API callbacks | Internal-only managed diagnostics lifecycle | Not an ordinary reporting integration |
| `ManagedDiagnosticsCallback.config` | 5, callback_choices | Removed public edge; diagnostics | All six leaves enumerated above |
| `ManagedDiagnosticsConfig` | FIELDS observability | Profile/internal-only diagnostics record | Same six ManagedDiagnosticsCallback.config leaves |
| `GcpMonitoringCallback` | 5, callback_choices; API callbacks | Internal-only cloud monitoring lifecycle | Deployment heartbeat/performance service |
| `GcpMonitoringCallback.config` | 5, callback_choices | Removed public edge; diagnostics | All three leaves enumerated above |
| `GcpMonitoringConfig` | FIELDS observability | Profile/internal-only diagnostics record | Same three GcpMonitoringCallback.config leaves |
| `XProfCallback` | 5, callback_choices; training-args callbacks | Removed public deployment-server callback | launch_server/port already enumerated; SFTConfig.xprof_dir is direct public path |
| `MetricStorageConfig` | FIELDS observability; extension-args profiling | Removed public container | Common outputs are direct MetricsCallback args; internal metrics remain profile-owned |
| `MetricStorageConfig.metrics_file` | FIELDS observability; extension-args profiling | MetricsCallback.metrics_file | Retained public, no redundant container |
| `MetricStorageConfig.gcs_metrics` | FIELDS observability; extension-args profiling | MetricsCallback.gcs_metrics | Retained public output selection |
| `MetricStorageConfig.save_config_to_gcs` | FIELDS observability; extension-args profiling | MetricsCallback.save_config_to_gcs | Retained public output selection |
| `MetricStorageConfig.record_internal_nn_metrics` | FIELDS observability; extension-args profiling | diagnostics | Internal instrumentation |
| `MetricStorageConfig.prometheus_port` | FIELDS observability; extension-args profiling | diagnostics | Metrics server deployment |
| `GrainDataset` | API imports/DatasetLike and data-descriptor prose | Internal-only descriptor/type | Removed alongside public Grain loader; backend pipeline remains a dependency |
| `TfdsDataset` | API imports/DatasetLike and data-descriptor prose | Internal-only descriptor/type | Removed alongside public TFDS loader; backend pipeline remains a dependency |
| `GRPOEvaluationConfig.correct_only` | training-args algorithm declaration | Corrected draft alias; response_selection="incorrect" | False did not mean all responses; preserves existing eval_corr_lst=False selection |

GrainDataset/TfdsDataset were descriptor names without finite public constructor declarations in API.md. No unobserved constructor arguments are invented here. Shared removed nested record leaves are enumerated once at their original ownership paths above, with every removed enclosing edge identified here. Existing internal type names are not promised as future public exports or final YAML schema.

## Existing post-training checkpoint selectors

| Original public path | Retained owner | Applicability |
|---|---|---|
| `SFTConfig.save_only_lora_params`, `DPOConfig.save_only_lora_params`, `DistillationConfig.save_only_lora_params` | `TrainingCheckpointConfig.save_only_lora_params` | Original `None` default follows adapter enablement; `False` saves full model state on Tunix |
| `SFTConfig.restore_only_lora_params`, `DPOConfig.restore_only_lora_params`, `DistillationConfig.restore_only_lora_params` | `TrainingCheckpointConfig.restore_only_lora_params` | Original `None` default follows adapter enablement; `False` restores full model state on Tunix |

These source-existing selectors were omitted from the draft rather than explicitly removed. Their manager behavior remains; the native pretraining/SFT and MaxText GRPO backends reject explicit selection.

## SFT demo migration

These example-local helpers no longer expose backend training or rollout setup.

| Removed helper or argument | Current example owner |
|---|---|
| `get_test_dataset.config` | Explicit `token` forwarded to HF datasets |
| `get_test_dataset.tokenizer` | `processing_class` |
| `get_test_dataset.data_template_path` | Existing bundled GSM8K math template, local to the example |
| `evaluate_model.vllm_rollout` | Supplied public `model` and `processing_class` |
| `setup_trainer(trainer)` | `SFTTrainer` lifecycle |
| `create_rollout(trainer, tokenizer, model_version, mesh)` | `model.generate`; deployment/cache controls are not training prerequisites |
| `train_trainer(trainer, mesh)` | `trainer.train()` and its `TrainOutput` |
| `sync_lora_for_evaluation(trainer, config, vllm_rollout)` | Generation uses the same adapted model; repeated in-place delta addition is removed |

The example retains exact, within-10-percent and format scoring, seed42, batch4, prompt limit256,
generation limit768 and temperature0.0001/top-k1/top-p1.0. The former rollout construction's
cache size1280, HBM fraction0.3, random initialization, TPU backend and512-token initialization
setting are removed with that obsolete setup helper. They are not model-generation defaults.

## Native inference wrapper removal

| Former public operation | Current owner |
|---|---|
| `generation.engine.MaxEngine(config, devices=None)` | Internal model-generation and benchmark adapter in `generation._engine` |
| `generation.engine.generate(native_model, runtime, padded_tokens, true_length, construction_config, max_new_tokens, rng, algorithm, topk, nucleus_topp, temperature, positions, mrope_deltas, images, image_masks, videos, video_masks, video_grid_thw, audio_values, audio_masks)` | `MaxTextModel.generate` with canonical input names; internal numerical implementation retained |
| `generation.engine.create_engine_from_config(config, batch_size, max_prefill_predict_length, max_target_length, devices, **overrides)` | Existing MLPerf benchmark consumer, now importing the private adapter |
| `serving.vllm.decode_with_vllm(config)` | Existing backend implementation remains private; public serving uses `create_vllm_llm(model, serving_config=...)` |
| `serving.vllm.decode_with_tunix(model)` | Existing backend implementation remains private; public model generation owns supplied-model decoding |

The engine move changes imports only. Its sampling, cache and tensor computation bodies are preserved.

Public model generation sizes the existing private prefill-cache setting from the supplied prompt
when it exceeds the source default. Short prompts retain the original setting. The model context
length and existing per-engine decode-budget checks remain authoritative; no fixed prompt cap is
added to GenerationConfig and model graph metadata is not relabeled.

`TrainingArguments.resolve(runtime, train_dataset_provided, eval_dataset_provided)` is internalized
as `_resolve`. Focused rollout/training records' `source_values()` translations are also private.
Actual Trainer callers use the private methods; ordinary callers do not install a native runtime.

## Resolved implementation details

Profile categories in the historical tables map to the exact sections below. Within a section,
retained source fields keep their source spelling. The full typed section declarations are linked
from EXECUTION_PROFILE.md. Provisioning and job-submission arguments remain private operations;
they are not automatically converted to execution-profile settings.

| Historical category | Implemented YAML section |
|---|---|
| topology | `mesh`, `ici`, `dcn` |
| sharding | `sharding`, with physical mesh settings in `mesh` |
| pipeline | `pipeline` |
| init | `process` and `debugging` |
| storage | `storage` |
| recovery | `recovery` for elastic settings, `replication` for emergency/multitier settings |
| transport | `diloco` for DiLoCo settings, `transport` for quantization-scale and TE overlap |
| compiler | `compilation`, `debugging`, `hlo`, and top-level `xla_flags` |
| diagnostics | `profiling`, `rl_profiling`, `xprof`, `metrics`, `goodput`, `diagnostics`, `monitoring` |
| execution | `execution` |
| tensors | `remat` |
| kernels | `kernels` and `splash` |
| `XProfCallback.launch_server`, `.port` | `xprof.launch_server`, `xprof.port` |
| `RLProfilerCallback.max_num_hosts` | `rl_profiling.profiler_max_num_hosts` |
| `JetStreamConfig.enable_jax_profiler`, `.jax_profiler_port` | `profiling.enable_jax_profiler`, `profiling.jax_profiler_port` |

`BenchmarkModel` in `lilax.benchmarks.cluster` names the existing specialist benchmark recipe
record, with `model_name`, `model_type`, `tuning_params`, `xla_flags`, `pathways_tuning_params`
and `pathways_xla_flag_options`. `WorkloadConfig.model` accepts that record. It is not a public
job-submission spec or the trainable `lilax.MaxTextModel`.

`MaxTextConfig.moe_intermediate_size` exposes the effective expert width that source loading
and ModelScalingConfig already produced. `None` retains the selected model's expert width.
It maps to the existing `base_moe_mlp_dim` with global scale1 after dimensions are effective.
ModelScalingConfig applies the original power-of-two formulas before this projection.
The field follows Transformers5.12.1 Qwen3MoeConfig's effective-dimension naming; no generic
expert width or new numerical default is introduced.

Previously unassigned multimodal fields follow their actual source consumers. Vision encoder
selection, freezing, image size and vision position embedding belong to VisionConfig. Audio
selection and freezing belong to AudioConfig. MRoPE selection/sections/temporal granularity belong
to PositionConfig. AutoProcessor owns audio_placeholder, use_audio_in_video and the three optional
video_max_grid_t/h/w preprocessing bounds, alongside its existing image/video controls. Omitted
controls retain saved processor preferences or their original source defaults. The source video-grid
validator still requires bounds together. image_path, video_path and audio_path are application
media inputs, replaced by processor inputs and canonical model media arrays, not new config paths.

Direct AutoProcessor calls compose the existing native family preprocessors and token expansion.
Pass model-native media markers in text or use the processor's HF chat template. Images accept
the existing PIL, NumPy and file inputs. Video and Ming audio use the native file-path routes.
Batch media shapes must match, with max_num_images_per_example retaining native image padding.
The output binds directly to model forward and generation. Qwen position_ids retain the native
three-coordinate layout, and generation derives its existing MRoPE offset from those coordinates.
Processor save/reload retains the native model recipe and custom processing preferences.

Generation adapts the existing inference cache extent to the longest supplied prompt when it exceeds
the private prefill default. The model context limit and native decode checks still apply. No cache
kernel, sampling algorithm or context-extension behavior changes.

`RolloutConfig(engine="maxtext")` runs the existing native GRPO loop through GRPOTrainer. That source
uses its fixed length reward and has no evaluation-input pipeline. Explicit Tunix-only options,
custom rewards, agentic environments, LoRA and QAT raise applicability errors on that engine.
The existing native L-versus-L-1 log-probability issue and reference-model checkpoint recovery are
not resolved by this API migration. Exact streaming and rollout data-cursor resume is not established
by the existing Tunix model/optimizer checkpointer. Engine-local imports keep public declarations
importable on CPU; they do not resolve the conflicting pinned accelerator dependencies.

Trainer callbacks use Transformers identities and persistent completed-optimizer-step state.
Ordinary Orbax0.12.9 on_save dispatch requires the actual successful response, matching
`lilax_save_action` in exact native-step metadata and a nonnull commit timestamp. Emergency and
replicator receipts remain pending without on_save; no remote durability tier is claimed.
Dataset repetition keeps its existing backend behavior; epoch callbacks bracket that execution,
without inventing fractional epoch positions for streaming or packed input.
