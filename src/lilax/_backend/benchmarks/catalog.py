"""Source benchmark model configurations used by the runner and recipes."""

import os.path
from tempfile import gettempdir

from lilax._backend.benchmarks.cluster_runner import MaxTextModel
from lilax._backend.benchmarks.convergence.c4_exp import c4_deepseek_v3_ep_256_v5p_512_gbs_1024
from lilax._backend.utils.globals import MAXTEXT_ASSETS_ROOT


_ALL_GATHER = (
    " --xla_tpu_enable_async_collective_fusion=true"
    " --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true"
    " --xla_tpu_enable_async_collective_fusion_multiple_steps=true"
    " --xla_tpu_overlap_compute_collective_tc=true"
    " --xla_enable_async_all_gather=true"
)
_DATA_PARALLEL = (
    " --xla_tpu_enable_data_parallel_all_reduce_opt=true"
    " --xla_tpu_data_parallel_opt_different_sized_ops=true"
)
_REDUCE_SCATTER = (
    " --xla_tpu_use_minor_sharding_for_major_trivial_input=true"
    " --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1"
    " --xla_tpu_assign_all_reduce_scatter_layout=true"
)
_HOST_OFFLOAD = (
    " --xla_tpu_enable_all_experimental_scheduler_features=true"
    " --xla_tpu_enable_scheduler_memory_pressure_tracking=true"
    " --xla_tpu_host_transfer_overlap_limit=24"
    " --xla_tpu_aggressive_opt_barrier_removal=ENABLED"
    " --xla_lhs_prioritize_async_depth_over_stall=ENABLED"
    " --xla_tpu_enable_ag_backward_pipelining=true"
    " --xla_should_allow_loop_variant_parameter_in_chain=ENABLED"
    " --xla_should_add_loop_invariant_op_in_chain=ENABLED"
    " --xla_max_concurrent_host_send_recv=100"
    " --xla_tpu_scheduler_percent_shared_memory_limit=100"
    " --xla_latency_hiding_scheduler_rerun=2"
)
_SPARSE_ALL_REDUCE = (
    " --xla_tpu_enable_async_collective_fusion_fuse_all_reduce=false"
    " --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true"
    " --xla_tpu_enable_all_reduce_offload_tracing=true"
    " --xla_tpu_use_tc_device_shape_on_sc=true"
    " --xla_sc_enable_instruction_fusion=false"
    " --xla_sc_disjoint_spmem=false"
    " --xla_sc_disable_megacore_partitioning=true"
)
_LAUNCH_BARRIER = " --xla_tpu_use_enhanced_launch_barrier=true"
_SPARSE_BASE = (
    " --xla_tpu_use_tc_device_shape_on_sc=true"
    " --xla_sc_enable_instruction_fusion=false"
    " --xla_sc_disjoint_spmem=false"
    " --xla_sc_disable_megacore_partitioning=true"
)
_SPARSE_ALL_GATHER = (
    " --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false"
    " --xla_tpu_enable_sparse_core_collective_offload_all_gather=true"
    " --xla_tpu_enable_all_gather_offload_tracing=true"
) + _SPARSE_BASE
_SPARSE_ALL = (
    " --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false"
    " --xla_tpu_enable_async_collective_fusion_fuse_all_reduce=false"
    " --xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=false"
    " --xla_tpu_enable_sparse_core_collective_offload_all_gather=true"
    " --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true"
    " --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true"
    " --xla_tpu_enable_all_gather_offload_tracing=true"
    " --xla_tpu_enable_reduce_scatter_offload_tracing=true"
    " --xla_tpu_enable_all_reduce_offload_tracing=true"
) + _SPARSE_BASE

AVAILABLE_MODELS = {
    "v6e": {
        "llama3_1_70b_129024": MaxTextModel("llama3_1-70b-129024", "llama3.1-70b", {
            "per_device_batch_size": 0.125, "ici_fsdp_parallelism": -1, "ici_sequence_parallelism": 8,
            "remat_policy": "custom", "decoder_layer_input": "offload", "out_proj": "offload",
            "query_proj": "offload", "key_proj": "offload", "value_proj": "offload", "max_target_length": 129024,
            "attention": "flash", "use_iota_embed": True, "dataset_path": "gs://max-datasets-rogue",
            "dataset_type": "synthetic", "enable_checkpointing": False, "sa_block_q": 2048, "sa_block_kv": 2048,
            "sa_block_kv_compute": 2048, "sa_block_q_dkv": 2048, "sa_block_kv_dkv": 2048,
            "sa_block_kv_dkv_compute": 2048, "sa_block_q_dq": 2048, "sa_block_kv_dq": 2048,
            "sa_use_fused_bwd_kernel": True, "profiler": "xplane", "skip_first_n_steps_for_profiler": 10,
            "profiler_steps": 5, "allow_split_physical_axes": True, "custom_mesh": "hybrid_ring_32x8",
        }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _REDUCE_SCATTER + _DATA_PARALLEL + _SPARSE_ALL_GATHER
           + _HOST_OFFLOAD,
            pathways_xla_flag_options={"remove": ["--2a886c8_chip_config_name=megachip_tccontrol"],
                                       "add_server": _LAUNCH_BARRIER, "add_proxy": _LAUNCH_BARRIER,
                                       "add_worker": _LAUNCH_BARRIER}),
        "llama3_1_70b_131072": MaxTextModel("llama3_1_70b_131072", "llama3.1-70b", {
            "per_device_batch_size": 0.125, "ici_fsdp_parallelism": -1, "ici_context_parallelism": 16,
            "remat_policy": "custom", "decoder_layer_input": "offload", "out_proj": "offload",
            "query_proj": "offload", "key_proj": "offload", "value_proj": "offload", "max_target_length": 131072,
            "attention": "flash", "use_iota_embed": True, "dataset_path": "gs://max-datasets-rogue",
            "dataset_type": "synthetic", "enable_checkpointing": False, "sa_block_q": 2048, "sa_block_kv": 2048,
            "sa_block_kv_compute": 2048, "sa_block_q_dkv": 2048, "sa_block_kv_dkv": 2048,
            "sa_block_kv_dkv_compute": 2048, "sa_block_q_dq": 2048, "sa_block_kv_dq": 2048,
            "sa_use_fused_bwd_kernel": True, "profiler": "xplane", "skip_first_n_steps_for_profiler": 10,
            "profiler_steps": 5, "tokenizer_type": "tiktoken",
            "tokenizer_path": os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer_llama3.tiktoken"),
            "packing": False,
        }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _REDUCE_SCATTER + _DATA_PARALLEL + _SPARSE_ALL
           + _HOST_OFFLOAD,
            pathways_xla_flag_options={"remove": ["--2a886c8_chip_config_name=megachip_tccontrol"],
                                       "add_server": _LAUNCH_BARRIER, "add_proxy": _LAUNCH_BARRIER,
                                       "add_worker": _LAUNCH_BARRIER}),
        "custom_moe_700b": MaxTextModel("custom_moe_700b", "default", {
            "per_device_batch_size": 1, "max_target_length": 8192, "decoder_block": "mixtral",
            "base_emb_dim": 8192, "base_mlp_dim": 32768, "base_num_decoder_layers": 56, "head_dim": 256,
            "base_num_kv_heads": 8, "base_num_query_heads": 128, "vocab_size": 32000, "enable_dropout": False,
            "logits_via_embedding": False, "normalization_layer_epsilon": 1.0e-5, "rope_max_timescale": 1_000_000,
            "num_experts": 16, "num_experts_per_tok": 2, "ici_fsdp_parallelism": 16,
            "ici_expert_parallelism": 16, "remat_policy": "custom", "decoder_layer_input": "offload",
            "out_proj": "offload", "query_proj": "offload", "key_proj": "offload", "value_proj": "offload",
            "dataset_type": "synthetic", "reuse_example_batch": 1, "enable_checkpointing": False,
            "profiler": "xplane", "sa_block_q": 2048, "sa_block_kv": 2048, "sa_block_kv_compute": 2048,
            "sa_block_q_dkv": 2048, "sa_block_kv_dkv": 2048, "sa_block_kv_dkv_compute": 2048,
            "sa_block_q_dq": 2048, "sa_block_kv_dq": 2048, "sa_use_fused_bwd_kernel": True,
            "sparse_matmul": False, "capacity_factor": 1.5,
            "tokenizer_path": os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer.mistral-v1"),
            "dtype": "bfloat16", "weight_dtype": "bfloat16", "opt_type": "sgd", "attention": "flash",
        }, " --xla_tpu_scoped_vmem_limit_kib=81920" + _ALL_GATHER + _DATA_PARALLEL + _REDUCE_SCATTER + _HOST_OFFLOAD),
        "llama3_1_70b_8192_pw_lr_rd": MaxTextModel("llama3_1-70b-8192-pw-lr-rd", "llama3.1-70b", {
            "per_device_batch_size": 4, "ici_fsdp_parallelism": -1, "remat_policy": "custom",
            "decoder_layer_input": "offload", "query_proj": "offload", "key_proj": "offload",
            "value_proj": "offload", "max_target_length": 8192, "attention": "flash", "use_iota_embed": True,
            "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic", "enable_checkpointing": False,
            "sa_block_q": 2048, "sa_block_kv": 2048, "sa_block_kv_compute": 2048, "sa_block_q_dkv": 2048,
            "sa_block_kv_dkv": 2048, "sa_block_kv_dkv_compute": 2048, "sa_block_q_dq": 2048,
            "sa_block_kv_dq": 2048, "sa_use_fused_bwd_kernel": True, "profiler": "xplane",
            "skip_first_n_steps_for_profiler": 10, "profiler_steps": 5,
        }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _REDUCE_SCATTER + _DATA_PARALLEL + _ALL_GATHER + _HOST_OFFLOAD,
            pathways_tuning_params={"enable_checkpointing": True, "async_checkpointing": True,
                                    "checkpoint_period": 100, "enable_checkpoint_cloud_logger": True,
                                    "enable_goodput_recording": True},
            pathways_xla_flag_options={"remove": ["--2a886c8_chip_config_name=megachip_tccontrol"],
                                       "add_server": _LAUNCH_BARRIER, "add_proxy": _LAUNCH_BARRIER,
                                       "add_worker": _LAUNCH_BARRIER}),
        "llama3_1_70b_8192_iter_real_data_and_checkpointing_tfds": MaxTextModel(
            "llama3_1-70b-8192-iter-real-data-and-checkpointing-tfds", "llama3.1-70b", {
                "per_device_batch_size": 2, "ici_fsdp_parallelism": -1, "remat_policy": "custom",
                "decoder_layer_input": "offload", "query_proj": "offload", "key_proj": "offload",
                "value_proj": "offload", "max_target_length": 8192, "attention": "flash", "use_iota_embed": True,
                "dataset_path": "gs://trillium-scale-datasets-q1-25-west", "dataset_type": "tfds",
                "enable_checkpointing": True, "async_checkpointing": True, "checkpoint_period": 20,
                "enable_checkpoint_cloud_logger": True, "sa_block_q": 2048, "sa_block_kv": 2048,
                "sa_block_kv_compute": 2048, "sa_block_q_dkv": 2048, "sa_block_kv_dkv": 2048,
                "sa_block_kv_dkv_compute": 2048, "sa_block_q_dq": 2048, "sa_block_kv_dq": 2048,
                "sa_use_fused_bwd_kernel": True, "gcs_metrics": True, "profiler": "xplane",
                "skip_first_n_steps_for_profiler": 10, "profiler_steps": 5, "tokenizer_type": "tiktoken",
                "tokenizer_path": os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer_llama3.tiktoken"),
            }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _REDUCE_SCATTER + _DATA_PARALLEL + _ALL_GATHER
               + _HOST_OFFLOAD + _SPARSE_ALL_REDUCE + " --xla_tpu_iova_dma_chunk_size_bytes=104857",
            pathways_xla_flag_options={"remove": ["--2a886c8_chip_config_name=megachip_tccontrol"],
                                       "add_server": _LAUNCH_BARRIER, "add_proxy": _LAUNCH_BARRIER,
                                       "add_worker": _LAUNCH_BARRIER}),
        "llama3_1_70b_8192_synth": MaxTextModel("llama3_1-70b-8192-synth", "llama3.1-70b", {
            "per_device_batch_size": 2, "ici_fsdp_parallelism": -1, "remat_policy": "custom",
            "decoder_layer_input": "offload", "query_proj": "offload", "key_proj": "offload",
            "value_proj": "offload", "max_target_length": 8192, "attention": "flash", "use_iota_embed": True,
            "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic", "enable_checkpointing": True,
            "async_checkpointing": True, "checkpoint_period": 20, "enable_checkpoint_cloud_logger": True,
            "sa_block_q": 2048, "sa_block_kv": 2048, "sa_block_kv_compute": 2048, "sa_block_q_dkv": 2048,
            "sa_block_kv_dkv": 2048, "sa_block_kv_dkv_compute": 2048, "sa_block_q_dq": 2048,
            "sa_block_kv_dq": 2048, "sa_use_fused_bwd_kernel": True, "gcs_metrics": True, "profiler": "xplane",
            "skip_first_n_steps_for_profiler": 10, "profiler_steps": 5, "tokenizer_type": "tiktoken",
            "tokenizer_path": os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer_llama3.tiktoken"),
        }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _REDUCE_SCATTER + _DATA_PARALLEL + _ALL_GATHER
           + _HOST_OFFLOAD + _SPARSE_ALL_REDUCE + " --xla_tpu_iova_dma_chunk_size_bytes=104857",
            pathways_xla_flag_options={"remove": ["--2a886c8_chip_config_name=megachip_tccontrol"],
                                       "add_server": _LAUNCH_BARRIER, "add_proxy": _LAUNCH_BARRIER,
                                       "add_worker": _LAUNCH_BARRIER}),
        "llama3_1_70b_8192_synthetic": MaxTextModel("llama3_1_70b_8192_synthetic", "llama3.1-70b", {
            "per_device_batch_size": 2, "ici_fsdp_parallelism": -1, "remat_policy": "custom",
            "decoder_layer_input": "offload", "query_proj": "offload", "key_proj": "offload",
            "value_proj": "offload", "max_target_length": 8192, "attention": "flash", "use_iota_embed": True,
            "dataset_type": "synthetic", "enable_checkpointing": False, "sa_block_q": 2048,
            "sa_block_kv": 2048, "sa_block_kv_compute": 2048, "sa_block_q_dkv": 2048, "sa_block_kv_dkv": 2048,
            "sa_block_kv_dkv_compute": 2048, "sa_block_q_dq": 2048, "sa_block_kv_dq": 2048,
            "sa_use_fused_bwd_kernel": True,
        }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _REDUCE_SCATTER + _DATA_PARALLEL + _ALL_GATHER
           + _HOST_OFFLOAD + _SPARSE_ALL_REDUCE + " --xla_tpu_iova_dma_chunk_size_bytes=104857"),
        "llama3_1_70b_8192_rd_grain": MaxTextModel("llama3_1_70b_8192_rd_grain", "llama3.1-70b", {
            "per_device_batch_size": 2, "ici_fsdp_parallelism": -1, "remat_policy": "custom",
            "decoder_layer_input": "offload", "query_proj": "offload", "key_proj": "offload",
            "value_proj": "offload", "max_target_length": 8192, "attention": "flash", "use_iota_embed": True,
            "dataset_path": "/tmp/dataset", "dataset_type": "grain",
            "grain_train_files": "/tmp/dataset/array-record/c4/en/3.0.1/c4-train.array_record*",
            "grain_worker_count": 24, "enable_checkpointing": False, "sa_block_q": 2048,
            "sa_block_kv": 2048, "sa_block_kv_compute": 2048, "sa_block_q_dkv": 2048, "sa_block_kv_dkv": 2048,
            "sa_block_kv_dkv_compute": 2048, "sa_block_q_dq": 2048, "sa_block_kv_dq": 2048,
            "sa_use_fused_bwd_kernel": True,
        }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _REDUCE_SCATTER + _DATA_PARALLEL + _ALL_GATHER
           + _HOST_OFFLOAD + _SPARSE_ALL_REDUCE + " --xla_tpu_iova_dma_chunk_size_bytes=104857"),
        "llama3_1_70b_8192_synthetic_ckpt": MaxTextModel("llama3_1_70b_8192_synthetic_ckpt", "llama3.1-70b", {
            "per_device_batch_size": 2, "ici_fsdp_parallelism": -1, "remat_policy": "custom",
            "decoder_layer_input": "offload", "query_proj": "offload", "key_proj": "offload",
            "value_proj": "offload", "max_target_length": 8192, "attention": "flash", "use_iota_embed": True,
            "dataset_type": "synthetic", "enable_checkpointing": True, "async_checkpointing": True,
            "checkpoint_period": 20, "sa_block_q": 2048, "sa_block_kv": 2048, "sa_block_kv_compute": 2048,
            "sa_block_q_dkv": 2048, "sa_block_kv_dkv": 2048, "sa_block_kv_dkv_compute": 2048,
            "sa_block_q_dq": 2048, "sa_block_kv_dq": 2048, "sa_use_fused_bwd_kernel": True,
        }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _REDUCE_SCATTER + _DATA_PARALLEL + _ALL_GATHER
           + _HOST_OFFLOAD + _SPARSE_ALL_REDUCE + " --xla_tpu_iova_dma_chunk_size_bytes=104857"),
        "llama3_1_70b_8192_rd_ckpt_grain": MaxTextModel("llama3_1_70b_8192_rd_ckpt_grain", "llama3.1-70b", {
            "per_device_batch_size": 2, "ici_fsdp_parallelism": -1, "remat_policy": "custom",
            "decoder_layer_input": "offload", "query_proj": "offload", "key_proj": "offload",
            "value_proj": "offload", "max_target_length": 8192, "attention": "flash", "use_iota_embed": True,
            "dataset_path": "/tmp/dataset", "dataset_type": "grain",
            "grain_train_files": "/tmp/dataset/array-record/c4/en/3.0.1/c4-train.array_record*",
            "grain_worker_count": 24, "enable_checkpointing": True, "async_checkpointing": True,
            "checkpoint_period": 20, "sa_block_q": 2048, "sa_block_kv": 2048, "sa_block_kv_compute": 2048,
            "sa_block_q_dkv": 2048, "sa_block_kv_dkv": 2048, "sa_block_kv_dkv_compute": 2048,
            "sa_block_q_dq": 2048, "sa_block_kv_dq": 2048, "sa_use_fused_bwd_kernel": True,
        }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _REDUCE_SCATTER + _DATA_PARALLEL + _ALL_GATHER
           + _HOST_OFFLOAD + _SPARSE_ALL_REDUCE + " --xla_tpu_iova_dma_chunk_size_bytes=104857"),
        "mistral_7b": MaxTextModel("mistral-7b", "mistral-7b", {
            "per_device_batch_size": 6, "ici_fsdp_parallelism": -1, "remat_policy": "custom",
            "decoder_layer_input": "offload", "out_proj": "offload", "query_proj": "offload",
            "key_proj": "offload", "value_proj": "offload", "max_target_length": 8192, "attention": "flash",
            "use_iota_embed": True, "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic",
            "enable_checkpointing": False, "sa_block_q": 2048, "sa_block_kv": 2048, "sa_block_kv_compute": 2048,
            "sa_block_q_dkv": 2048, "sa_block_kv_dkv": 2048, "sa_block_kv_dkv_compute": 2048,
            "sa_block_q_dq": 2048, "sa_block_kv_dq": 2048, "sa_use_fused_bwd_kernel": True,
            "profiler": "xplane", "skip_first_n_steps_for_profiler": 10, "profiler_steps": 5,
        }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _REDUCE_SCATTER + _DATA_PARALLEL + _ALL_GATHER
           + _SPARSE_ALL_REDUCE + _HOST_OFFLOAD + " --xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000"),
        "mixtral_8x7b_dropless": MaxTextModel("mixtral_8x7b_dropless", "mixtral-8x7b", {
            "per_device_batch_size": 12, "ici_fsdp_parallelism": -1, "max_target_length": 4096,
            "remat_policy": "full", "attention": "flash", "gcs_metrics": True, "use_iota_embed": True,
            "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic", "reuse_example_batch": 1,
            "enable_checkpointing": False, "profiler": "xplane", "sa_block_q": 2048, "sa_block_q_dkv": 2048,
            "sa_block_q_dq": 2048, "megablox": True, "sparse_matmul": True,
        }, " --xla_tpu_scoped_vmem_limit_kib=81920" + _ALL_GATHER + _DATA_PARALLEL),
        "mixtral_8x7b_dropped": MaxTextModel("mixtral_8x7b_dropped", "mixtral-8x7b", {
            "per_device_batch_size": 12, "ici_fsdp_parallelism": -1, "max_target_length": 4096,
            "remat_policy": "custom", "decoder_layer_input": "offload", "out_proj": "offload",
            "query_proj": "offload", "key_proj": "offload", "value_proj": "offload", "attention": "flash",
            "gcs_metrics": True, "use_iota_embed": True, "dataset_path": "gs://max-datasets-rogue",
            "dataset_type": "synthetic", "reuse_example_batch": 1, "enable_checkpointing": False,
            "profiler": "xplane", "sa_block_q": 2048, "sa_block_q_dkv": 2048, "sa_block_q_dq": 2048,
            "megablox": False, "sparse_matmul": False, "capacity_factor": 1.25,
            "tokenizer_path": os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer.mistral-v1"),
        }, " --xla_tpu_scoped_vmem_limit_kib=81920" + _ALL_GATHER + _DATA_PARALLEL),
        "mixtral_8x7b_dropped_int8": MaxTextModel("mixtral_8x7b_dropped_int8", "mixtral-8x7b", {
            "per_device_batch_size": 8, "ici_fsdp_parallelism": -1, "max_target_length": 4096,
            "remat_policy": "full", "attention": "flash", "gcs_metrics": True,
            "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic", "reuse_example_batch": 1,
            "enable_checkpointing": False, "profiler": "xplane", "sa_block_q": 2048, "sa_block_q_dkv": 2048,
            "sa_block_q_dq": 2048, "megablox": False, "sparse_matmul": False, "capacity_factor": 1.25,
            "quantization": "int8",
            "tokenizer_path": os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer.mistral-v1"),
        }, " --xla_tpu_scoped_vmem_limit_kib=81920" + _ALL_GATHER + _DATA_PARALLEL),
        "mixtral_8x22b_dropped": MaxTextModel("mixtral_8x22b_dropped", "mixtral-8x22b", {
            "per_device_batch_size": 8, "max_target_length": 4096, "ici_fsdp_parallelism": 64,
            "ici_expert_parallelism": 4, "remat_policy": "custom", "decoder_layer_input": "offload",
            "out_proj": "offload", "query_proj": "offload", "key_proj": "offload", "value_proj": "offload",
            "attention": "flash", "gcs_metrics": True, "use_iota_embed": True,
            "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic", "reuse_example_batch": 1,
            "enable_checkpointing": False, "profiler": "xplane", "sa_block_q": 2048, "sa_block_q_dkv": 2048,
            "sa_block_q_dq": 2048, "megablox": False, "sparse_matmul": False, "capacity_factor": 1.25,
            "tokenizer_path": os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer.mistral-v3"),
            "dtype": "bfloat16", "weight_dtype": "bfloat16", "allow_split_physical_axes": True,
            "custom_mesh": "hybrid_ring_64x4",
        }, " --xla_tpu_scoped_vmem_limit_kib=81920" + _ALL_GATHER + _DATA_PARALLEL),
        "deepseek_v3_ep16": MaxTextModel("deepseek_v3_ep16", "deepseek3-671b", {
            "per_device_batch_size": 1, "max_target_length": 8192, "ici_fsdp_parallelism": 16,
            "ici_expert_parallelism": 16, "dcn_fsdp_parallelism": 2, "remat_policy": "custom",
            "decoder_layer_input": "offload", "gcs_metrics": True, "use_iota_embed": True,
            "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic", "reuse_example_batch": 1,
            "enable_checkpointing": False, "profiler": "xplane", "sa_block_q": 2048, "sa_block_q_dkv": 2048,
            "sa_block_q_dq": 2048, "megablox": False, "sparse_matmul": False, "capacity_factor": 1.0,
            "tokenizer_path": os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer.mistral-v3"),
            "dtype": "bfloat16", "opt_type": "sgd", "weight_dtype": "bfloat16", "attention": "flash",
        }, " --xla_tpu_scoped_vmem_limit_kib=81920" + _ALL_GATHER + _DATA_PARALLEL),
        "gemma2_9b_8192": MaxTextModel("gemma2-9b-8192", "gemma2-9b", {
            "per_device_batch_size": 3, "ici_fsdp_transpose_parallelism": 256, "remat_policy": "full",
            "max_target_length": 8192, "attention": "flash", "gcs_metrics": True, "use_iota_embed": True,
            "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic", "reuse_example_batch": 1,
            "enable_checkpointing": False, "profiler": "xplane",
            "tokenizer_path": os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer.llama2"),
            "sa_block_q": 2048, "sa_block_q_dkv": 2048, "sa_block_q_dq": 2048,
        }, "--xla_tpu_scoped_vmem_limit_kib=114688"
           + " --xla_tpu_use_minor_sharding_for_major_trivial_input=true"
           + " --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1" + _ALL_GATHER + _REDUCE_SCATTER),
        "gemma2_27b_8192": MaxTextModel("gemma2-27b-8192", "gemma2-27b", {
            "per_device_batch_size": 2, "ici_fsdp_transpose_parallelism": 256, "remat_policy": "full",
            "max_target_length": 8192, "attention": "flash", "gcs_metrics": True, "use_iota_embed": True,
            "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic", "reuse_example_batch": 1,
            "enable_checkpointing": False, "profiler": "xplane",
            "tokenizer_path": os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer.llama2"),
            "sa_block_q": 2048, "sa_block_q_dkv": 2048, "sa_block_q_dq": 2048,
        }, "--xla_tpu_scoped_vmem_limit_kib=122880"
           + " --xla_tpu_use_minor_sharding_for_major_trivial_input=true"
           + " --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1" + _ALL_GATHER + _REDUCE_SCATTER),
        "gemma3_12b_32768_v6e256": MaxTextModel("gemma3-12b-32768-v6e256", "gemma3-12b", {
            "per_device_batch_size": 1, "num_vocab_tiling": 16, "ici_fsdp_parallelism": -1,
            "remat_policy": "custom", "decoder_layer_input": "device", "query_proj": "remat",
            "key_proj": "remat", "value_proj": "remat", "max_target_length": 32768, "attention": "flash",
            "gcs_metrics": True, "use_iota_embed": True, "dataset_path": "gs://max-datasets-rogue",
            "dataset_type": "synthetic", "reuse_example_batch": 1, "enable_checkpointing": False,
            "profiler": "xplane", "skip_first_n_steps_for_profiler": 10, "profiler_steps": 2,
            "tokenizer_path": os.path.join("assets", "tokenizers", "tokenizer.gemma3"), "sa_block_q": 1024,
            "sa_block_kv": 1024, "sa_block_kv_compute": 1024, "sa_block_q_dkv": 512,
            "sa_block_kv_dkv": 2048, "sa_block_kv_dkv_compute": 512, "sa_block_q_dq": 1024,
            "sa_block_kv_dq": 1024,
        }, "--xla_tpu_scoped_vmem_limit_kib=122880"),
        "gemma3_12b_32768_2x_v6e256": MaxTextModel("gemma3-12b-32768-2x-v6e256", "gemma3-12b", {
            "per_device_batch_size": 1, "num_vocab_tiling": 16, "ici_fsdp_parallelism": 1,
            "ici_fsdp_transpose_parallelism": -1, "remat_policy": "custom", "decoder_layer_input": "device",
            "query_proj": "remat", "key_proj": "remat", "value_proj": "remat", "max_target_length": 32768,
            "attention": "flash", "gcs_metrics": True, "use_iota_embed": True,
            "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic", "reuse_example_batch": 1,
            "enable_checkpointing": False, "profiler": "xplane", "skip_first_n_steps_for_profiler": 10,
            "profiler_steps": 2, "tokenizer_path": os.path.join("assets", "tokenizers", "tokenizer.gemma3"),
            "sa_block_q": 1024, "sa_block_kv": 1024, "sa_block_kv_compute": 1024, "sa_block_q_dkv": 512,
            "sa_block_kv_dkv": 2048, "sa_block_kv_dkv_compute": 512, "sa_block_q_dq": 1024,
            "sa_block_kv_dq": 1024,
        }, "--xla_tpu_scoped_vmem_limit_kib=122880"),
        "gemma3_12b_32768_4x_v6e256": MaxTextModel("gemma3-12b-32768-4x-v6e256", "gemma3-12b", {
            "per_device_batch_size": 1, "num_vocab_tiling": 16, "ici_fsdp_parallelism": 1,
            "ici_fsdp_transpose_parallelism": -1, "remat_policy": "custom", "decoder_layer_input": "device",
            "query_proj": "remat", "key_proj": "remat", "value_proj": "remat", "max_target_length": 32768,
            "attention": "flash", "gcs_metrics": True, "use_iota_embed": True,
            "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic", "reuse_example_batch": 1,
            "enable_checkpointing": False, "profiler": "xplane", "skip_first_n_steps_for_profiler": 10,
            "profiler_steps": 2, "tokenizer_path": os.path.join("assets", "tokenizers", "tokenizer.gemma3"),
            "sa_block_q": 1024, "sa_block_kv": 1024, "sa_block_kv_compute": 1024, "sa_block_q_dkv": 512,
            "sa_block_kv_dkv": 2048, "sa_block_kv_dkv_compute": 512, "sa_block_q_dq": 1024,
            "sa_block_kv_dq": 1024,
        }, "--xla_tpu_scoped_vmem_limit_kib=122880"),
        "llama3_1_8b_8192_bs5": MaxTextModel("llama3_1-8b-8192-bs5", "llama3.1-8b", {
            "per_device_batch_size": 5, "ici_fsdp_parallelism": -1, "remat_policy": "custom",
            "decoder_layer_input": "offload", "out_proj": "offload", "query_proj": "offload",
            "key_proj": "offload", "value_proj": "offload", "max_target_length": 8192, "attention": "flash",
            "use_iota_embed": True, "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic",
            "enable_checkpointing": False, "sa_block_q": 2048, "sa_block_kv": 2048, "sa_block_kv_compute": 2048,
            "sa_block_q_dkv": 2048, "sa_block_kv_dkv": 2048, "sa_block_kv_dkv_compute": 2048,
            "sa_block_q_dq": 2048, "sa_block_kv_dq": 2048, "sa_use_fused_bwd_kernel": True,
            "profiler": "xplane", "skip_first_n_steps_for_profiler": 10, "profiler_steps": 5,
        }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _REDUCE_SCATTER + _DATA_PARALLEL + _ALL_GATHER
           + _SPARSE_ALL_REDUCE + _HOST_OFFLOAD),
        "llama3_1_8b_8192_no_collective_matmul": MaxTextModel(
            "llama3_1-8b-8192-no-collective-matmul", "llama3.1-8b", {
                "per_device_batch_size": 3, "ici_fsdp_parallelism": -1, "remat_policy": "custom",
                "decoder_layer_input": "offload", "out_proj": "offload", "query_proj": "offload",
                "key_proj": "offload", "value_proj": "offload", "max_target_length": 8192, "attention": "flash",
                "use_iota_embed": True, "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic",
                "enable_checkpointing": False, "sa_block_q": 2048, "sa_block_kv": 2048,
                "sa_block_kv_compute": 2048, "sa_block_q_dkv": 2048, "sa_block_kv_dkv": 2048,
                "sa_block_kv_dkv_compute": 2048, "sa_block_q_dq": 2048, "sa_block_kv_dq": 2048,
                "sa_use_fused_bwd_kernel": True, "profiler": "xplane", "skip_first_n_steps_for_profiler": 10,
                "profiler_steps": 5,
            }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _REDUCE_SCATTER + _DATA_PARALLEL + _ALL_GATHER
               + _SPARSE_ALL_REDUCE + _HOST_OFFLOAD + " --xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000"),
        "llama3_1_70b_8192": MaxTextModel("llama3_1-70b-8192", "llama3.1-70b", {
            "per_device_batch_size": 5, "ici_fsdp_parallelism": -1, "remat_policy": "custom",
            "decoder_layer_input": "offload", "query_proj": "offload", "key_proj": "offload",
            "value_proj": "offload", "max_target_length": 8192, "attention": "flash", "use_iota_embed": True,
            "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic", "enable_checkpointing": False,
            "sa_block_q": 2048, "sa_block_kv": 2048, "sa_block_kv_compute": 2048, "sa_block_q_dkv": 2048,
            "sa_block_kv_dkv": 2048, "sa_block_kv_dkv_compute": 2048, "sa_block_q_dq": 2048,
            "sa_block_kv_dq": 2048, "sa_use_fused_bwd_kernel": True, "profiler": "xplane",
            "skip_first_n_steps_for_profiler": 10, "profiler_steps": 5,
        }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _REDUCE_SCATTER + _DATA_PARALLEL + _ALL_GATHER + _HOST_OFFLOAD),
        "llama3_1_70b_8192_bs2": MaxTextModel("llama3_1-70b-8192-bs2", "llama3.1-70b", {
            "per_device_batch_size": 2, "ici_fsdp_parallelism": -1, "remat_policy": "custom",
            "decoder_layer_input": "offload", "query_proj": "offload", "key_proj": "offload",
            "value_proj": "offload", "max_target_length": 8192, "attention": "flash", "use_iota_embed": True,
            "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic", "enable_checkpointing": False,
            "sa_block_q": 2048, "sa_block_kv": 2048, "sa_block_kv_compute": 2048, "sa_block_q_dkv": 2048,
            "sa_block_kv_dkv": 2048, "sa_block_kv_dkv_compute": 2048, "sa_block_q_dq": 2048,
            "sa_block_kv_dq": 2048, "sa_use_fused_bwd_kernel": True, "profiler": "xplane",
            "skip_first_n_steps_for_profiler": 10, "profiler_steps": 5,
        }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _REDUCE_SCATTER + _DATA_PARALLEL + _ALL_GATHER + _HOST_OFFLOAD),
        "llama3_1_70b_8192_bs2_bfloat16_no_collective_matmul": MaxTextModel(
            "llama3_1-70b-8192-bs2-bfloat16-no-collective-matmul", "llama3.1-70b", {
                "per_device_batch_size": 2, "ici_fsdp_parallelism": -1, "remat_policy": "custom",
                "decoder_layer_input": "offload", "query_proj": "offload", "key_proj": "offload",
                "value_proj": "offload", "max_target_length": 8192, "attention": "flash", "use_iota_embed": True,
                "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic",
                "enable_checkpointing": False, "sa_block_q": 2048, "sa_block_kv": 2048,
                "sa_block_kv_compute": 2048, "sa_block_q_dkv": 2048, "sa_block_kv_dkv": 2048,
                "sa_block_kv_dkv_compute": 2048, "sa_block_q_dq": 2048, "sa_block_kv_dq": 2048,
                "sa_use_fused_bwd_kernel": True, "profiler": "xplane", "skip_first_n_steps_for_profiler": 10,
                "profiler_steps": 5, "weight_dtype": "bfloat16",
            }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _REDUCE_SCATTER + _DATA_PARALLEL + _ALL_GATHER
               + _HOST_OFFLOAD + " --xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000"),
        "llama3_1_70b_8192_bs4": MaxTextModel("llama3_1-70b-8192-bs4", "llama3.1-70b", {
            "per_device_batch_size": 4, "ici_fsdp_parallelism": -1, "remat_policy": "custom",
            "decoder_layer_input": "offload", "query_proj": "offload", "key_proj": "offload",
            "value_proj": "offload", "max_target_length": 8192, "attention": "flash", "use_iota_embed": True,
            "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic", "enable_checkpointing": False,
            "sa_block_q": 2048, "sa_block_kv": 2048, "sa_block_kv_compute": 2048, "sa_block_q_dkv": 2048,
            "sa_block_kv_dkv": 2048, "sa_block_kv_dkv_compute": 2048, "sa_block_q_dq": 2048,
            "sa_block_kv_dq": 2048, "sa_use_fused_bwd_kernel": True, "profiler": "xplane",
            "skip_first_n_steps_for_profiler": 10, "profiler_steps": 5,
        }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _REDUCE_SCATTER + _DATA_PARALLEL + _ALL_GATHER + _HOST_OFFLOAD),
        "llama3_8b_8192": MaxTextModel("llama3-8b-8192", "llama3-8b", {
            "per_device_batch_size": 8, "ici_fsdp_parallelism": -1, "remat_policy": "full",
            "max_target_length": 8192, "attention": "flash", "gcs_metrics": True, "use_iota_embed": True,
            "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic", "reuse_example_batch": 1,
            "enable_checkpointing": False, "profiler": "xplane", "sa_block_q": 1024,
            "sa_block_q_dkv": 2048, "sa_block_q_dq": 2048,
        }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _ALL_GATHER,
            pathways_xla_flag_options={"add_server": _LAUNCH_BARRIER, "add_proxy": _LAUNCH_BARRIER,
                                       "add_worker": _LAUNCH_BARRIER}),
        "llama3_70b_8192": MaxTextModel("llama3-70b-8192", "llama3-70b", {
            "per_device_batch_size": 2, "ici_fsdp_parallelism": -1, "remat_policy": "full",
            "optimizer_memory_host_offload": True, "gradient_clipping_threshold": 0, "max_target_length": 8192,
            "attention": "flash", "gcs_metrics": True, "use_iota_embed": True,
            "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic", "reuse_example_batch": 1,
            "enable_checkpointing": False, "profiler": "xplane", "sa_block_q": 1024,
            "sa_block_q_dkv": 2048, "sa_block_q_dq": 2048,
        }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _ALL_GATHER + _HOST_OFFLOAD
           + " --xla_tpu_scheduler_percent_shared_memory_limit=90",
            pathways_xla_flag_options={"remove": ["--2a886c8_chip_config_name=megachip_tccontrol"],
                                       "add_server": _LAUNCH_BARRIER, "add_proxy": _LAUNCH_BARRIER,
                                       "add_worker": _LAUNCH_BARRIER}),
        "llama3_1_405b_8192_fsdp_dcn": MaxTextModel("llama3-1-405b-8192-fsdp-dcn", "llama3.1-405b", {
            "per_device_batch_size": 1, "ici_fsdp_parallelism": 64, "ici_tensor_parallelism": 4,
            "dcn_fsdp_parallelism": 2, "allow_split_physical_axes": True, "custom_mesh": "hybrid_ring_64x4",
            "remat_policy": "custom", "decoder_layer_input": "offload", "query_proj": "offload",
            "key_proj": "offload", "value_proj": "offload", "out_proj": "offload", "max_target_length": 8192,
            "attention": "flash", "gcs_metrics": True, "use_iota_embed": True,
            "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic", "reuse_example_batch": 1,
            "enable_checkpointing": False, "profiler": "xplane", "sa_block_q": 1024,
            "sa_block_q_dkv": 2048, "sa_block_q_dq": 2048,
        }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _ALL_GATHER + _HOST_OFFLOAD,
            pathways_xla_flag_options={"remove": ["--2a886c8_chip_config_name=megachip_tccontrol"],
                                       "add_server": _LAUNCH_BARRIER, "add_proxy": _LAUNCH_BARRIER,
                                       "add_worker": _LAUNCH_BARRIER}),
        "llama3_1_405b_8192_pure_fsdp_ici": MaxTextModel("llama3-1-405b-8192-pure-fsdp-ici", "llama3.1-405b", {
            "per_device_batch_size": 1, "ici_fsdp_parallelism": 256, "dcn_fsdp_parallelism": 2,
            "remat_policy": "custom", "decoder_layer_input": "offload", "max_target_length": 8192,
            "attention": "flash", "gcs_metrics": True, "use_iota_embed": True,
            "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic", "reuse_example_batch": 1,
            "enable_checkpointing": False, "profiler": "xplane", "sa_block_q": 1024,
            "sa_block_q_dkv": 2048, "sa_block_q_dq": 2048,
        }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _ALL_GATHER + _HOST_OFFLOAD),
        "llama2_70b_4096_synthetic": MaxTextModel("llama2_70b_4096_synthetic", "llama2-70b", {
            "per_device_batch_size": 2, "ici_fsdp_parallelism": 1, "ici_fsdp_transpose_parallelism": -1,
            "ici_tensor_parallelism": 1, "remat_policy": "qkv_proj_offloaded", "max_target_length": 4096,
            "attention": "flash", "gcs_metrics": True, "use_iota_embed": True,
            "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic", "enable_checkpointing": False,
            "profiler": "xplane", "sa_block_q": 1024, "sa_block_q_dkv": 2048, "sa_block_q_dq": 2048,
        }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _ALL_GATHER),
        "llama2_70b_4096_sc": MaxTextModel("llama2-70b-4096-sc", "llama2-70b", {
            "per_device_batch_size": 3, "ici_fsdp_parallelism": 1, "ici_fsdp_transpose_parallelism": -1,
            "ici_tensor_parallelism": 1, "remat_policy": "qkv_proj_offloaded", "max_target_length": 4096,
            "attention": "flash", "gcs_metrics": True, "use_iota_embed": True,
            "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic", "enable_checkpointing": False,
            "profiler": "xplane", "sa_block_q": 1024, "sa_block_q_dkv": 2048, "sa_block_q_dq": 2048,
        }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _ALL_GATHER + _SPARSE_ALL_REDUCE,
            pathways_xla_flag_options={"remove": ["--2a886c8_chip_config_name=megachip_tccontrol"],
                                       "add_server": _LAUNCH_BARRIER, "add_proxy": _LAUNCH_BARRIER,
                                       "add_worker": _LAUNCH_BARRIER}),
        "llama2_70b_4096_sc_real_data_tfds": MaxTextModel("llama2-70b-4096-sc-real-data-tfds", "llama2-70b", {
            "per_device_batch_size": 2, "ici_fsdp_parallelism": 1, "ici_fsdp_transpose_parallelism": -1,
            "ici_tensor_parallelism": 1, "remat_policy": "qkv_proj_offloaded", "max_target_length": 4096,
            "attention": "flash", "gcs_metrics": True, "use_iota_embed": True,
            "dataset_path": "gs://trillium-storage-datasets-sr", "enable_checkpointing": False,
            "profiler": "xplane", "sa_block_q": 1024, "sa_block_q_dkv": 2048, "sa_block_q_dq": 2048,
        }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _ALL_GATHER + _SPARSE_ALL_REDUCE,
            pathways_xla_flag_options={"remove": ["--2a886c8_chip_config_name=megachip_tccontrol"],
                                       "add_server": _LAUNCH_BARRIER, "add_proxy": _LAUNCH_BARRIER,
                                       "add_worker": _LAUNCH_BARRIER}),
        "llama2_70b_4096_sc_real_data_grain": MaxTextModel("llama2-70b-4096-sc-real-data-grain", "llama2-70b", {
            "per_device_batch_size": 2, "ici_fsdp_parallelism": 1, "ici_fsdp_transpose_parallelism": -1,
            "ici_tensor_parallelism": 1, "remat_policy": "qkv_proj_offloaded", "max_target_length": 4096,
            "attention": "flash", "gcs_metrics": True, "use_iota_embed": True,
            "dataset_path": "gs://trillium-storage-datasets-sr",
            "base_output_directory": "gs://trillium-storage-tests-nov24-sr/long-run-dec11",
            "enable_checkpointing": False, "dataset_type": "grain",
            "grain_train_files": os.path.join(gettempdir(), "dataset", "array-record", "c4", "en", "3.0.1",
                                               "c4-train.array_record*"),
            "grain_worker_count": 24, "profiler": "xplane", "sa_block_q": 1024,
            "sa_block_q_dkv": 2048, "sa_block_q_dq": 2048, "profile_cleanly": False,
        }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _ALL_GATHER + _SPARSE_ALL_REDUCE,
            pathways_tuning_params={"enable_checkpointing": True, "async_checkpointing": True,
                                    "checkpoint_period": 100, "enable_checkpoint_cloud_logger": True,
                                    "enable_goodput_recording": True},
            pathways_xla_flag_options={"remove": ["--2a886c8_chip_config_name=megachip_tccontrol"],
                                       "add_server": _LAUNCH_BARRIER, "add_proxy": _LAUNCH_BARRIER,
                                       "add_worker": _LAUNCH_BARRIER}),
        "llama2_70b_4096_sc_real_data_grain_checkpoint": MaxTextModel(
            "llama2-70b-4096-sc-real-data-grain-checkpoint", "llama2-70b", {
                "per_device_batch_size": 2, "ici_fsdp_parallelism": 1, "ici_fsdp_transpose_parallelism": -1,
                "ici_tensor_parallelism": 1, "remat_policy": "qkv_proj_offloaded", "max_target_length": 4096,
                "attention": "flash", "gcs_metrics": True, "use_iota_embed": True,
                "dataset_path": "gs://trillium-storage-datasets-sr",
                "base_output_directory": "gs://trillium-storage-tests-nov24-sr/long-run-dec11",
                "checkpoint_period": 100, "enable_checkpointing": True, "async_checkpointing": True,
                "dataset_type": "grain", "grain_train_files": os.path.join(
                    gettempdir(), "dataset", "array-record", "c4", "en", "3.0.1", "c4-train.array_record*"),
                "grain_worker_count": 24, "profiler": "xplane", "sa_block_q": 1024,
                "sa_block_q_dkv": 2048, "sa_block_q_dq": 2048,
            }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _ALL_GATHER + _SPARSE_ALL_REDUCE,
            pathways_tuning_params={"enable_checkpointing": True, "async_checkpointing": True,
                                    "checkpoint_period": 100, "enable_checkpoint_cloud_logger": True,
                                    "enable_goodput_recording": True},
            pathways_xla_flag_options={"remove": ["--2a886c8_chip_config_name=megachip_tccontrol"],
                                       "add_server": _LAUNCH_BARRIER, "add_proxy": _LAUNCH_BARRIER,
                                       "add_worker": _LAUNCH_BARRIER}),
        "llama2_70b_4096_rd_lr": MaxTextModel("llama2-70b-4096-rd-lr", "llama2-70b", {
            "per_device_batch_size": 4, "ici_fsdp_parallelism": -1, "remat_policy": "full",
            "max_target_length": 4096, "attention": "flash", "gcs_metrics": True, "use_iota_embed": True,
            "reuse_example_batch": 0, "profiler": "xplane", "dataset_path": "gs://max-datasets-rogue",
            "dataset_type": "tfds", "tokenizer_path": os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer.llama2"),
            "sa_block_q": 1024, "sa_block_q_dkv": 2048, "sa_block_q_dq": 2048,
        }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _ALL_GATHER,
            pathways_tuning_params={"enable_checkpointing": True, "async_checkpointing": True,
                                    "checkpoint_period": 100, "enable_checkpoint_cloud_logger": True,
                                    "enable_goodput_recording": True},
            pathways_xla_flag_options={"add_server": _LAUNCH_BARRIER, "add_proxy": _LAUNCH_BARRIER,
                                       "add_worker": _LAUNCH_BARRIER}),
        "default_basic_1": MaxTextModel("default-basic-1", "default", {
            "per_device_batch_size": 1, "remat_policy": "full", "global_parameter_scale": 1,
            "attention": "flash", "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic",
            "reuse_example_batch": 1, "enable_checkpointing": False, "profiler": "xplane",
        }, ""),
        "default_32": MaxTextModel("default-32", "default", {
            "per_device_batch_size": 13, "ici_fsdp_parallelism": -1, "remat_policy": "full",
            "global_parameter_scale": 32, "attention": "flash", "gcs_metrics": True, "use_iota_embed": True,
            "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic", "reuse_example_batch": 1,
            "enable_checkpointing": False, "profiler": "xplane", "sa_block_q": 1024,
            "sa_block_q_dkv": 1024, "sa_block_q_dq": 2048,
        }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _ALL_GATHER),
        "default_64": MaxTextModel("default-64", "default", {
            "per_device_batch_size": 6, "ici_fsdp_parallelism": -1, "remat_policy": "full",
            "global_parameter_scale": 64, "attention": "flash", "gcs_metrics": True, "use_iota_embed": True,
            "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic", "reuse_example_batch": 1,
            "enable_checkpointing": False, "profiler": "xplane", "sa_block_q": 1024,
            "sa_block_q_dkv": 2048, "sa_block_q_dq": 2048,
        }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _ALL_GATHER),
        "default_128": MaxTextModel("default-128", "default", {
            "per_device_batch_size": 2, "ici_fsdp_parallelism": -1, "remat_policy": "full",
            "global_parameter_scale": 128, "attention": "flash", "gcs_metrics": True, "use_iota_embed": True,
            "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic", "reuse_example_batch": 1,
            "enable_checkpointing": False, "profiler": "xplane", "sa_block_q": 1024,
            "sa_block_q_dkv": 2048, "sa_block_q_dq": 2048,
        }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _ALL_GATHER),
        "default_256": MaxTextModel("default-256", "default", {
            "per_device_batch_size": 1, "ici_fsdp_parallelism": -1, "dcn_fsdp_transpose_parallelism": -1,
            "remat_policy": "full", "global_parameter_scale": 256, "attention": "flash", "gcs_metrics": True,
            "use_iota_embed": True, "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic",
            "reuse_example_batch": 1, "enable_checkpointing": False, "profiler": "xplane", "sa_block_q": 1024,
            "sa_block_q_dkv": 2048, "sa_block_q_dq": 2048,
        }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _ALL_GATHER),
        "default_512": MaxTextModel("default-512", "default", {
            "per_device_batch_size": 1, "ici_fsdp_parallelism": -1, "dcn_fsdp_parallelism": -1,
            "remat_policy": "full", "global_parameter_scale": 512, "attention": "flash", "gcs_metrics": True,
            "use_iota_embed": True, "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic",
            "reuse_example_batch": 1, "enable_checkpointing": False, "profiler": "xplane", "sa_block_q": 1024,
            "sa_block_q_dkv": 2048, "sa_block_q_dq": 2048,
        }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _ALL_GATHER),
        "gpt_3_175b": MaxTextModel("gpt-3-175b", "gpt3-175b", {
            "per_device_batch_size": 3, "ici_fsdp_parallelism": -1, "remat_policy": "full", "attention": "flash",
            "quantization": "int8", "gcs_metrics": True, "dataset_type": "synthetic", "reuse_example_batch": 1,
            "enable_checkpointing": False, "profiler": "xplane", "sa_block_q": 1024,
            "sa_block_q_dkv": 2048, "sa_block_q_dq": 2048,
        }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _ALL_GATHER + _DATA_PARALLEL
           + " --xla_tpu_use_bundle_aware_cost_model_for_fusions=false"),
        "gpt_3_175b_bf16": MaxTextModel("gpt-3-175b-bf16", "gpt3-175b", {
            "per_device_batch_size": 3, "ici_fsdp_parallelism": -1, "remat_policy": "full", "attention": "flash",
            "gcs_metrics": True, "dataset_type": "synthetic", "reuse_example_batch": 1,
            "enable_checkpointing": False, "profiler": "xplane", "sa_block_q": 1024,
            "sa_block_q_dkv": 2048, "sa_block_q_dq": 2048,
        }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _ALL_GATHER + _DATA_PARALLEL
           + " --xla_tpu_use_bundle_aware_cost_model_for_fusions=false"),
        "llama3_1_8b_8192": MaxTextModel("llama3_1-8b-8192", "llama3.1-8b", {
            "per_device_batch_size": 4, "ici_fsdp_parallelism": -1, "remat_policy": "custom",
            "decoder_layer_input": "offload", "out_proj": "offload", "query_proj": "offload",
            "key_proj": "offload", "value_proj": "offload", "max_target_length": 8192,
            "attention": "flash", "use_iota_embed": True, "dataset_path": "gs://max-datasets-rogue",
            "dataset_type": "synthetic", "enable_checkpointing": False, "sa_block_q": 2048,
            "sa_block_kv": 2048, "sa_block_kv_compute": 2048, "sa_block_q_dkv": 2048,
            "sa_block_kv_dkv": 2048, "sa_block_kv_dkv_compute": 2048, "sa_block_q_dq": 2048,
            "sa_block_kv_dq": 2048, "sa_use_fused_bwd_kernel": True, "profiler": "xplane",
            "skip_first_n_steps_for_profiler": 10, "profiler_steps": 5,
        }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _REDUCE_SCATTER + _DATA_PARALLEL + _ALL_GATHER
           + _SPARSE_ALL_REDUCE + _HOST_OFFLOAD, pathways_xla_flag_options={
               "remove": ["--2a886c8_chip_config_name=megachip_tccontrol"],
               "add_server": _LAUNCH_BARRIER, "add_proxy": _LAUNCH_BARRIER, "add_worker": _LAUNCH_BARRIER,
           }),
        "llama2_7b_4096": MaxTextModel("llama2-7b-4096", "llama2-7b", {
            "per_device_batch_size": 12, "ici_fsdp_parallelism": -1, "remat_policy": "full",
            "max_target_length": 4096, "attention": "flash", "gcs_metrics": True, "use_iota_embed": True,
            "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic", "reuse_example_batch": 1,
            "enable_checkpointing": False, "profiler": "xplane", "sa_block_q": 1024,
            "sa_block_q_dkv": 2048, "sa_block_q_dq": 2048,
        }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _ALL_GATHER),
        "llama2_70b_4096": MaxTextModel("llama2-70b-4096", "llama2-70b", {
            "per_device_batch_size": 4, "ici_fsdp_parallelism": 1, "ici_fsdp_transpose_parallelism": -1,
            "ici_tensor_parallelism": 1, "remat_policy": "full", "max_target_length": 4096,
            "attention": "flash", "gcs_metrics": True, "use_iota_embed": True,
            "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic", "reuse_example_batch": 1,
            "enable_checkpointing": False, "profiler": "xplane", "sa_block_q": 1024,
            "sa_block_q_dkv": 2048, "sa_block_q_dq": 2048,
        }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _ALL_GATHER),
    },
    "v5litepod": {},
    "v5p": {
        "llama2_70b_v5p_128": MaxTextModel("llama2_70b_v5p_128", "llama2-70b", {
            "ici_fsdp_parallelism": -1, "per_device_batch_size": 2, "remat_policy": "save_dot_except_mlpwi",
            "max_target_length": 4096, "use_iota_embed": True, "dataset_path": "gs://max-datasets-rogue",
            "dataset_type": "synthetic", "reuse_example_batch": 1, "enable_checkpointing": False,
            "profiler": "xplane",
        }, _DATA_PARALLEL + _ALL_GATHER),
        "llama4_scout_dropless_v5p_256": MaxTextModel("llama4_scout_dropless_v5p_256", "llama4-17b-16e", {
            "per_device_batch_size": 8, "max_target_length": 8192, "ici_fsdp_parallelism": -1,
            "enable_checkpointing": False, "dtype": "bfloat16", "weight_dtype": "float32", "megablox": True,
            "sparse_matmul": True, "dataset_type": "synthetic", "opt_type": "adamw",
            "skip_first_n_steps_for_profiler": 5, "profiler_steps": 3, "profiler": "xplane",
            "remat_policy": "custom", "decoder_layer_input": "offload", "reuse_example_batch": 1,
            "sa_block_q": 2048, "sa_block_kv": 2048, "sa_block_kv_compute": 2048, "sa_block_q_dkv": 2048,
            "sa_block_kv_dkv": 2048, "sa_block_kv_dkv_compute": 2048, "sa_block_q_dq": 2048,
            "sa_block_kv_dq": 2048, "tokenizer_path": "meta-llama/Llama-4-Scout-17B-16E",
        }, " --xla_tpu_scoped_vmem_limit_kib=81920" + _ALL_GATHER + _DATA_PARALLEL + _REDUCE_SCATTER + _HOST_OFFLOAD),
        "deepseek_v3_ep_256_v5p_512": MaxTextModel("deepseek_v3_ep_256_v5p_512", "deepseek3-671b", {
            "per_device_batch_size": 4, "max_target_length": 8192, "ici_fsdp_parallelism": 1,
            "ici_expert_parallelism": -1, "remat_policy": "custom", "decoder_layer_input": "offload",
            "gcs_metrics": True, "use_iota_embed": True, "dataset_path": "gs://max-datasets-rogue",
            "dataset_type": "synthetic", "reuse_example_batch": 1, "enable_checkpointing": False,
            "skip_first_n_steps_for_profiler": 5, "profiler_steps": 5, "profiler": "xplane",
            "sa_block_q": 2048, "sa_block_q_dkv": 2048, "sa_block_q_dq": 2048, "megablox": False,
            "sparse_matmul": False, "capacity_factor": 1.0, "tokenizer_type": "huggingface",
            "tokenizer_path": "deepseek-ai/DeepSeek-V3", "dtype": "bfloat16", "opt_type": "adam_pax",
            "attention": "flash",
        }, " --xla_tpu_scoped_vmem_limit_kib=81920" + _ALL_GATHER + _DATA_PARALLEL),
    },
}

C4_MODELS = {
    "deepseek_v3_ep_256_v5p_512_c4mlperf": MaxTextModel(
        "deepseek_v3_ep_256_v5p_512-c4mlperf", "deepseek3-671b",
        c4_deepseek_v3_ep_256_v5p_512_gbs_1024(),
        " --xla_tpu_scoped_vmem_limit_kib=81920" + _ALL_GATHER + _DATA_PARALLEL,
    ),
}

for scale, batch in ((16, 6), (32, 4), (64, 2)):
  name = f"default-{scale}b-v5e-256"
  AVAILABLE_MODELS["v5litepod"][name.replace("-", "_")] = MaxTextModel(name, "default", {
      "per_device_batch_size": batch, "remat_policy": "full", "global_parameter_scale": scale,
      "max_target_length": 2048, "attention": "flash", "use_iota_embed": True,
      "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic", "reuse_example_batch": 1,
      "enable_checkpointing": False, "profiler": "xplane", "use_qwix_quantization": True,
  }, _DATA_PARALLEL + _ALL_GATHER)

AVAILABLE_MODELS["v5litepod"]["default_128b_v5e_256"] = MaxTextModel("default-128b-v5e-256", "default", {
    "ici_fsdp_parallelism": -1, "ici_tensor_parallelism": 16, "per_device_batch_size": 1,
    "remat_policy": "qkv_proj_offloaded", "global_parameter_scale": 128, "max_target_length": 2048,
    "attention": "flash", "use_iota_embed": True, "fused_qkv": True, "fused_mlp": True,
    "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic", "reuse_example_batch": 1,
    "enable_checkpointing": False, "profiler": "xplane", "use_qwix_quantization": True,
}, _DATA_PARALLEL + _ALL_GATHER)
AVAILABLE_MODELS["v5litepod"]["gpt_3_175b_v5e_256"] = MaxTextModel("gpt-3-175b-v5e-256", "gpt3-175b", {
    "ici_fsdp_parallelism": -1, "ici_tensor_parallelism": 16, "per_device_batch_size": 0.5,
    "remat_policy": "full", "max_target_length": 2048, "attention": "flash", "dataset_type": "synthetic",
    "reuse_example_batch": 1, "enable_checkpointing": False, "profiler": "xplane", "use_qwix_quantization": True,
}, _DATA_PARALLEL + _ALL_GATHER)

for size, batch, remat in ((7, 4, "save_qkv_proj"), (13, 6, "qkv_proj_offloaded"), (70, 2, "qkv_proj_offloaded")):
  name = f"llama2-{size}b-v5e-256"
  tuning = {
      "ici_fsdp_parallelism": -1, "per_device_batch_size": batch, "remat_policy": remat,
      "max_target_length": 2048, "use_iota_embed": True, "dataset_path": "gs://max-datasets-rogue",
      "dataset_type": "synthetic", "reuse_example_batch": 1, "enable_checkpointing": False,
      "profiler": "xplane", "use_qwix_quantization": True,
  }
  if size != 70:
    tuning["tokenizer_path"] = os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer.llama2")
  AVAILABLE_MODELS["v5litepod"][name.replace("-", "_")] = MaxTextModel(name, f"llama2-{size}b", tuning,
                                                                          _DATA_PARALLEL + _ALL_GATHER)

_v5e_llama = AVAILABLE_MODELS["v6e"]["llama3_1_8b_8192"]
AVAILABLE_MODELS["v5litepod"]["llama3_1_8b_8192_v5e_256"] = MaxTextModel(
    "llama3_1-8b-8192-v5e-256", _v5e_llama.model_type,
    {**_v5e_llama.tuning_params, "per_device_batch_size": 2, "use_qwix_quantization": True},
    _v5e_llama.xla_flags, pathways_xla_flag_options=_v5e_llama.pathways_xla_flag_options.copy(),
)

AVAILABLE_MODELS["v5p"].update({
    "llama3_1_405b_8192_v5p_1024": MaxTextModel("llama3-1-405b-8192-v5p-1024", "llama3.1-405b", {
        "per_device_batch_size": 2, "ici_fsdp_parallelism": -1, "max_target_length": 8192,
        "attention": "flash", "gcs_metrics": True, "use_iota_embed": True, "remat_policy": "custom",
        "context": "offload", "mlpwo": "offload", "num_vocab_tiling": 4, "sa_block_q": 2048,
        "sa_block_kv": 2048, "sa_block_kv_compute": 2048, "sa_block_q_dkv": 2048, "sa_block_kv_dkv": 2048,
        "sa_block_kv_dkv_compute": 2048, "sa_block_q_dq": 2048, "sa_block_kv_dq": 2048,
        "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic", "reuse_example_batch": 1,
        "enable_checkpointing": False, "profiler": "xplane", "skip_first_n_steps_for_profiler": 5,
        "profiler_steps": 5,
    }, " --xla_tpu_scoped_vmem_limit_kib=98304" + _ALL_GATHER + _HOST_OFFLOAD),
    "llama4_maverick_dropless_v5p_256": MaxTextModel("llama4_maverick_dropless_v5p_256", "llama4-17b-128e", {
        **AVAILABLE_MODELS["v5p"]["llama4_scout_dropless_v5p_256"].tuning_params,
        "per_device_batch_size": 4, "ici_fsdp_parallelism": 32, "ici_tensor_parallelism": 4,
        "out_proj": "offload", "query_proj": "offload", "key_proj": "offload", "value_proj": "offload",
        "tokenizer_path": "meta-llama/Llama-4-Maverick-17B-128E",
    }, " --xla_tpu_scoped_vmem_limit_kib=81920" + _ALL_GATHER + _DATA_PARALLEL + _REDUCE_SCATTER + _HOST_OFFLOAD),
    "llama2_7b_v5p_128": MaxTextModel("llama2_7b_v5p_128", "llama2-7b", {
        "ici_fsdp_parallelism": -1, "per_device_batch_size": 4, "remat_policy": "minimal",
        "max_target_length": 4096, "use_iota_embed": True,
        "tokenizer_path": os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer.llama2"),
        "dataset_path": "gs://max-datasets-rogue", "dataset_type": "synthetic", "reuse_example_batch": 1,
        "enable_checkpointing": False, "profiler": "xplane",
    }, _DATA_PARALLEL + _ALL_GATHER),
    "deepseek3_671b_v5p_1024": MaxTextModel("deepseek3_671b_v5p_1024", "deepseek3-671b", {
        "ici_fsdp_parallelism": -1, "ici_tensor_parallelism": 1, "per_device_batch_size": 6,
        "max_target_length": 8192, "dtype": "bfloat16", "weight_dtype": "float32", "megablox": True,
        "sparse_matmul": True, "scan_layers": True, "use_custom_sort_vjp": True, "sa_block_q": 2048,
        "sa_block_kv": 2048, "sa_block_kv_compute": 2048, "sa_block_q_dkv": 2048, "sa_block_kv_dkv": 2048,
        "sa_block_kv_dkv_compute": 2048, "sa_block_q_dq": 2048, "sa_block_kv_dq": 2048,
        "remat_policy": "custom", "decoder_layer_input": "offload", "out_proj": "offload",
        "tile_batch_seq": 512, "tile_embed_dim": 1024, "tile_mlp_dim": 1024,
        "sa_use_fused_bwd_kernel": False, "dataset_type": "synthetic", "enable_checkpointing": False,
        "profiler": "xplane", "skip_first_n_steps_for_profiler": 5, "profiler_steps": 5,
    }, _SPARSE_ALL + _HOST_OFFLOAD + _REDUCE_SCATTER + _DATA_PARALLEL
       + " --xla_tpu_scoped_vmem_limit_kib=81920 --xla_tpu_megacore_fusion_allow_ags=false"
       + " --xla_enable_async_collective_permute=true"),
})

for suffix, flags in (("", _ALL_GATHER), ("_sc", _SPARSE_ALL_GATHER)):
  name = f"gpt_3_175b_v5p_128{suffix}"
  AVAILABLE_MODELS["v5p"][name] = MaxTextModel(name, "gpt3-175b", {
      "ici_fsdp_parallelism": -1, "ici_tensor_parallelism": 2, "per_device_batch_size": 0.5,
      "allow_split_physical_axes": True, "remat_policy": "save_dot_with_context_except_mlp",
      "max_target_length": 4096, "attention": "flash", "dataset_type": "synthetic", "reuse_example_batch": 1,
      "enable_checkpointing": False, "profiler": "xplane",
  }, _DATA_PARALLEL + flags)
