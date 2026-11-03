"""C4 convergence benchmark parameters and the active DeepSeek V3 recipe."""

from .convergence_utils import ConvHParams, DatasetHParams, setup_model_convergence

c4_mlperf_hp = DatasetHParams(
    name="c4mlperf", dataset_path="gs://max-datasets-rogue", dataset_name="c4/en:3.0.7",
    eval_dataset_name="c4/en:3.0.9", dataset_type="c4_mlperf", train_split="train2",
    eval_split="validation", eval_tokens=47185920, add_bos=False, add_eos=False,
)
c4_en_hp = DatasetHParams(
    name="c4en", dataset_path="gs://maxtext-dataset", dataset_name="c4/en:3.0.1",
    dataset_type="grain", grain_file_type="tfrecord", train_split="train", eval_split="validation",
    eval_tokens=75497472, add_bos=False, add_eos=False,
)
c4_mutil_hp = DatasetHParams(
    name="c4multi", dataset_path="gs://mlperf-llm-public2", dataset_name="c4/multilingual:3.1.0",
    dataset_type="grain", grain_file_type="tfrecord", train_split="en", eval_split="en-validation",
    eval_tokens=824 * 512, add_bos=False, add_eos=False,
)
llama3_405b_hp = ConvHParams(
    global_batch_size=1152, warmup_samples=8216000, decay_end_samples=1382400000.0,
    total_tokens_to_train=2.64e9, training_scaleing_factor=1.0, learning_rate=6.944e-8,
    eval_tokens=47185920, eval_interval=377487360,
)
deepseek_671b_hp = ConvHParams(
    global_batch_size=1152, warmup_samples=8216000, decay_end_samples=1382400000.0,
    total_tokens_to_train=2.64e9, training_scaleing_factor=1.0, learning_rate=6.944e-8,
    eval_tokens=47185920, eval_interval=377487360,
)

DEEPSEEK_V3_LIBTPU_INIT_ARGS = (
    " --xla_tpu_scoped_vmem_limit_kib=81920"
    " --xla_tpu_enable_async_collective_fusion=true"
    " --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true"
    " --xla_tpu_enable_async_collective_fusion_multiple_steps=true"
    " --xla_tpu_overlap_compute_collective_tc=true"
    " --xla_enable_async_all_gather=true"
    " --xla_tpu_enable_data_parallel_all_reduce_opt=true"
    " --xla_tpu_data_parallel_opt_different_sized_ops=true"
)


def c4_deepseek_v3_ep_256_v5p_512_gbs_1024():
  tuning_params = {
      "per_device_batch_size": 4, "max_target_length": 8192, "ici_fsdp_parallelism": 1,
      "ici_expert_parallelism": -1, "remat_policy": "custom", "decoder_layer_input": "offload",
      "gcs_metrics": True, "use_iota_embed": True, "dataset_path": "gs://max-datasets-rogue",
      "dataset_type": "synthetic", "reuse_example_batch": 1, "enable_checkpointing": False,
      "skip_first_n_steps_for_profiler": 5, "profiler_steps": 5, "profiler": "xplane",
      "sa_block_q": 2048, "sa_block_q_dkv": 2048, "sa_block_q_dq": 2048,
      "megablox": False, "sparse_matmul": False, "capacity_factor": 1.0,
      "tokenizer_type": "huggingface", "tokenizer_path": "deepseek-ai/DeepSeek-V3",
      "dtype": "bfloat16", "opt_type": "adam_pax", "attention": "flash",
  }
  return setup_model_convergence(tuning_params, c4_mlperf_hp, deepseek_671b_hp, 256, 1024)
