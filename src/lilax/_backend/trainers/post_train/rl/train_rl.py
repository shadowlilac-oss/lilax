import dataclasses
import functools
import inspect
import json
import logging
import os
from typing import Any, Callable, Optional

import datasets
import grain
import jax
from absl import logging as absl_logging
from etils import epath
from flax import nnx
from jax.sharding import Mesh
from orbax import checkpoint as ocp
from transformers import AutoTokenizer
from tunix.perf import export as perf_export, metrics as perf_metrics
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.agentic import agentic_grpo_learner
from tunix.rl.grpo import grpo_learner
from tunix.rl.metrics import mldiag_backend
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger

from lilax._backend.common.common_types import DecoderBlockType
from lilax._backend.common.goodput import (
    GoodputEvent, RECORD_JOB_END_TIME, RECORD_JOB_START_TIME, create_goodput_recorder,
    maybe_monitor_goodput, maybe_record_goodput, record_goodput,
)
from lilax._backend.common.managed_mldiagnostics import ManagedMLDiagnostics
from lilax._backend.configs.pyconfig import _lists_to_tuples
from lilax._backend.configs.rl_fields import RL
from lilax._backend.input_pipeline.instruction_data_processing import load_chat_template_from_file, load_data_template_from_file
from lilax._backend.integration.tunix.tunix_adapter import TunixMaxTextAdapter
from lilax._backend.integration.vllm import maxtext_vllm_adapter
from lilax._backend.integration.vllm.maxtext_vllm_rollout import MaxTextVllmRollout
from lilax._backend.trainers.post_train.checkpoint_selection import restore_trainer_step
from lilax._backend.trainers.post_train.rl import utils_rl
from lilax._backend.trainers.post_train.rl.evaluate_rl import evaluate
from lilax._backend.distributed import device_mesh
from lilax._backend.utils import max_logging
from lilax._backend.trainers.post_train.rl.setup.rollout_parallelism import get_rollout_kwargs_for_parallelism
from lilax._backend.utils.globals import MAXTEXT_CONFIGS_DIR, MAXTEXT_PKG_DIR
from lilax.configuration.loading import load_config_values
from lilax.trainers.post_training.openenv.factory import FactoryAgent, FactoryEnvironment
from lilax.trainers.post_training.openenv.learner import FactoryGRPOLearner
from lilax.trainers.post_training.openenv.parser import QwenFactoryParser
from lilax.trainers.post_training.openenv.registry import EnvironmentRegistry


_RECURRENT_ROLLOUT_DECODER_BLOCKS = frozenset((DecoderBlockType.QWEN3_NEXT, DecoderBlockType.QWEN3_5))


def _template_path(path):
  prefix = "maxtext/examples/chat_templates/"
  if path and path.startswith(prefix):
    return os.path.join(MAXTEXT_PKG_DIR, "examples", "chat_templates", path[len(prefix):])
  return path


def rollout_prefix_caching_enabled(trainer_config: Any) -> bool:
  enabled = bool(getattr(trainer_config, "enable_prefix_caching", False))
  if enabled and getattr(trainer_config, "decoder_block", None) in _RECURRENT_ROLLOUT_DECODER_BLOCKS:
    max_logging.log(
        "Disabling vLLM prefix caching for the hybrid recurrent rollout model: "
        "GDN state is not part of the MaxText adapter's prefix cache."
    )
    return False
  return enabled


def install_pathways_multihost_profiler_patch(max_hosts: int) -> None:
  if max_hosts <= 1:
    return
  orig_start_trace = jax.profiler.start_trace
  if getattr(orig_start_trace, "_is_pathways_multihost_patched", False):
    orig_start_trace._max_hosts = max_hosts
    return
  try:
    if "max_num_hosts" not in inspect.signature(orig_start_trace).parameters:
      max_logging.warning(
          "rl.profiler_max_num_hosts > 1 requested, but jax.profiler.start_trace "
          "does not accept max_num_hosts. Skipping monkeypatch."
      )
      return
  except (ValueError, TypeError):
    return

  @functools.wraps(orig_start_trace)
  def start_trace_wrapper(*args, **kwargs):
    if "max_num_hosts" not in kwargs and len(args) <= 4:
      kwargs["max_num_hosts"] = getattr(start_trace_wrapper, "_max_hosts", max_hosts)
    return orig_start_trace(*args, **kwargs)

  start_trace_wrapper._is_pathways_multihost_patched = True
  start_trace_wrapper._max_hosts = max_hosts
  jax.profiler.start_trace = start_trace_wrapper
  jax_src_profiler = getattr(getattr(jax, "_src", None), "profiler", None)
  if jax_src_profiler is not None:
    jax_src_profiler.start_trace = start_trace_wrapper


def get_dataset(tmvp_config: Any, split: str = "train", data_files: Optional[str] = None, dataset_name: Optional[str] = None) -> grain.MapDataset:
  if data_files is None:
    data = datasets.load_dataset(dataset_name, name=tmvp_config.hf_subset, split=split)
  else:
    data = datasets.load_dataset("parquet", data_files={split: data_files}, split=split)
  if tmvp_config.debug:
    max_logging.log(f"Loaded Hugging Face dataset {dataset_name} with split {split}. Size: {len(data)}")
  return data


def prepare_train_and_eval_dataset(trainer_config: Any, test_size: float = 0.05) -> dict[str, datasets.Dataset]:
  max_logging.log(
      "WARNING: For reproducible experiments, preprocess the dataset once and "
      "define your own HfDataset subclass that directly uses the preprocessed datasets."
  )
  original_ds = get_dataset(
      trainer_config, split=trainer_config.train_split, data_files=trainer_config.hf_train_files,
      dataset_name=trainer_config.dataset_name,
  )
  if "OpenMathReasoning" in trainer_config.dataset_name:
    original_ds = original_ds.filter(lambda row: row.get("problem_type") == "has_answer_extracted")
  split_ds = original_ds.train_test_split(test_size=test_size, seed=trainer_config.data_shuffle_seed)
  return {"train": split_ds["train"], "validation": split_ds["test"]}


def prepare_datasets(
    trainer_config: Any, model_tokenizer: AutoTokenizer, *, train_dataset=None, eval_dataset=None,
    prepare_train=True, prepare_eval=True,
) -> tuple[grain.IterDataset | None, grain.IterDataset | None]:
  template_config = load_data_template_from_file(_template_path(trainer_config.data_template_path))
  if template_config is None:
    raise ValueError(f"Data template is required for processing dataset but failed to load from {trainer_config.data_template_path}")
  custom_processor_path = getattr(trainer_config, "dataset_processor_path", "") or ""
  if custom_processor_path:
    process_data = utils_rl.load_custom_callable(custom_processor_path, "process_data")
    max_logging.log(f"prepare_datasets: using custom process_data from {custom_processor_path}")
  else:
    process_data = utils_rl.process_data
  eval_dataset_name = getattr(trainer_config, "eval_dataset_name", None)
  test_dataset = None
  if train_dataset is None and eval_dataset is None and trainer_config.dataset_name in [
      "nvidia/OpenMathInstruct-2", "nvidia/OpenMathReasoning", "open-r1/OpenR1-Math-220k", "bethgelab/CuratedThoughts",
  ] and eval_dataset_name == trainer_config.dataset_name:
    splits = prepare_train_and_eval_dataset(trainer_config)
    if prepare_train:
      train_dataset = grain.MapDataset.source(splits["train"]).shuffle(seed=trainer_config.data_shuffle_seed).map(
          lambda row: process_data(trainer_config.dataset_name, model_tokenizer, template_config, trainer_config, row)
      )
    if prepare_eval and trainer_config.num_test_batches > 0:
      test_dataset = grain.MapDataset.source(splits["validation"]).shuffle(seed=trainer_config.data_shuffle_seed).map(
          lambda row: process_data(trainer_config.dataset_name, model_tokenizer, template_config, trainer_config, row)
      )
  else:
    if not eval_dataset_name:
      eval_dataset_name = trainer_config.dataset_name
    if prepare_train:
      if train_dataset is None:
        train_dataset = get_dataset(
            trainer_config, split=trainer_config.train_split, data_files=trainer_config.hf_train_files,
            dataset_name=trainer_config.dataset_name,
        )
      train_dataset = grain.MapDataset.source(train_dataset).shuffle(seed=trainer_config.data_shuffle_seed).map(
          lambda row: process_data(trainer_config.dataset_name, model_tokenizer, template_config, trainer_config, row)
      )
    if prepare_eval and trainer_config.num_test_batches > 0:
      test_dataset = eval_dataset
      if test_dataset is None:
        test_dataset = get_dataset(
            trainer_config, split=trainer_config.eval_split, data_files=trainer_config.hf_eval_files,
            dataset_name=eval_dataset_name,
        )
      test_dataset = grain.MapDataset.source(test_dataset).shuffle(seed=trainer_config.data_shuffle_seed).map(
          lambda row: process_data(eval_dataset_name, model_tokenizer, template_config, trainer_config, row)
      )

  def filter_long_prompts(row):
    return len(model_tokenizer.tokenize(row["prompts"])) <= trainer_config.max_prefill_predict_length

  if prepare_train:
    train_dataset = train_dataset.filter(filter_long_prompts)
    if getattr(trainer_config.rl, "use_agentic_rollout", False):
      def use_raw_prompt(row):
        row["prompts"] = row["question"]
        return row
      train_dataset = train_dataset.map(use_raw_prompt)
    dataset_size = int(trainer_config.num_batches * trainer_config.batch_size * trainer_config.train_fraction)
    train_dataset = train_dataset[:dataset_size].repeat(trainer_config.num_epoch)
    train_dataset = train_dataset.to_iter_dataset().batch(trainer_config.batch_size, drop_remainder=True)
  if prepare_eval and trainer_config.num_test_batches > 0:
    eval_batch_size = trainer_config.batch_size if getattr(trainer_config, "eval_batch_size", -1) <= 0 else trainer_config.eval_batch_size
    test_dataset = test_dataset.filter(filter_long_prompts)
    test_dataset = test_dataset[trainer_config.test_batch_start_index:trainer_config.num_test_batches * eval_batch_size]
    test_dataset = test_dataset.to_iter_dataset().batch(eval_batch_size, drop_remainder=True)
  return train_dataset, test_dataset


def build_reward_fns(trainer_config: Any, make_reward_fn: Callable) -> list:
  custom_rewards_path = getattr(trainer_config, "reward_functions_path", "") or ""
  custom_rewards_names = getattr(trainer_config, "reward_functions", "") or ""
  if custom_rewards_path and custom_rewards_names:
    names = [name.strip() for name in custom_rewards_names.split(",") if name.strip()]
    reward_fns = [
        make_reward_fn(utils_rl.load_custom_callable(custom_rewards_path, name, importable_by_workers=True)) for name in names
    ]
    max_logging.log(f"reward_fns: using {len(reward_fns)} custom reward function(s) {names} from {custom_rewards_path}")
    return reward_fns
  return [make_reward_fn(utils_rl.match_format_exactly), make_reward_fn(utils_rl.match_format_approximately),
          make_reward_fn(utils_rl.check_numbers)]


class _RewardFn:
  def __init__(self, fn: Callable, tmvp_config: Any):
    functools.update_wrapper(self, fn)
    self.fn = fn
    self.tmvp_config = tmvp_config

  def __call__(self, **kwargs: Any) -> Any:
    return self.fn(tmvp_config=self.tmvp_config, **kwargs)


def build_metrics_logging_options(trainer_config: Any, metric_backend=None, report_to=None) -> metrics_logger.MetricsLoggerOptions | None:
  backends = []
  if trainer_config.managed_mldiagnostics:
    backends.append(mldiag_backend.MLDiagScalarBackend)
    if trainer_config.enable_tensorboard and report_to is None and metric_backend is None:
      backends.append(functools.partial(
          metrics_logger.TensorboardBackend, log_dir=trainer_config.tensorboard_dir,
          flush_every_n_steps=trainer_config.log_period,
      ))
  if metric_backend is not None:
    backends.append(lambda: metric_backend)
  if report_to is not None and not backends:
    return None
  return metrics_logger.MetricsLoggerOptions(
      log_dir=trainer_config.tensorboard_dir, flush_every_n_steps=trainer_config.log_period,
      backend_kwargs={"custom_backend": backends},
  )


def build_rl_profiler_config(trainer_config: Any) -> rl_cluster_lib.RLProfileConfig | None:
  if trainer_config.profiler != "xplane":
    return None
  rl_cfg = getattr(trainer_config, "rl", None)
  rl_fields = getattr(RL, "model_fields")
  start_invocation = getattr(rl_cfg, "profiler_start_invocation", getattr(
      trainer_config, "skip_first_n_steps_for_profiler", rl_fields["profiler_start_invocation"].default
  ))
  num_invocations = getattr(rl_cfg, "profiler_num_invocations", getattr(
      trainer_config, "profiler_steps", rl_fields["profiler_num_invocations"].default
  ))
  timeout_secs = getattr(rl_cfg, "profiler_timeout_secs", rl_fields["profiler_timeout_secs"].default)
  return rl_cluster_lib.RLProfileConfig(
      start_invocation=start_invocation, num_invocations=num_invocations, output_dir=trainer_config.tensorboard_dir,
      managed_mldiagnostics=trainer_config.managed_mldiagnostics, timeout_secs=timeout_secs,
  )


def create_rl_components(
    trainer_config, sampler_config, sampler_devices, actor_model, actor_mesh, reference_model, reference_mesh,
    rollout_mesh, model_tokenizer, *, vllm_config, checkpoint_step=None, metric_backend=None, report_to=None,
    agent_class=None, agent_kwargs=None, env_class=None, env_kwargs=None, reward_funcs=None, environment_factory=None,
):
  optimizer = utils_rl.get_optimizer(trainer_config)
  if trainer_config.enable_checkpointing:
    checkpointing_options = ocp.CheckpointManagerOptions(
        save_interval_steps=trainer_config.checkpoint_period, max_to_keep=trainer_config.max_num_checkpoints_to_keep
    )
    checkpoint_dir = trainer_config.checkpoint_dir
  else:
    checkpointing_options = None
    checkpoint_dir = None
  if checkpoint_step is not None and checkpoint_dir is None:
    raise ValueError("Checkpointing must be enabled to restore a selected RL checkpoint.")
  train_micro_batch_size = None if trainer_config.train_micro_batch_size == -1 else trainer_config.train_micro_batch_size
  rollout_micro_batch_size = None if trainer_config.rollout_micro_batch_size == -1 else trainer_config.rollout_micro_batch_size
  metrics_logging_options = build_metrics_logging_options(trainer_config, metric_backend, report_to)
  rl_profiler_config = build_rl_profiler_config(trainer_config)
  rollout_additional_config = None
  if trainer_config.vllm_additional_config:
    if isinstance(trainer_config.vllm_additional_config, dict):
      rollout_additional_config = trainer_config.vllm_additional_config
    elif isinstance(trainer_config.vllm_additional_config, str):
      try:
        rollout_additional_config = json.loads(trainer_config.vllm_additional_config)
      except json.JSONDecodeError as error:
        raise ValueError(f"Failed to parse additional_config JSON: {error}") from error
  rl_rollout_engine = functools.partial(MaxTextVllmRollout, maxtext_config=trainer_config)
  rollout_vllm_kwargs = {
      "hf_overrides": trainer_config.vllm_hf_overrides,
      "enable_expert_parallel": sampler_config.enable_expert_parallel,
      "enable_prefix_caching": rollout_prefix_caching_enabled(trainer_config),
      "dtype": trainer_config.weight_dtype.value,
  }
  if trainer_config.vllm_block_size is not None:
    rollout_vllm_kwargs["block_size"] = trainer_config.vllm_block_size
  cluster_config = rl_cluster_lib.ClusterConfig(
      role_to_mesh={
          rl_cluster_lib.Role.ACTOR: actor_mesh, rl_cluster_lib.Role.REFERENCE: reference_mesh,
          rl_cluster_lib.Role.ROLLOUT: rollout_mesh,
      },
      role_to_logical_axis_rule={
          rl_cluster_lib.Role.ACTOR: trainer_config.logical_axis_rules,
          rl_cluster_lib.Role.REFERENCE: trainer_config.logical_axis_rules,
          rl_cluster_lib.Role.ROLLOUT: (
              _lists_to_tuples(vllm_config["logical_axis_rules"])
              if isinstance(vllm_config, dict) else vllm_config.logical_axis_rules
          ),
      },
      rollout_engine=rl_rollout_engine, offload_to_cpu=False,
      gc_collect_after_weight_sync=trainer_config.gc_collect_after_weight_sync,
      training_config=rl_cluster_lib.RLTrainingConfig(
          actor_optimizer=optimizer, eval_every_n_steps=trainer_config.eval_interval, max_steps=trainer_config.train_steps,
          mini_batch_size=trainer_config.batch_size, train_micro_batch_size=train_micro_batch_size,
          rollout_micro_batch_size=rollout_micro_batch_size, max_seq_token_per_tpu=trainer_config.max_seq_token_per_tpu or None,
          metrics_logging_options=metrics_logging_options, profiler_options=None, rl_profiler_config=rl_profiler_config,
          checkpoint_root_directory=None if checkpoint_step is not None else checkpoint_dir,
          checkpointing_options=checkpointing_options,
      ),
      rollout_config=base_rollout.RolloutConfig(
          max_tokens_to_generate=trainer_config.max_target_length - trainer_config.max_prefill_predict_length,
          max_prompt_length=trainer_config.max_prefill_predict_length,
          kv_cache_size=trainer_config.max_target_length + trainer_config.kv_cache_buffer,
          temperature=trainer_config.decode_sampling_temperature, top_p=trainer_config.decode_sampling_nucleus_p,
          top_k=trainer_config.decode_sampling_top_k, rollout_vllm_model_version=trainer_config.tokenizer_path,
          rollout_vllm_hbm_utilization=trainer_config.hbm_utilization_vllm,
          rollout_vllm_tpu_backend_type=getattr(
              trainer_config, "vllm_tpu_backend_type",
              "maxtext" if "MaxText" in str(trainer_config.vllm_hf_overrides.get("architectures", [])) else "jax",
          ),
          rollout_vllm_hf_config_path=trainer_config.vllm_hf_config_path,
          rollout_vllm_additional_config=rollout_additional_config, rollout_vllm_init_with_random_weights=True,
          rollout_vllm_enable_dp_attention=trainer_config.enable_dp_attention,
          rollout_vllm_max_num_batched_tokens=trainer_config.max_num_batched_tokens,
          rollout_vllm_max_num_seqs=trainer_config.max_num_seqs,
          rollout_vllm_async_scheduling=trainer_config.async_scheduling,
          rollout_vllm_server_mode=trainer_config.rl.use_agentic_rollout,
          rollout_vllm_reshard_chunk_size=trainer_config.rl.reshard_chunk_size,
          rollout_vllm_free_kv_cache_during_weight_sync=trainer_config.free_kv_cache_during_weight_sync,
          rollout_vllm_kwargs=rollout_vllm_kwargs,
          rollout_vllm_sampling_kwargs={
              "stop": trainer_config.stop_strings, "detokenize": trainer_config.stop_strings is not None,
              "include_stop_str_in_output": trainer_config.stop_strings is not None,
          },
          **({"return_logprobs": True} if trainer_config.rl.use_agentic_rollout else {}),
          **get_rollout_kwargs_for_parallelism(sampler_config, len(sampler_devices)),
      ),
  )
  rl_cluster_kwargs = {}
  if trainer_config.enable_tunix_perf_metrics:
    perf_config = perf_metrics.PerfMetricsConfig()
    perf_config.custom_export_fn = perf_export.PerfMetricsExport.create_metrics_export_fn(cluster_config)
    rl_cluster_kwargs["perf_config"] = perf_config
  rl_cluster = rl_cluster_lib.RLEngine(
      actor=actor_model, reference=reference_model, tokenizer=model_tokenizer, cluster_config=cluster_config,
      **rl_cluster_kwargs,
  )
  if checkpoint_step is not None:
    cluster_config.training_config.checkpoint_root_directory = checkpoint_dir
    with jax.set_mesh(actor_mesh):
      restore_trainer_step(rl_cluster.actor_trainer, os.path.join(checkpoint_dir, "actor"), checkpoint_step)
    rl_cluster.sync_weights()
    rl_cluster.global_steps = 0
  reward_fns = (build_reward_fns(trainer_config, lambda fn: _RewardFn(fn, trainer_config))
                if reward_funcs is None and environment_factory is None else [] if reward_funcs is None
                else [reward_funcs] if callable(reward_funcs) else list(reward_funcs))
  if trainer_config.rl.use_agentic_rollout:
    grpo_config = agentic_grpo_learner.GRPOConfig(
        num_generations=trainer_config.rl.num_generations, num_iterations=trainer_config.rl.num_iterations,
        beta=trainer_config.rl.grpo_beta, epsilon=trainer_config.rl.grpo_epsilon, loss_algo=trainer_config.rl.loss_algo,
        loss_agg_mode=trainer_config.rl.loss_agg_mode,
        max_response_length=trainer_config.max_target_length - trainer_config.max_prefill_predict_length,
        max_concurrency=trainer_config.rl.max_concurrency, off_policy_steps=trainer_config.rl.off_policy_steps,
        system_prompt=trainer_config.rl.system_prompt, epsilon_high=trainer_config.rl.epsilon_high,
        use_rollout_logps=trainer_config.rl.use_rollout_logps, reward_num_workers=trainer_config.reward_num_workers,
        reward_worker_timeout_seconds=trainer_config.reward_worker_timeout_seconds,
    )
    max_logging.log("GRPO config resolved:\n" + "\n".join(f"  {key} = {value!r}" for key, value in sorted(dataclasses.asdict(grpo_config).items())))
    if environment_factory is None:
      template_config = load_data_template_from_file(_template_path(trainer_config.data_template_path))
      if template_config is None:
        raise ValueError(f"Data template is required for AgenticGRPOLearner but failed to load from {trainer_config.data_template_path}")
      chat_parser = utils_rl.MaxTextChatParser(model_tokenizer, template_config, trainer_config)
    else:
      with open(os.path.join(MAXTEXT_PKG_DIR, "assets/tokenizers/qwen3-tokenizer/tokenizer_config.json"),
                encoding="utf-8") as template_file:
        expected_template = json.load(template_file)["chat_template"]
      chat_parser = QwenFactoryParser(model_tokenizer, expected_template)
      agent_class, env_class = FactoryAgent, FactoryEnvironment
      env_kwargs = {"environment_factory": environment_factory, "has_reward_funcs": bool(reward_fns)}
      if reward_fns:
        if trainer_config.reward_num_workers not in (0, 1):
          raise ValueError("environment_factory with reward_funcs requires reward_num_workers 0 or 1.")
        environment_registry = EnvironmentRegistry()
        env_kwargs["environment_registry"] = environment_registry
    agentic_kwargs = {
        key: value for key, value in {
            "agent_class": agent_class, "agent_kwargs": agent_kwargs,
            "env_class": env_class, "env_kwargs": env_kwargs,
        }.items() if value is not None
    }
    learner_class = FactoryGRPOLearner if environment_factory is not None and reward_fns else agentic_grpo_learner.GRPOLearner
    learner_kwargs = {"environment_registry": environment_registry} if learner_class is FactoryGRPOLearner else {}
    rl_trainer = learner_class(
        rl_engine=rl_cluster, reward_fns=reward_fns, algo_config=grpo_config, chat_parser=chat_parser,
        metric_fns=[utils_rl.get_correctness_metrics] if environment_factory is None else None,
        **agentic_kwargs, **learner_kwargs,
    )
  else:
    grpo_config = grpo_learner.GRPOConfig(
        num_generations=trainer_config.rl.num_generations, num_iterations=trainer_config.rl.num_iterations,
        beta=trainer_config.rl.grpo_beta, epsilon=trainer_config.rl.grpo_epsilon, loss_algo=trainer_config.rl.loss_algo,
        loss_agg_mode=trainer_config.rl.loss_agg_mode, reward_num_workers=trainer_config.reward_num_workers,
        reward_worker_timeout_seconds=trainer_config.reward_worker_timeout_seconds,
    )
    max_logging.log("GRPO config resolved:\n" + "\n".join(f"  {key} = {value!r}" for key, value in sorted(dataclasses.asdict(grpo_config).items())))
    rl_trainer = grpo_learner.GRPOLearner(rl_engine=rl_cluster, reward_fns=reward_fns, algo_config=grpo_config)
  return rl_cluster, rl_trainer, optimizer, reward_fns


def configure_tokenizer_chat_template(model_tokenizer: Any, trainer_config: Any) -> None:
  if getattr(model_tokenizer, "chat_template", None) is None:
    if getattr(trainer_config, "chat_template", None):
      model_tokenizer.chat_template = trainer_config.chat_template
    elif getattr(trainer_config, "chat_template_path", None):
      model_tokenizer.chat_template = load_chat_template_from_file(_template_path(trainer_config.chat_template_path))
    else:
      raise ValueError(
          f"Tokenizer {getattr(trainer_config, 'tokenizer_path', None)!r} has no chat_template "
          "and config.chat_template / config.chat_template_path "
          "are both empty. Either pick an instruction-tuned tokenizer that "
          "ships with a chat_template, set config.chat_template to a Jinja "
          "string, or set config.chat_template_path to a JSON file "
          "with a 'chat_template' key."
      )


def _train(
    trainer_config, sampler_config, sampler_devices, *, actor_model, mesh, vllm_config=None,
    reference_model=None, reference_mesh=None, rollout_mesh=None, model_tokenizer=None,
    train_dataset=None, eval_dataset=None, train_iterator=None, eval_iterator=None, goodput_recorder=None,
    checkpoint_step=None, metric_backend=None, report_to=None,
    agent_class=None, agent_kwargs=None, env_class=None, env_kwargs=None, reward_funcs=None, environment_factory=None,
):
  maxtext_vllm_adapter.register()
  if model_tokenizer is None:
    model_tokenizer = AutoTokenizer.from_pretrained(
        trainer_config.tokenizer_path, token=trainer_config.hf_access_token or None
    )
  configure_tokenizer_chat_template(model_tokenizer, trainer_config)
  if reference_model is None:
    reference_model = nnx.clone(actor_model)
  if reference_mesh is None:
    reference_mesh = mesh
  if rollout_mesh is None:
    devices_array = device_mesh.create_device_mesh(sampler_config, sampler_devices)
    rollout_mesh = Mesh(devices_array, sampler_config.mesh_axes)
  if not trainer_config.debug:
    noise_filter = max_logging.NoisyLogFilter()
    logging.getLogger().addFilter(noise_filter)
    absl_logging.get_absl_logger().addFilter(noise_filter)
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
  epath.Path(trainer_config.tensorboard_dir).mkdir(parents=True, exist_ok=True)
  epath.Path(trainer_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
  use_no_op_mappings = "maxtext_config" in trainer_config.vllm_additional_config
  with mesh:
    actor_model = TunixMaxTextAdapter(
        base_model=actor_model, use_no_op_mappings=use_no_op_mappings, pad_id=model_tokenizer.pad_token_id
    )
    actor_model.config = None
  with reference_mesh:
    reference_model = TunixMaxTextAdapter(
        base_model=reference_model, use_no_op_mappings=use_no_op_mappings, pad_id=model_tokenizer.pad_token_id
    )
    reference_model.config = None
  if train_iterator is not None and train_dataset is not None:
    raise ValueError("Pass either train_dataset or train_iterator, not both")
  if eval_iterator is not None and eval_dataset is not None:
    raise ValueError("Pass either eval_dataset or eval_iterator, not both")
  if train_iterator is None or (trainer_config.num_test_batches > 0 and eval_iterator is None):
    default_train, default_eval = prepare_datasets(
        trainer_config, model_tokenizer, train_dataset=train_dataset, eval_dataset=eval_dataset,
        prepare_train=train_iterator is None,
        prepare_eval=trainer_config.num_test_batches > 0 and eval_iterator is None,
    )
    if train_iterator is None:
      train_iterator = default_train
    if eval_iterator is None:
      eval_iterator = default_eval
  if vllm_config is None:
    vllm_config = load_config_values(base_config=os.path.join(MAXTEXT_CONFIGS_DIR, "inference", "vllm.yml"), log_config=False)
  if trainer_config.num_test_batches > 0 and iter(eval_iterator) is eval_iterator:
    raise ValueError("RL evaluation data must be reiterable for pre-training, intermediate, and post-training evaluation")
  with maybe_record_goodput(goodput_recorder, GoodputEvent.TRAINING_PREPARATION):
    rl_cluster, rl_trainer, optimizer, reward_fns = create_rl_components(
        trainer_config, sampler_config, sampler_devices, actor_model, mesh, reference_model, reference_mesh,
        rollout_mesh, model_tokenizer, vllm_config=vllm_config, checkpoint_step=checkpoint_step,
        metric_backend=metric_backend, report_to=report_to,
        agent_class=agent_class, agent_kwargs=agent_kwargs, env_class=env_class, env_kwargs=env_kwargs,
        reward_funcs=reward_funcs, environment_factory=environment_factory,
    )
  if trainer_config.num_test_batches > 0:
    rl_cluster.rollout.update_params(nnx.state(rl_cluster.actor_trainer.model, nnx.Param))
    (corr, total, accuracy, partial_accuracy, format_accuracy, mean_reward), _ = evaluate(
        trainer_config, eval_iterator, rl_cluster=rl_cluster, num_passes=trainer_config.num_eval_passes,
        corr_lst=trainer_config.eval_corr_lst, make_lst=trainer_config.eval_make_lst, reward_fns=reward_fns,
    )
    max_logging.warning(
        f"Pre RL Training: {corr=}, {total=}, {accuracy=}%, {partial_accuracy=}%,"
        f" {format_accuracy=}%, {mean_reward=:.4f}"
    )
  if trainer_config.load_checkpoint_only_once:
    ref_state_before = nnx.to_pure_dict(nnx.state(reference_model.base, nnx.Param))
  utils_rl.install_training_hooks(
      rl_cluster, trainer_config, eval_iterator, reward_fns, goodput_recorder=goodput_recorder, evaluator=evaluate
  )
  rl_trainer.train(train_iterator)
  if trainer_config.load_checkpoint_only_once:
    ref_state_after = nnx.to_pure_dict(nnx.state(reference_model.base, nnx.Param))
    check = jax.tree_util.tree_map(jax.numpy.array_equal, ref_state_before, ref_state_after)
    if not jax.tree_util.tree_all(check):
      raise ValueError("Reference model parameters changed during training!")
  if trainer_config.num_test_batches > 0:
    (corr, total, accuracy, partial_accuracy, format_accuracy, mean_reward), _ = evaluate(
        trainer_config, eval_iterator, rl_cluster=rl_cluster, num_passes=trainer_config.num_eval_passes,
        corr_lst=trainer_config.eval_corr_lst, make_lst=trainer_config.eval_make_lst, reward_fns=reward_fns,
    )
    max_logging.warning(
        f"Post RL Training: {corr=}, {total=}, {accuracy=}%, {partial_accuracy=}%,"
        f" {format_accuracy=}%, {mean_reward=:.4f}"
    )
  return rl_cluster, rl_trainer, optimizer, reward_fns, eval_iterator


def train(
    trainer_config, sampler_config=None, sampler_devices=None, *, actor_model, mesh, vllm_config=None,
    reference_model=None, reference_mesh=None, rollout_mesh=None, model_tokenizer=None,
    train_dataset=None, eval_dataset=None, train_iterator=None, eval_iterator=None, goodput_recorder=None,
    checkpoint_step=None, metric_backend=None, report_to=None,
    agent_class=None, agent_kwargs=None, env_class=None, env_kwargs=None, reward_funcs=None, environment_factory=None,
):
  if environment_factory is not None:
    if not trainer_config.rl.use_agentic_rollout:
      raise ValueError("environment_factory requires agentic rollout.")
    if any(value is not None for value in (agent_class, agent_kwargs, env_class, env_kwargs)):
      raise ValueError("environment_factory cannot be combined with custom agent or environment options.")
    if trainer_config.num_test_batches > 0:
      raise ValueError("environment_factory does not support native RL evaluation.")
  if not trainer_config.rl.use_agentic_rollout and any(
      value is not None for value in (agent_class, agent_kwargs, env_class, env_kwargs)
  ):
    raise ValueError("Custom agent and environment options require agentic rollout.")
  install_pathways_multihost_profiler_patch(trainer_config.rl.profiler_max_num_hosts)
  if sampler_config is None:
    if trainer_config.num_trainer_slices != -1 or trainer_config.num_samplers_slices != -1:
      raise ValueError("A resolved sampler_config is required for multislice RL training")
    sampler_config = trainer_config
  if sampler_devices is None:
    if trainer_config.num_trainer_slices != -1 or trainer_config.num_samplers_slices != -1:
      raise ValueError("sampler_devices are required for multislice RL training")
    sampler_devices = jax.devices()
    if len(sampler_devices) // trainer_config.chips_per_vm >= 2 and trainer_config.use_pathways:
      count = int(len(sampler_devices) * trainer_config.sampler_devices_fraction)
      sampler_devices = sampler_devices[len(sampler_devices) - count:]
  if trainer_config.managed_mldiagnostics:
    ManagedMLDiagnostics(trainer_config, sampler_config=sampler_config)
  if goodput_recorder is None:
    goodput_recorder = create_goodput_recorder(trainer_config)
  record_goodput(goodput_recorder, RECORD_JOB_START_TIME)
  with maybe_monitor_goodput(trainer_config):
    result = _train(
        trainer_config, sampler_config, sampler_devices, actor_model=actor_model, mesh=mesh,
        vllm_config=vllm_config, reference_model=reference_model, reference_mesh=reference_mesh,
        rollout_mesh=rollout_mesh, model_tokenizer=model_tokenizer, train_dataset=train_dataset,
        eval_dataset=eval_dataset, train_iterator=train_iterator, eval_iterator=eval_iterator,
        goodput_recorder=goodput_recorder, checkpoint_step=checkpoint_step, metric_backend=metric_backend,
        report_to=report_to,
        agent_class=agent_class, agent_kwargs=agent_kwargs, env_class=env_class, env_kwargs=env_kwargs,
        reward_funcs=reward_funcs, environment_factory=environment_factory,
    )
  record_goodput(goodput_recorder, RECORD_JOB_END_TIME)
  return result
