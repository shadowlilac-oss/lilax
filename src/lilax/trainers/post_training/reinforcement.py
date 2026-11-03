"""GRPO training through the native reinforcement-learning path."""

import os
import threading
from contextlib import nullcontext

from flax import nnx
import jax
from jax.sharding import Mesh

from lilax._backend.configs import pyconfig, types
from lilax._backend.common.profiler import xprof_server
from lilax._backend.trainers.post_train.rl import train_rl
from lilax._backend.trainers.post_train.rl.evaluate_rl import generate_responses
from lilax._backend.utils import device_mesh
from lilax._backend.utils.model_creation import pretrained
from lilax.modeling.models import MaxTextModel
from lilax.runtime import Runtime
from tunix.sft import checkpoint_manager

from ..arguments.training_args import TrainingArguments
from ..telemetry.reporting import MetricReporter


class _RLMetricBackend:
  def __init__(self, reporter):
    self.reporter = reporter
    self.lock = threading.Lock()

  def log_scalar(self, event, value, **kwargs):
    with self.lock:
      if self.reporter is not None:
        self.reporter.log({"scalar": {event: value}}, kwargs.get("step"), None)

  def close(self):
    pass


class GRPOConfig(TrainingArguments):
  def resolve(self, runtime):
    if self.max_steps is not None or self.gradient_accumulation_steps is not None:
      raise ValueError("Native RL derives its training steps from the dataset and does not use gradient accumulation.")
    unsupported = {
        key for key in self._overrides()
        if key not in {"base_output_directory", "learning_rate", "eval_interval", "checkpoint_period",
                       "enable_tensorboard", "enable_wandb"}
    }
    if unsupported:
      raise ValueError(f"Native RL does not support these training argument overrides: {sorted(unsupported)}")
    overrides = {
        key: value for key, value in (
            ("base_output_directory", self.output_dir), ("learning_rate", self.learning_rate),
            ("eval_interval", self.eval_steps), ("checkpoint_period", self.save_steps),
        ) if value is not None
    }
    overrides.update({key: value for key, value in self._overrides().items()
                      if key in {"enable_tensorboard", "enable_wandb"}})
    if not overrides:
      return runtime.config
    overrides["run_name"] = runtime.config.run_name
    values = runtime.public_config.get_keys() | overrides
    prepared = pyconfig._prepare_for_pydantic(values, config_class=types.RLConfig)
    config = types.RLConfig.model_validate(prepared, context={"resolved_values": True})
    partition = {
        key: getattr(runtime.config, key) for key in (
            "num_slices", "ici_fsdp_parallelism", "ici_tensor_parallelism", "dcn_data_parallelism",
            "ici_parallelism", "dcn_parallelism",
        )
    }
    config = config.model_copy(update=partition)
    object.__setattr__(
        config, "__pydantic_fields_set__",
        (runtime.config._pydantic_config.model_fields_set | overrides.keys()) & types.RLConfig.model_fields.keys(),
    )
    return pyconfig.HyperParameters(config)


class GRPOTrainer:
  @classmethod
  def from_config(cls, config, args=None, *, devices=None, reference_model=None, rollout_mesh=None, **kwargs):
    runtime, sampler_config, sampler_devices = Runtime.for_rl(config, devices=devices)
    trainer_config = GRPOConfig.resolve(args, runtime) if args is not None else runtime.config
    if reference_model is None:
      reference_native, reference_config = pretrained.from_pretrained(
          trainer_config, mesh=runtime.mesh, return_effective_config=True
      )
      reference_model = MaxTextModel(reference_native, runtime, construction_config=reference_config)
      clone_reference = trainer_config.load_checkpoint_only_once
    else:
      clone_reference = False
    if rollout_mesh is None:
      devices_array = device_mesh.create_device_mesh(sampler_config, sampler_devices)
      rollout_mesh = Mesh(devices_array, sampler_config.mesh_axes)
    if clone_reference:
      with runtime.mesh:
        actor_native = nnx.clone(reference_model.native_model)
      actor_config = reference_model.construction_config
    else:
      actor_native, actor_config = pretrained.from_pretrained(
          trainer_config, mesh=runtime.mesh, return_effective_config=True
      )
    model = MaxTextModel(actor_native, runtime, construction_config=actor_config)
    return cls(
        model, args=args, reference_model=reference_model, sampler_config=sampler_config,
        sampler_devices=sampler_devices, rollout_mesh=rollout_mesh, **kwargs,
    )

  def __init__(
      self, model, args=None, train_dataset=None, eval_dataset=None, processing_class=None,
      data_collator=None, optimizers=(None, None), callbacks=None, *, reference_model=None,
      sampler_config=None, sampler_devices=None, vllm_config=None, rollout_mesh=None, model_tokenizer=None,
      train_iterator=None, eval_iterator=None, goodput_recorder=None,
  ):
    if processing_class is not None or data_collator is not None:
      raise NotImplementedError("Native RL does not accept a processing class or data collator.")
    if optimizers != (None, None) or callbacks:
      raise NotImplementedError("Native RL does not accept custom optimizers or callbacks.")
    self.model = model
    self.config = GRPOConfig.resolve(args, model.runtime) if args is not None else model.runtime.config
    if self.config.scan_layers != model.construction_config.scan_layers:
      self.config = pyconfig.HyperParameters(
          self.config._pydantic_config.model_copy(update={"scan_layers": model.construction_config.scan_layers})
      )
    self.args = args
    self.train_dataset = train_dataset
    self.eval_dataset = eval_dataset
    self.reference_model = reference_model
    self.sampler_config = sampler_config
    self.sampler_devices = sampler_devices
    self.vllm_config = vllm_config
    self.rollout_mesh = rollout_mesh
    self.model_tokenizer = model_tokenizer
    self.train_iterator = train_iterator
    self.eval_iterator = eval_iterator
    self.goodput_recorder = goodput_recorder
    self.native_trainer = None
    self.rl_cluster = None
    self.optimizer = None
    self.reward_fns = None

  def train(self, resume_from_checkpoint=None):
    if resume_from_checkpoint is None or resume_from_checkpoint is True:
      checkpoint_step = None
    elif (
        isinstance(resume_from_checkpoint, int) and not isinstance(resume_from_checkpoint, bool)
        and resume_from_checkpoint >= 0
    ):
      checkpoint_step = resume_from_checkpoint
    else:
      raise ValueError("resume_from_checkpoint must be None, True, or a nonnegative checkpoint step.")
    config = self.config
    report_to = self.args.report_to if self.args is not None else None
    if (self.eval_dataset is not None or self.eval_iterator is not None) and config.num_test_batches <= 0:
      raise ValueError("Supplying an evaluation dataset requires a positive num_test_batches.")
    reference_model = self.reference_model
    if reference_model is None and not config.load_checkpoint_only_once:
      reference_native, reference_config = pretrained.from_pretrained(
          self.model.construction_config, mesh=self.model.runtime.mesh, return_effective_config=True
      )
      reference_model = MaxTextModel(
          reference_native, self.model.runtime, construction_config=reference_config
      )
    server = (
        xprof_server(config.tensorboard_dir, self.args.xprof_port, jax.process_index())
        if self.args is not None and self.args.launch_xprof else nullcontext()
    )
    reporter = MetricReporter(config, report_to, tensorboard=True) if report_to else None
    backend = _RLMetricBackend(reporter) if reporter is not None else None
    try:
      with server:
        result = train_rl.train(
          config, self.sampler_config, self.sampler_devices,
          actor_model=self.model.native_model, mesh=self.model.runtime.mesh,
          reference_model=None if reference_model is None else reference_model.native_model,
          reference_mesh=None if reference_model is None else reference_model.runtime.mesh,
          train_dataset=self.train_dataset, eval_dataset=self.eval_dataset,
          vllm_config=self.vllm_config, rollout_mesh=self.rollout_mesh, model_tokenizer=self.model_tokenizer,
          train_iterator=self.train_iterator, eval_iterator=self.eval_iterator, goodput_recorder=self.goodput_recorder,
            checkpoint_step=checkpoint_step, metric_backend=backend, report_to=report_to,
        )
    finally:
      if backend is not None:
        with backend.lock:
          backend.reporter = None
          reporter.close()
    self.rl_cluster, self.native_trainer, self.optimizer, self.reward_fns, self.eval_iterator = result
    return self.native_trainer

  def evaluate(self, eval_iterator=None):
    if self.rl_cluster is None:
      raise RuntimeError("Training must complete before evaluation.")
    if eval_iterator is None:
      eval_iterator = self.eval_iterator
    if eval_iterator is None:
      raise ValueError("An RL evaluation iterator is required.")
    return train_rl.evaluate(
        self.config, eval_iterator, rl_cluster=self.rl_cluster, num_passes=self.config.num_eval_passes,
        corr_lst=self.config.eval_corr_lst, make_lst=self.config.eval_make_lst, reward_fns=self.reward_fns,
    )

  def generate(self, prompts, num_passes=1):
    if self.rl_cluster is None:
      raise RuntimeError("Training must complete before generation.")
    return generate_responses(self.config, prompts, self.rl_cluster, num_passes=num_passes)

  def save_model(self, output_dir=None):
    if self.rl_cluster is None:
      raise RuntimeError("Training must complete before saving a checkpoint.")
    trainer = self.rl_cluster.actor_trainer
    root = trainer.config.checkpoint_root_directory if output_dir is None else os.path.join(str(output_dir), "actor")
    if root is None:
      raise ValueError("Checkpointing is disabled; provide an output directory.")
    manager = trainer.checkpoint_manager if output_dir is None else checkpoint_manager.CheckpointManager(
        root_directory=root, options=trainer.config.checkpointing_options
    )
    try:
      latest_step = manager.latest_step()
      if latest_step is None or latest_step < trainer.train_steps:
        manager.save(
            trainer.train_steps, trainer.model, trainer.optimizer, save_only_lora_params=trainer._lora_enabled,
            force=True, custom_metadata=trainer.custom_checkpoint_metadata(),
        )
    finally:
      if output_dir is not None:
        manager.close()

  def save_state(self):
    self.save_model()
