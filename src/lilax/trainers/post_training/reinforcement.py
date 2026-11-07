"""GRPO training through the native reinforcement-learning path."""

import os
from contextlib import ExitStack, nullcontext

from flax import nnx
import jax

from lilax._backend.configs import pyconfig
from lilax._backend.common.profiler import xprof_server
from lilax._backend.models.creation import pretrained
from lilax.modeling.models import MaxTextModel
from lilax._backend.trainers.post_train.checkpoint_selection import TrainingCheckpointManager, training_checkpoint_options

from lilax.data.rl import _batch_rl_dataset
from lilax.data_pipeline.trainer import validate_eval_dataset
from lilax.processing import AutoTokenizer
from ..lifecycle.callbacks import TrainerEvents
from ..lifecycle.checkpoint_request import checkpoint_request
from ..lifecycle._post_base_checkpoint import prepare_training_checkpoint
from .grpo_config import GRPOConfig, GRPOEvaluationConfig
from .grpo_runtime import install_grpo_runtime
from . import _native_grpo


class GRPOTrainer(TrainerEvents):
  def __init__(
      self, model, reward_funcs=None, *, args: GRPOConfig, train_dataset=None, eval_dataset=None,
      processing_class=None, callbacks=None, optimizers=(None, None), ref_model=None, environment_factory=None,
      rollout_config=None, agentic_config=None,
  ):
    validate_eval_dataset(eval_dataset)
    callbacks = [callback() if isinstance(callback, type) else callback for callback in (callbacks or [])]
    self._uses_maxtext_engine = rollout_config is not None and rollout_config.engine == "maxtext"
    if self._uses_maxtext_engine:
      _native_grpo.validate_native_options(
          args, model, reward_funcs, eval_dataset, rollout_config, agentic_config, environment_factory,
      )
    self.model = model
    self.config, self.sampler_config, self.sampler_devices = install_grpo_runtime(
        model, args, processing_class, train_dataset, eval_dataset, rollout_config, agentic_config, callbacks,
    )
    self.args = args
    self.train_dataset = train_dataset
    self.eval_dataset = eval_dataset
    self.processing_class = processing_class
    self.optimizers = optimizers
    self.reference_model = ref_model
    self.rollout_config = rollout_config
    self.agentic_config = agentic_config
    self.rollout_mesh = None
    self.eval_iterator = None
    binding = agentic_config.agent if agentic_config is not None else None
    self.agent_class = binding.agent_class if binding is not None else None
    self.agent_kwargs = binding.agent_kwargs if binding is not None else None
    self.env_class = binding.environment_class if binding is not None else None
    self.env_kwargs = binding.environment_kwargs if binding is not None else None
    self.reward_funcs = reward_funcs
    self.environment_factory = environment_factory
    self.native_trainer = None
    self.rl_cluster = None
    self.optimizer = None
    self.reward_fns = None
    self._init_events(callbacks)
    self._native_grpo_setup = None
    self._native_grpo_state = None
    self._checkpoint_root = None

  def train(self, resume_from_checkpoint=None, *, checkpoint_step=None):
    request = checkpoint_request(resume_from_checkpoint, checkpoint_step)
    if self.train_dataset is None:
      raise ValueError("GRPO training requires train_dataset.")
    if self._uses_maxtext_engine:
      if self.processing_class is None:
        self.processing_class = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        self.callback_handler.processing_class = self.processing_class
      _native_grpo.train_native(self, request)
      return self._train_output()
    self._run(request)
    return self._train_output()

  def _run(self, request, *, evaluation_only=False):
    from lilax._backend.trainers.post_train.rl import train_rl

    with ExitStack() as resources:
      if request.directory is not None:
        self.config = pyconfig.HyperParameters(self.config._pydantic_config.model_copy(
            update={"checkpoint_dir": request.directory},
        ))
      config = self.config
      report_to = []
      if self.processing_class is None:
        self.processing_class = AutoTokenizer.from_pretrained(config.tokenizer_path)
        self.callback_handler.processing_class = self.processing_class
      train_iterator = iter(()) if evaluation_only else _batch_rl_dataset(
          self.train_dataset, config, self.processing_class, agentic=config.rl.use_agentic_rollout,
      )
      self.eval_iterator = None if self.eval_dataset is None else _batch_rl_dataset(
          next(iter(self.eval_dataset.values())) if isinstance(self.eval_dataset, dict) else self.eval_dataset,
          config, self.processing_class, evaluation=True,
      )
      options = training_checkpoint_options(config) if config.enable_checkpointing else None
      root = (request.directory or os.path.join(config.checkpoint_dir, "actor")) if config.enable_checkpointing else None
      self._checkpoint_root = root
      manager = TrainingCheckpointManager(root, options)
      resources.callback(manager.close)
      manager, selected_step = prepare_training_checkpoint(
          request.step, manager, self.args, config, self.model, resume=request.resume,
      )
      self.model.materialize(runtime=self.model.runtime)
      qat = self.model.existing_graph_qat_config
      manager.training_graph = {"version": 1, "kind": "qat" if qat is not None else "fp",
                                "qat_config": None if qat is None else qat.model_dump(mode="json")}
      reference_model = self.reference_model
      if reference_model is None and not config.load_checkpoint_only_once:
        reference_native, reference_config = pretrained.from_pretrained(
            self.model.construction_config, mesh=self.model.runtime.mesh, return_effective_config=True
        )
        reference_model = MaxTextModel(
            reference_native, self.model.runtime, construction_config=reference_config
        )
      if reference_model is not None:
        reference_model.runtime.resolve_config()
        if reference_model.runtime.mesh is None:
          reference_model.runtime.mesh = self.model.runtime.mesh
          reference_model.runtime.devices = self.model.runtime.devices
      xprof = self.model.runtime._profile.xprof if self.model.runtime._profile is not None else None
      server = (
          xprof_server(config.tensorboard_dir, xprof.port, jax.process_index())
          if xprof is not None and xprof.launch_server else nullcontext()
      )
      generation = self.rollout_config.generation_config if self.rollout_config is not None else None
      eos_ids = generation.eos_token_id if generation is not None else None
      if eos_ids is None:
        eos_ids = getattr(self.model.config, "eos_token_id", None)
      if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
      with (nullcontext() if evaluation_only else self._reporting_context()), server:
        operation = train_rl._train if evaluation_only else train_rl.train
        result = operation(
            config, self.sampler_config, self.sampler_devices,
            actor_model=self.model.native_model, mesh=self.model.runtime.mesh,
            reference_model=None if reference_model is None else reference_model.native_model,
            reference_mesh=None if reference_model is None else reference_model.runtime.mesh,
            rollout_mesh=self.rollout_mesh,
            model_tokenizer=getattr(self.processing_class, "tokenizer", self.processing_class),
            train_iterator=train_iterator, eval_iterator=self.eval_iterator,
            agent_class=self.agent_class, agent_kwargs=self.agent_kwargs, env_class=self.env_class, env_kwargs=self.env_kwargs,
            reward_funcs=self.reward_funcs, environment_factory=self.environment_factory,
            checkpoint_step=selected_step, resume=request.resume, report_to=report_to,
            lifecycle=self, optimizer=self.optimizers[0], learning_rate_schedule=self.optimizers[1],
            eos_token_ids=eos_ids,
            checkpoint_manager=manager, **({"evaluation_only": True} if evaluation_only else {}),
        )
      self.rl_cluster, self._learner, self.optimizer, self.reward_fns, self.eval_iterator = result
      self.native_trainer = self.rl_cluster.actor_trainer
      self.model.native_model = self.native_trainer.model.base

  def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
    dataset = self.eval_dataset if eval_dataset is None else eval_dataset
    validate_eval_dataset(dataset)
    if self._uses_maxtext_engine:
      raise NotImplementedError("The MaxText GRPO engine has no evaluation-input pipeline.")
    if dataset is None:
      raise ValueError("Trainer.evaluate requires an evaluation dataset.")
    if isinstance(dataset, dict):
      metrics = {}
      for name, source in dataset.items():
        metrics.update(self.evaluate(source, ignore_keys, f"{metric_key_prefix}_{name}"))
      return metrics
    if self.environment_factory is not None:
      raise NotImplementedError("Native RL evaluation does not support environment_factory.")
    from lilax._backend.trainers.post_train.rl import train_rl

    temporary_cluster = self.rl_cluster is None or self.rl_cluster._lifecycle_finished
    if temporary_cluster:
      self._run(checkpoint_request(None), evaluation_only=True)
    try:
      eval_iterator = _batch_rl_dataset(dataset, self.config, self.processing_class, evaluation=True)
      self.rl_cluster.rollout.update_params(nnx.state(self.rl_cluster.actor_trainer.model, nnx.Param))
      values, responses = train_rl.evaluate(
          self.config, eval_iterator, rl_cluster=self.rl_cluster, num_passes=self.config.num_eval_passes,
          corr_lst=self.config.eval_corr_lst, make_lst=self.config.eval_make_lst, reward_fns=self.reward_fns,
      )
    finally:
      if temporary_cluster:
        self.rl_cluster.close()
        self.rl_cluster, self.native_trainer = None, None
    metrics = dict(zip(("eval_correct", "eval_total", "eval_accuracy", "eval_partial_accuracy",
                        "eval_format_accuracy", "eval_mean_reward"), values))
    if self.config.eval_make_lst:
      metrics["eval_responses"] = responses
    return self._evaluation_metrics(metrics, metric_key_prefix)

  def generate(self, prompts, num_passes=1):
    if self._uses_maxtext_engine:
      if self._native_grpo_state is None:
        raise RuntimeError("Training must complete before generation.")
      return _native_grpo.generate_native(self, prompts, num_passes)
    if self.rl_cluster is None:
      raise RuntimeError("Training must complete before generation.")
    from lilax._backend.trainers.post_train.rl.evaluate_rl import generate_responses

    self.rl_cluster.rollout.update_params(nnx.state(self.rl_cluster.actor_trainer.model, nnx.Param))
    return generate_responses(self.config, prompts, self.rl_cluster, num_passes=num_passes)

  def save_model(self, output_dir=None):
    if self.rl_cluster is not None:
      self.model.native_model = self.rl_cluster.actor_trainer.model.base
    destination = self.args.output_dir if output_dir is None else output_dir
    self.model.save_pretrained(destination)
    if self.processing_class is not None:
      self.processing_class.save_pretrained(destination)

  def save_state(self):
    if self._uses_maxtext_engine:
      return _native_grpo.save_native_state(self)
    if self.rl_cluster is None:
      raise RuntimeError("Training must complete before saving a checkpoint.")
    trainer = self.rl_cluster.actor_trainer
    root = self._checkpoint_root
    if root is None:
      raise ValueError("Checkpointing is disabled; provide an output directory.")
    manager = TrainingCheckpointManager(
        root_directory=root, options=trainer.config.checkpointing_options
    )
    manager.training_graph = trainer.checkpoint_manager.training_graph
    manager.save_only_lora_params = trainer.checkpoint_manager.save_only_lora_params
    manager.lifecycle = self
    self.control.should_save = False
    try:
      latest_step = manager.latest_step()
      if latest_step is None or latest_step < trainer.train_steps:
        manager.save(
            trainer.train_steps, trainer.model, trainer.optimizer, save_only_lora_params=trainer._lora_enabled,
            force=True, custom_metadata=trainer.custom_checkpoint_metadata(),
        )
      for receipt in manager.save_receipts:
        self._resolve_save(receipt, manager._checkpointer)
    finally:
      manager.close()
