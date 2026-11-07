"""Bind the existing offline MaxText GRPO computation to the public Trainer lifecycle."""

from contextlib import ExitStack

from flax import nnx

from lilax._backend.common.checkpoints import checkpointing
from lilax._backend.common.goodput import (
    RECORD_JOB_START_TIME, create_goodput_recorder, maybe_monitor_goodput, record_goodput,
)
from lilax._backend.configs.pyconfig import HyperParameters
from lilax._backend.experimental.rl import grpo_trainer
from lilax._backend.input_pipeline.multihost_dataloading import _close_iterator
from lilax._backend.utils import train_utils
from lilax.generation.configuration import GenerationConfig
from lilax.peft import PeftModel


def validate_native_options(args, model, reward_funcs, eval_dataset, rollout, agentic, environment_factory):
  policy = args.checkpoint_config
  if policy is not None and (policy.save_only_lora_params is not None or policy.restore_only_lora_params is not None):
    raise ValueError("The MaxText GRPO engine does not support LoRA-only checkpoints.")
  unsupported = args.model_fields_set & {
      "batch_size", "train_micro_batch_size", "rollout_micro_batch_size", "max_seq_token_per_tpu",
      "epsilon_high", "loss_type", "loss_aggregation", "use_rollout_logps", "reward_num_workers",
      "reward_worker_timeout_seconds", "rewards", "evaluation_config",
  }
  if unsupported:
    raise ValueError(f"The MaxText GRPO engine does not consume these Tunix options: {sorted(unsupported)}")
  if args.num_iterations != 1:
    raise ValueError("The MaxText GRPO engine performs one optimizer update per rollout batch.")
  if eval_dataset is not None or args.eval_on_start or args.eval_steps > 0:
    raise ValueError("The MaxText GRPO engine has no evaluation-input pipeline; use the vLLM engine for evaluation.")
  if reward_funcs is not None:
    raise ValueError("The MaxText GRPO engine uses its existing fixed length reward and does not accept reward_funcs.")
  if agentic is not None or environment_factory is not None:
    raise ValueError("Agentic rollouts and environments require the vLLM GRPO engine.")
  if rollout.vllm is not None or rollout.weight_sync is not None or rollout.reshard_chunk_size is not None:
    raise ValueError("vLLM and Tunix weight-sync options are not consumed by the MaxText GRPO engine.")
  if rollout.cluster is not None:
    unused = rollout.cluster.model_fields_set - {
        "inference_replicas", "inference_devices_per_replica", "inference_rollouts", "use_pathways_reshard",
    }
    if unused:
      raise ValueError(f"The MaxText GRPO engine uses prefix device allocation, not these Tunix settings: {sorted(unused)}")
  if model.pending_qat_config is not None or model.existing_graph_qat_config is not None:
    raise NotImplementedError("The MaxText GRPO objective does not support Qwix QAT graphs.")
  if isinstance(model, PeftModel) or model.runtime.public_config.lora.enable_lora:
    raise NotImplementedError("The MaxText GRPO objective differentiates nnx.Param and does not support LoRA training.")


def train_native(trainer, request):
  config = trainer.config
  if not 1 <= config.inference_rollouts <= config.steps:
    raise ValueError("inference_rollouts must be between one and the configured training duration.")
  if config.decode_sampling_strategy == "greedy" or config.decode_sampling_temperature == 0.0:
    raise ValueError("The MaxText GRPO engine requires stochastic generation with positive temperature.")
  if config.per_device_batch_size < 1.0 or trainer.sampler_config.per_device_batch_size < 1.0:
    raise ValueError("The MaxText GRPO engine does not support per-device batch sizes below one.")
  updates = {"use_grpo": True, "train_data_columns": ["prompts"]}
  if request.directory is not None:
    updates["checkpoint_dir"] = request.directory
  trainer.config = config = HyperParameters(config._pydantic_config.model_copy(update=updates))
  inference_config = HyperParameters(trainer.sampler_config._pydantic_config.model_copy(update=updates))
  dataset = trainer.train_dataset
  columns = dataset.column_names or next(iter(dataset), {}).keys()
  if "prompts" not in columns:
    if "prompt" not in columns:
      raise ValueError("The MaxText GRPO dataset requires a prompt or prompts column; rename your dataset column.")
    dataset = dataset.rename_column("prompt", "prompts")
  if config.num_epoch > 1:
    dataset = dataset.repeat(config.num_epoch)
  model = trainer.model.native_model
  reference = None
  if trainer.reference_model is not None:
    runtime = trainer.reference_model.runtime.resolve_config()
    if runtime.mesh is None:
      runtime.mesh, runtime.devices = trainer.model.runtime.mesh, trainer.model.runtime.devices
    elif runtime.mesh != trainer.model.runtime.mesh:
      raise ValueError("The MaxText GRPO reference and actor must use the same mesh.")
    reference = trainer.reference_model.native_model
  recorder = create_goodput_recorder(config)
  record_goodput(recorder, RECORD_JOB_START_TIME)
  with ExitStack() as resources, trainer._reporting_context(), maybe_monitor_goodput(config):
    setup = grpo_trainer.setup_train_loop(
        config, inference_config, recorder, actor_model=model, reference_model=reference,
        mesh=trainer.model.runtime.mesh, train_dataset=dataset,
        processing_class=getattr(trainer.processing_class, "tokenizer", trainer.processing_class),
        optimizer=trainer.optimizers[0], learning_rate_schedule=trainer.optimizers[1],
        checkpoint_step=-1 if request.step is None else request.step, resume=request.resume,
    )
    trainer._native_grpo_setup = setup
    trainer._native_grpo_manager_closed = False
    manager, iterator = setup[1], setup[9]
    if manager is not None:
      resources.callback(manager.close)
    resources.callback(setattr, trainer, "_native_grpo_manager_closed", True)
    resources.callback(_close_iterator, iterator)
    trainer.callback_handler.optimizer = setup[11].optimizer
    trainer.callback_handler.lr_scheduler = setup[8]
    generation = trainer.rollout_config.generation_config
    eos = generation.eos_token_id if generation is not None else None
    if eos is None:
      eos = getattr(trainer.model.config, "eos_token_id", None)
    if isinstance(eos, int):
      eos = [eos]
    state = grpo_trainer.train_loop(
        config, inference_config, recorder, setup=setup,
        processing_class=getattr(trainer.processing_class, "tokenizer", trainer.processing_class), lifecycle=trainer,
        eos_token_ids=eos,
    )
    nnx.update(model, state["model"])


def save_native_state(trainer):
  setup = trainer._native_grpo_setup
  if setup is None:
    raise RuntimeError("Training must begin before saving its state.")
  with ExitStack() as resources:
    manager = setup[1]
    if trainer._native_grpo_manager_closed:
      graph = nnx.graphdef(setup[11])
      manager = train_utils.create_checkpoint_manager(
          trainer.config, trainer.model.runtime.mesh, lambda: nnx.merge(graph, trainer._native_grpo_state),
      )
      if manager is not None:
        resources.callback(manager.close)
    if manager is None:
      raise ValueError("Checkpointing is disabled in the training configuration.")
    receipt = checkpointing.maybe_save_checkpoint(
        manager, trainer._native_grpo_state, trainer.config, setup[9], force=True,
        **trainer._save_action(trainer.state.global_step), return_result=True,
    )
    trainer._resolve_save(receipt, manager)
    checkpointing.wait_until_finished(manager)


def generate_native(trainer, prompts, num_passes):
  tokenizer = getattr(trainer.processing_class, "tokenizer", trainer.processing_class)
  strategy = trainer.config.generation_configs[trainer.config.eval_sampling_strategy]
  generation = GenerationConfig(
      max_new_tokens=trainer.config.max_target_length - trainer.config.max_prefill_predict_length,
      do_sample=strategy["eval_temperature"] > 0, temperature=strategy["eval_temperature"],
      top_k=strategy["eval_top_k"], top_p=strategy["eval_top_p"],
      eos_token_id=getattr(trainer.model.config, "eos_token_id", None),
  )
  inputs = [tokenizer.encode(prompt) for prompt in prompts]
  responses = [[] for _ in prompts]
  for _ in range(num_passes):
    for index, output in enumerate(trainer.model.generate_batch(inputs, generation)):
      responses[index].append(tokenizer.decode(output.token_ids[output.prompt_length:], skip_special_tokens=True))
  return responses
