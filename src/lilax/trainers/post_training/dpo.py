"""Preference training through the Tunix path."""

from typing import Literal

from lilax._backend.configs.pyconfig import HyperParameters
from lilax._backend.configs.runtime import resolve_initialized_config
from lilax._backend.trainers.post_train.dpo import train_dpo
from lilax.data_pipeline.trainer import dataset_overrides, validate_eval_dataset
from lilax._processing.preferences import processing_overrides
from lilax.callbacks._configuration import callback_overrides

from ..arguments.training_args import MMAP_FIELDS
from .arguments import _PostTrainingArguments
from .data_options import _check_post_training_data_options
from .managed import _ManagedPostTrainer


class DPOConfig(_PostTrainingArguments):
  algorithm: Literal["dpo", "orpo"] = "dpo"
  beta: float = 0.1
  label_smoothing: float = 0.0
  orpo_lambda: float = 0.1
  max_prompt_length: int | None = None

  def _overrides(self):
    return super()._overrides() | {"dpo": {
        "algo": self.algorithm, "dpo_beta": self.beta, "dpo_label_smoothing": self.label_smoothing,
        "orpo_lambda": self.orpo_lambda, "max_prompt_length": self.max_prompt_length,
    }}


class DPOTrainer(_ManagedPostTrainer):
  _native_train = staticmethod(train_dpo.train)
  _tunix_config = staticmethod(train_dpo.get_tunix_config)
  _setup_native = staticmethod(train_dpo.setup_trainer_state)

  def __init__(self, model, *, args: DPOConfig, data_collator=None, train_dataset=None, eval_dataset=None,
               processing_class=None, callbacks=None, optimizers=(None, None)):
    validate_eval_dataset(eval_dataset)
    callbacks = [callback() if isinstance(callback, type) else callback for callback in (callbacks or [])]
    if args is not None:
      unsupported = [name for name in MMAP_FIELDS if name in args._overrides()]
      if args._overrides().get("grain_file_type") in ("mmap", "mmap_npy"):
        unsupported.append("grain_file_type")
      if unsupported:
        raise ValueError(f"Native DPO does not support these data overrides: {sorted(unsupported)}")
    runtime = model.runtime
    args_overrides = args._overrides() if args is not None else {}
    args_overrides.update(callback_overrides(callbacks))
    args_overrides.update(processing_overrides(processing_class))
    args_overrides.update(dataset_overrides(train_dataset))
    objective_overrides = {"use_dpo": True, "use_sft": False}
    if "packing" not in runtime.public_config._explicit_keys:
      objective_overrides["packing"] = False
    datasets_provided = {
        "train_dataset_provided": train_dataset is not None, "eval_dataset_provided": eval_dataset is not None,
    }
    graph_config = model.existing_graph_config
    if graph_config is not None:
      self._check_precision_overrides(graph_config, args_overrides)
    else:
      runtime.initialize(overrides=args_overrides, **datasets_provided)
      self._check_precision_options(runtime.config, args_overrides)
    config = resolve_initialized_config(
        runtime.public_config, runtime.config, overrides=runtime._overrides | objective_overrides | args_overrides,
        **datasets_provided,
    )
    # W&B starts only through an explicit report_to request; the suppression stays out of the shared runtime.
    config = HyperParameters(config._pydantic_config.model_copy(update={"enable_wandb": False}))
    super().__init__(
        model, args, train_dataset, eval_dataset, processing_class, data_collator, optimizers, callbacks,
        resolved_config=config, defer_materialization=True,
    )
    _check_post_training_data_options(self.config, model.runtime, args)

  def _backend_options(self):
    return super()._backend_options() | {"processing_class": self.processing_class}
