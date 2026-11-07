"""Supervised training through the native and Tunix paths."""

from lilax._backend.configs.pyconfig import HyperParameters
from lilax._backend.configs.runtime import resolve_initialized_config
from lilax._backend.trainers.post_train.sft import train_sft
from lilax.data_pipeline.trainer import dataset_overrides, validate_eval_dataset
from lilax._processing.preferences import processing_overrides
from lilax.callbacks._configuration import callback_overrides

from ..pretraining.trainer import Trainer as _NativeTrainer
from ..lifecycle._base import _PostTrainer
from ..arguments.training_args import MMAP_FIELDS
from .arguments import _PostTrainingArguments
from .data_options import _check_post_training_data_options
from .managed import _ManagedPostTrainer


class SFTConfig(_PostTrainingArguments):
  learning_rate: float = 2e-5
  max_length: int | None = None
  packing: bool = True
  completion_only_loss: bool = True
  dataset_text_field: str = "text"
  xprof_dir: str | None = None

  def _overrides(self):
    values = super()._overrides() | {
        "use_sft": True, "packing": self.packing, "sft_train_on_completion_only": self.completion_only_loss,
    }
    if self.max_length is not None:
      values["max_target_length"] = self.max_length
    if self.xprof_dir is not None:
      values["profiler"] = "xplane"
    return values


class SFTTrainer(_NativeTrainer, _ManagedPostTrainer):
  _native_train = staticmethod(train_sft.train)
  _tunix_config = staticmethod(train_sft.get_tunix_config)
  _setup_native = staticmethod(train_sft.setup_trainer_state)

  def __init__(
      self, model, *, args: SFTConfig, formatting_func=None, data_collator=None, train_dataset=None,
      eval_dataset=None, processing_class=None, callbacks=None, optimizers=(None, None),
  ):
    validate_eval_dataset(eval_dataset)
    callbacks = [callback() if isinstance(callback, type) else callback for callback in (callbacks or [])]
    self.formatting_func = formatting_func
    overrides = {"use_sft": True} | processing_overrides(processing_class)
    runtime = model.runtime
    source = runtime.public_config.get_source_inputs()
    if source is not None and source[0] is None:
      defaults = {
          "use_tunix_gradient_accumulation": True, "sft_train_on_completion_only": True,
          "packing": True, "learning_rate": 2e-5,
      }
      overrides.update({key: value for key, value in defaults.items() if key not in runtime.public_config._explicit_keys})
    if train_dataset is not None or eval_dataset is not None:
      overrides["dataset_type"] = "hf"
    overrides.update(dataset_overrides(train_dataset))
    if args is not None:
      overrides.update(args._overrides())
    overrides.update(callback_overrides(callbacks))
    execution = runtime.public_config.get_source_values() | runtime._overrides | overrides
    if execution["use_multimodal"] and formatting_func is not None:
      raise ValueError("Multimodal SFT requires paired text and media columns. Apply dataset.map before the Trainer; "
                       "formatting_func returns text without a media or completion boundary.")
    if execution["enable_diloco"] or execution["use_multimodal"] or execution["optimizer_memory_host_offload"]:
      overrides["use_tunix_gradient_accumulation"] = False
    data_overrides = runtime._overrides | overrides
    explicit_data_fields = runtime.public_config._explicit_keys | data_overrides.keys()
    unsupported = sorted(set(MMAP_FIELDS) & explicit_data_fields)
    data_config = runtime.public_config.get_source_values() | data_overrides
    if data_config["grain_file_type"] in ("mmap", "mmap_npy") and (
        data_config["dataset_type"] == "grain" or "grain_file_type" in explicit_data_fields
    ):
      unsupported.append("grain_file_type")
    if unsupported:
      raise ValueError(f"SFT does not support these mmap data settings: {sorted(unsupported)}")
    self._training_overrides = overrides
    graph_config = model.existing_graph_config
    if graph_config is not None:
      self._check_precision_overrides(graph_config, overrides)
      config = resolve_initialized_config(
          runtime.public_config, runtime.config, overrides=runtime._overrides | overrides,
          train_dataset_provided=train_dataset is not None, eval_dataset_provided=eval_dataset is not None,
      )
    else:
      runtime.initialize(
          overrides=overrides, train_dataset_provided=train_dataset is not None,
          eval_dataset_provided=eval_dataset is not None,
      )
      self._check_precision_options(runtime.config, overrides)
      config = runtime.config
    _check_post_training_data_options(config, runtime, args)
    # W&B starts only through an explicit report_to request; the suppression stays out of the shared runtime.
    config = HyperParameters(config._pydantic_config.model_copy(update={"enable_wandb": False}))
    _PostTrainer.__init__(
        self, model, args, train_dataset, eval_dataset, processing_class, data_collator, optimizers, callbacks,
        resolved_config=config, defer_materialization=True,
    )
    self._uses_native_training = config.enable_diloco or config.use_multimodal or config.optimizer_memory_host_offload
    self._train_state = None
    self._setup = None
    self._graphdef = None
    self.model_init = None

  def train(self, resume_from_checkpoint=None, *, checkpoint_step=None):
    if self._uses_native_training:
      state = _NativeTrainer.train(self, resume_from_checkpoint, checkpoint_step=checkpoint_step)
      self.model.config.finetuning_task = "sft"
      return state
    result = _ManagedPostTrainer.train(self, resume_from_checkpoint, checkpoint_step=checkpoint_step)
    self.model.native_model = self.native_trainer.model
    self.model.config.finetuning_task = "sft"
    return result

  def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
    validate_eval_dataset(self.eval_dataset if eval_dataset is None else eval_dataset)
    if self._uses_native_training:
      return _NativeTrainer.evaluate(self, eval_dataset, ignore_keys, metric_key_prefix)
    return _ManagedPostTrainer.evaluate(self, eval_dataset, ignore_keys, metric_key_prefix)

  def _evaluate_iterator(self, eval_iterator, metric_key_prefix):
    if self._uses_native_training:
      return _NativeTrainer._evaluate_iterator(self, eval_iterator, metric_key_prefix)
    return _ManagedPostTrainer._evaluate_iterator(self, eval_iterator, metric_key_prefix)

  def save_model(self, output_dir=None):
    if self._uses_native_training:
      return _NativeTrainer.save_model(self, output_dir)
    return _ManagedPostTrainer.save_model(self, output_dir)

  def save_state(self):
    if self._uses_native_training:
      return _NativeTrainer.save_state(self)
    return _ManagedPostTrainer.save_state(self)


