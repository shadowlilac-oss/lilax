"""Preference training through the Tunix path."""

from lilax._backend.trainers.post_train.dpo import train_dpo

from ..training_args import MMAP_FIELDS
from .arguments import _PostTrainingArguments
from .data_options import _check_post_training_data_options
from .managed import _ManagedPostTrainer


class DPOConfig(_PostTrainingArguments):
  pass


class DPOTrainer(_ManagedPostTrainer):
  _native_train = staticmethod(train_dpo.train)
  _tunix_config = staticmethod(train_dpo.get_tunix_config)

  def __init__(self, model, args=None, *trainer_args, **kwargs):
    if args is not None:
      unsupported = [name for name in MMAP_FIELDS if name in args._overrides()]
      if args._overrides().get("grain_file_type") in ("mmap", "mmap_npy"):
        unsupported.append("grain_file_type")
      if unsupported:
        raise ValueError(f"Native DPO does not support these data overrides: {sorted(unsupported)}")
    super().__init__(model, args, *trainer_args, defer_materialization=True, **kwargs)
    _check_post_training_data_options(self.config, model.runtime, args)
