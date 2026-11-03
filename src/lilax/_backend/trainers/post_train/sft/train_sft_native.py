from lilax._backend.configs.pyconfig import HyperParameters
from lilax._backend.configs.runtime import resolve_initialized_config


class NativeSFTArguments:
  def __init__(self, args, lora, overrides):
    self.args = args
    self.report_to = args.report_to if args is not None else None
    self.launch_xprof = args.launch_xprof if args is not None else False
    self.xprof_port = args.xprof_port if args is not None else 8791
    self.lora = lora
    self.overrides = overrides

  def _overrides(self):
    return self.overrides

  def resolve(self, runtime, *, train_dataset_provided=False, eval_dataset_provided=False):
    config = resolve_initialized_config(
        runtime.public_config, runtime.config, overrides=runtime._overrides | self.overrides,
        train_dataset_provided=train_dataset_provided, eval_dataset_provided=eval_dataset_provided,
    )
    overrides = {"use_sft": True, "use_tunix_gradient_accumulation": False}
    if self.lora is not None:
      overrides["lora"] = self.lora.model_copy(deep=True)
    return HyperParameters(config._pydantic_config.model_copy(update=overrides))


def train(trainer, resume_from_checkpoint=None):
  return trainer.train(resume_from_checkpoint)
