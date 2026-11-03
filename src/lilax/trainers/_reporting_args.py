"""Map reporting arguments to configuration overrides."""


def reporting_overrides(args):
  return (
      {"enable_tensorboard": "tensorboard" in args.report_to, "enable_wandb": False}
      if args.report_to is not None else {}
  )
