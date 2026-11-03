"""Validation of elastic training backup configuration."""


def validate_elastic_backup(config, dataset_type):
  if config.elastic_enabled and not config.enable_single_controller:
    raise ValueError("Elastic training is only supported with Pathways (`enable_single_controller=True`).")
  if config.elastic_backup_kind not in ("snapshot", "checkpoint"):
    raise ValueError(
        "elastic_backup_kind must be one of 'snapshot' or 'checkpoint', got" f" '{config.elastic_backup_kind}'."
    )
  if config.elastic_enabled and config.elastic_backup_kind == "snapshot":
    if config.enable_diloco:
      raise ValueError("elastic_backup_kind='snapshot' is not supported with DiLoCo (`enable_diloco=True`).")
    if config.enable_rampup_batch_size:
      raise ValueError("elastic_backup_kind='snapshot' is not supported with `enable_rampup_batch_size=True`.")
    if config.dataset_type != dataset_type.SYNTHETIC:
      if config.dataset_type != dataset_type.GRAIN or not config.grain_use_elastic_iterator:
        raise ValueError(
            "elastic_backup_kind='snapshot' requires `dataset_type=grain` with `grain_use_elastic_iterator=True`"
            " (or synthetic data), so that the data iterator position can be snapshotted with the train state."
        )
