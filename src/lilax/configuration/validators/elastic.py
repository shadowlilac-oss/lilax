from lilax._backend.configs.dataset_types import DatasetType


def validate_elastic_training_options(self):
  if self.elastic_enabled and not self.enable_single_controller:
    raise ValueError("Elastic training is only supported with Pathways (`enable_single_controller=True`).")
  if self.elastic_backup_kind not in ("snapshot", "checkpoint"):
    raise ValueError(
        "elastic_backup_kind must be one of 'snapshot' or 'checkpoint', got" f" '{self.elastic_backup_kind}'."
    )
  if self.elastic_enabled and self.elastic_backup_kind == "snapshot":
    if self.enable_diloco:
      raise ValueError("elastic_backup_kind='snapshot' is not supported with DiLoCo (`enable_diloco=True`).")
    if self.enable_rampup_batch_size:
      raise ValueError("elastic_backup_kind='snapshot' is not supported with `enable_rampup_batch_size=True`.")
    if self.dataset_type != DatasetType.SYNTHETIC and (
        self.dataset_type != DatasetType.GRAIN or not self.grain_use_elastic_iterator
    ):
      raise ValueError(
          "elastic_backup_kind='snapshot' requires `dataset_type=grain` with `grain_use_elastic_iterator=True`"
          " (or synthetic data), so that the data iterator position can be snapshotted with the train state."
      )
