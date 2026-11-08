# Checkpoint storage and recovery profile fields

These are YAML sections of an [execution profile](execution-profile.md). Every section is optional and only supplied keys override the workflow. Paths must refer to storage available to the participating processes.

## `storage`

Trainer checkpoint intervals and output directories still belong to the Python workflow. Storage fields do not enable periodic checkpoint saving by themselves.

Zarr3 requires OCDBT. Colocated checkpointing requires single-controller Pathways.

<details>
<summary>Parameters</summary>

- `checkpoint_storage_target_data_file_size_bytes` (`int`, optional, defaults to `2147483648`). Target file size for Orbax array chunks.
- `checkpoint_storage_use_ocdbt` (`bool`, optional, defaults to `true`). Use OCDBT storage.
- `checkpoint_storage_use_zarr3` (`bool`, optional, defaults to `true`). Use Zarr3 with OCDBT.
- `checkpoint_storage_concurrent_gb` (`int`, optional, defaults to `96`). Concurrent I/O budget in GB.
- `checkpoint_storage_device_host_concurrent_gb` (`int | None`, optional, defaults to `8`). Device-to-host staging budget in GB. Null is unbounded.
- `enable_single_replica_ckpt_restoring` (`bool`, optional, defaults to `false`). One replica reads and broadcasts.
- `checkpoint_todelete_subdir` (`str | None`, optional, defaults to `null`). Move checkpoints here before deletion. Ignored for `gs://` directories.
- `checkpoint_todelete_full_path` (`str | None`, optional, defaults to `null`). Full destination for moved checkpoints.
- `colocated_python_checkpointing` (`bool`, optional, defaults to `false`). Run checkpoint work on remote TPU VMs instead of the head node.

</details>

## `replication`

Emergency or multi-tier checkpointing requires a nonempty local directory and a positive local save period. Multi-tier mode requires exactly one backup interval. Step-based backups must be at least one local save period apart.

For a training workflow with persistent checkpoint saving enabled, this complete profile selects local saves and step-based backups. The intervals illustrate the required relationship, not a recommended save rate. Replace the local path with your deployment's local storage path.

```yaml
version: 1
process:
  hardware: tpu
replication:
  enable_multi_tier_checkpointing: true
  local_checkpoint_directory: /checkpoints/local
  local_checkpoint_period: 1
  multi_tier_checkpointing_backup_interval_steps: 1
```

<details>
<summary>Parameters</summary>

- `enable_multi_tier_checkpointing` (`bool`, optional, defaults to `false`). Keep local checkpoints and persistent backups.
- `local_checkpoint_directory` (`str`, optional, defaults to `""`). Local checkpoint path.
- `local_checkpoint_period` (`NonNegativeInt`, optional, defaults to `0`). Steps between local saves.
- `multi_tier_checkpointing_backup_interval_minutes` (`PositiveInt | None`, optional, defaults to `null`). Minutes between backups.
- `multi_tier_checkpointing_backup_interval_steps` (`PositiveInt | None`, optional, defaults to `null`). Steps between backups.
- `mtc_data_parallelism` (`int`, optional, defaults to `0`). Identical pipeline count. Zero uses the slice count.
- `enable_emergency_checkpoint` (`bool`, optional, defaults to `false`). Legacy local-checkpoint switch. New profiles use `enable_multi_tier_checkpointing`.
- `use_replicator_service` (`bool`, optional, defaults to `false`). Compatibility field. The current workflow does not read it to start a service.
- `replicator_backup_interval_minutes` (`NonNegativeInt`, optional, defaults to `0`). Unread compatibility interval. Active backup scheduling uses the multi-tier interval fields.

</details>

## `recovery`

Elastic training requires a Pathways single-controller deployment. It does not work with McJAX. The profile does not provision Pathways, slices, or persistent checkpoint storage.

Snapshot recovery requires synthetic data or Grain with `grain_use_elastic_iterator=True`. It rejects DiLoCo and ramp-up batch size. Use `checkpoint` for those workloads. A snapshot survives slice loss only for state replicated across slices. Optimizer sharding over data therefore falls back to persistent checkpoints.

The Grain elastic iterator requires `grain_file_type="arrayrecord"`, `packing=False`, and `use_truncation=True`. It rejects dataset mixtures, including semicolon-separated source patterns and a nonempty mixture configuration path. TFRecord and Parquet are not supported by that iterator.

This profile selects checkpoint-based recovery for an existing Pathways training script with persistent checkpoints configured in its trainer arguments.

```yaml
version: 1
process:
  hardware: tpu
  worker_arrangement: single_controller
recovery:
  elastic_enabled: true
  elastic_backup_kind: checkpoint
```

Use either profile with the shipped [Python launch setup](../../guides/execution/launch.md), for example `lilax launch --config-file recovery.yaml train.py` after saving the recovery profile as `recovery.yaml`.

Invalid backup kinds fail validation. Waiting beyond the active-slice timeout raises `TimeoutError`.

<details>
<summary>Parameters</summary>

- `elastic_enabled` (`bool`, optional, defaults to `false`). Retry slice loss or scale-up in single-controller mode.
- `elastic_backup_kind` (`str`, optional, defaults to `"snapshot"`). `snapshot` retains train state and iterator position in pinned host memory. `checkpoint` restores the latest persistent checkpoint only.
- `elastic_snapshot_interval` (`PositiveInt`, optional, defaults to `10`). Steps between snapshots.
- `elastic_timeout_seconds` (`int`, optional, defaults to `300`). Maximum wait for the minimum slice count per attempt.
- `elastic_max_retries` (`int`, optional, defaults to `10`). Retry limit.
- `elastic_min_slice_count` (`int`, optional, defaults to `-1`). Minimum active slices. `-1` waits for all slices.

</details>
