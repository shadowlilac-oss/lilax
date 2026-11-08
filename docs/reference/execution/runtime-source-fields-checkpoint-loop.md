# Standalone checkpoint-loop source fields

The [checkpoint utilities](../checkpoints/checkpoint-utilities.md) use these fields on the [runtime-source object](runtime-source-fields.md). They are storage benchmark controls, not trainer arguments. The loop mutates optimizer moments to exercise saves.

Source and destination fields are listed under [checkpoint sources and destinations](runtime-source-fields.md#checkpoint-sources-and-destinations). Storage format and concurrency fields are listed under [storage profile fields](execution-profile-recovery.md#storage).

Dropping the host page cache requires privileges and affects the entire host.

<details>
<summary>Parameters</summary>

- `standalone_checkpointer_per_step_interval` (`float`, optional, defaults to `0.0`). Seconds between iterations.
- `standalone_checkpointer_drop_page_cache_before_restore` (`bool`, optional, defaults to `False`). Drop the host page cache before restore.
- `standalone_checkpointer_enable_restore_in_loop` (`bool`, optional, defaults to `True`). Restore after each save to exercise reads and writes.
- `standalone_checkpointer_start_from_checkpoint` (`bool`, optional, defaults to `False`). Attempt restore before initializing benchmark state.

</details>
