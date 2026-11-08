# Execution profiles and launch options

An execution profile is a sparse YAML mapping for `lilax launch`. It describes placement and execution choices shared by Python workflows. See [launch a Python training script](../../guides/execution/launch.md).

## Profile format

All domain sections default to `None`, so omitted sections contribute no overrides. Within a selected section, only explicitly supplied fields contribute overrides. Unknown sections and unknown section fields are rejected.

Multiple nodes require a coordinator IPv4 address reachable from every node, excluding loopback and unspecified addresses. GPU selection sets local device visibility and is accepted only for GPU hardware. Single-controller placement requires one TPU controller node.

<details>
<summary>Parameters</summary>

- `version` (`Literal[1]`, optional, defaults to `1`). Only version `1` is accepted.
- `process` (`ProcessConfig`, optional, defaults to `ProcessConfig()`). Controller placement mapping with fields below. A factory creates each default configuration.
- `process.hardware` (`Literal["cpu", "gpu", "tpu"]`, optional, defaults to `"gpu"`). Hardware kind.
- `process.num_nodes` (`int`, optional, defaults to `1`). Positive controller node count.
- `process.node_rank` (`int`, optional, defaults to `0`). Nonnegative rank strictly below `num_nodes`.
- `process.coordinator_ip` (`str`, optional, defaults to `"127.0.0.1"`). Coordinator IPv4 address.
- `process.coordinator_port` (`int`, optional, defaults to `12355`). Port in `[1, 65535]`.
- `process.gpu_ids` (`list[int] | None`, optional, defaults to `null`). Nonempty, distinct, nonnegative physical GPU IDs.
- `process.worker_arrangement` (`Literal["one_process_per_node", "single_controller"]`, optional, defaults to `"one_process_per_node"`). Uses all selected local devices. Alternative `single_controller` selects Pathways.

</details>

## Configuration and launch commands

`lilax config --output jaxcoord.yaml` asks interactive setup questions and creates a new file. `--output` defaults to `jaxcoord.yaml`. Existing destinations are preserved.

`lilax launch [options] script [script_args ...]` replaces the launcher with the current Python interpreter running the script. It launches only the current node.

Launch options precede the script path. Arguments after the script path belong to the script. Process overrides are validated again before execution.

<details>
<summary>Parameters</summary>

- `--output` (`str`, optional, defaults to `"jaxcoord.yaml"`). New profile destination for the config command only. Existing files are preserved.
- `--config-file` (`str | None`, optional, defaults to `None`). Profile YAML path. Omission uses default process settings without automatic file lookup.
- `--hardware` (`str | None`, optional, defaults to `None`). Override for `process.hardware`, accepting `cpu`, `gpu`, or `tpu`.
- `--num-nodes` (`int | None`, optional, defaults to `None`). Node-count override.
- `--node-rank` (`int | None`, optional, defaults to `None`). Node-rank override.
- `--main-process-ip` (`str | None`, optional, defaults to `None`). Coordinator IPv4 address override.
- `--main-process-port` (`int | None`, optional, defaults to `None`). Coordinator port override.
- `--gpu-ids` (`str | None`, optional, defaults to `None`). Local GPU override as comma-separated integers, such as `0,1`.
- `script` (`str`). Python script path required by launch.
- `script_args` (`list[str]`, optional, defaults to `[]`). Arguments passed to the script.

</details>

## Checkpoint conversion command

`lilax convert source destination [options]` imports supported Hugging Face weights or converts the projection layout of a Lilax `save_pretrained` weight-only checkpoint on CPU. It prints the destination path. The destination must not exist. Full training checkpoints are not accepted for projection-layout conversion.

```sh
lilax convert ./weights ./weights-fused --fused-qkv --fused-mlp
```

The conversion writes a new checkpoint and configuration. Layout support depends on the model family. Omitted layout flags preserve the source selection. Lilax projection-layout conversion rejects Qwix QAT, stored GPT-OSS prefused MoE weights, unsupported routed MLP layouts, and fused QKV with shared KV projections or layers. The destination must not be inside the source checkpoint.

<details>
<summary>Parameters</summary>

- `source` (`str`). Supported Hugging Face model source or Lilax weight-only checkpoint path.
- `destination` (`str`). New checkpoint directory.
- `--fused-qkv` (`bool | None`, optional, defaults to `None`). Fused QKV selection. `--no-fused-qkv` selects separate projections.
- `--fused-mlp` (`bool | None`, optional, defaults to `None`). Fused MLP selection. `--no-fused-mlp` selects separate projections.

</details>

## Parallelism and domain sections

`ici` and `dcn` accept their respective `ici_*_parallelism` and `dcn_*_parallelism` degrees. Axes cover data replication, FSDP, FSDP transpose, tensor, tensor sequence, sequence, context, USP Ulysses context, autoregressive context, pipeline, expert, autoregressive, and DiLoCo placement. `ici` applies within a hardware slice and `dcn` across slices.

Degrees must be positive integers or `-1`. At most one explicitly selected degree per section can be `-1`. The underlying defaults assign remaining ICI devices to FSDP and remaining DCN slices to data parallelism. Other axis defaults are `1`. Omitted fields remain subject to model and runtime configuration rather than being added as profile overrides.

`pipeline` controls pipeline scheduling. Its [field reference](execution-profile-placement.md#pipeline) lists layer, repeat, and microbatch defaults and derivation rules. Pipeline degrees belong in `ici` or `dcn`.

Other optional sections are grouped by the execution behavior they control:

| Sections | Purpose and related reference |
| --- | --- |
| `mesh`, `ici`, `dcn`, `pipeline`, `sharding` | [Placement and sharding YAML fields](execution-profile-placement.md) |
| `precision`, `remat`, `execution`, `diloco`, `transport`, `compilation` | [Execution and memory YAML fields](execution-profile-compute.md) |
| `kernels`, `splash` | [Attention kernel YAML fields](execution-profile-kernels.md) |
| `storage`, `replication`, `recovery` | [Checkpoint and recovery YAML fields](execution-profile-recovery.md) |
| `debugging`, `hlo`, `profiling`, `rl_profiling`, `xprof` | [Debugging and profiling YAML fields](execution-profile-profiling.md) |
| `metrics`, `goodput`, `diagnostics`, `monitoring` | [Metrics and monitoring YAML fields](execution-profile-monitoring.md) |

`xla_flags` defaults to `None`. A supplied string sets `XLA_FLAGS` only when the environment does not already define it. Duplicate settings supplied through multiple sections are rejected.

Explicit model or trainer settings take precedence over profile defaults for the same runtime setting. Hardware always comes from the process profile. Incompatible changes to an initialized mesh are rejected. Keep profile placement separate from model architecture and trainer objectives.

The field references describe declared section defaults. They do not mean that selecting one field inserts all other defaults from its section. YAML uses `true`, `false`, and `null` for Python `True`, `False`, and `None`.

<details>
<summary>Parameters</summary>

- `mesh` (`HardwareAndMesh | None`, optional, defaults to `null`). Sparse placement mapping.
- `ici` (`IciParallelism | None`, optional, defaults to `null`). Sparse within-slice degrees.
- `dcn` (`DcnParallelism | None`, optional, defaults to `null`). Sparse across-slice degrees.
- `pipeline` (`PipelineParallelism | None`, optional, defaults to `null`). Sparse pipeline scheduling mapping.
- `sharding` (`LayoutAndSharding | None`, optional, defaults to `null`). Sparse sharding mapping.
- `precision` (`DataTypes | None`, optional, defaults to `null`). Sparse precision mapping.
- `remat` (`RematAndOffload | None`, optional, defaults to `null`). Sparse rematerialization mapping.
- `execution` (`ExecutionTuning | None`, optional, defaults to `null`). Sparse execution mapping.
- `diloco` (`DilocoParams | None`, optional, defaults to `null`). Sparse DiLoCo mapping.
- `transport` (`QuantizationTransport | None`, optional, defaults to `null`). Sparse quantization transport mapping.
- `compilation` (`AOT | None`, optional, defaults to `null`). Sparse compilation mapping.
- `kernels` (`AttentionTuning | None`, optional, defaults to `null`). Sparse attention tuning mapping.
- `splash` (`SplashAttention | None`, optional, defaults to `null`). Sparse Splash mapping.
- `storage` (`CheckpointStorage | None`, optional, defaults to `null`). Sparse checkpoint storage mapping.
- `replication` (`EmergencyCheckpointing | None`, optional, defaults to `null`). Sparse checkpoint replication mapping.
- `recovery` (`ElasticTraining | None`, optional, defaults to `null`). Sparse elastic recovery mapping.
- `debugging` (`DevelopmentAndDebugging | None`, optional, defaults to `null`). Sparse debugging mapping.
- `hlo` (`HloDump | None`, optional, defaults to `null`). Sparse dump mapping.
- `profiling` (`Profiling | None`, optional, defaults to `null`). Sparse profiling mapping.
- `rl_profiling` (`RLProfilingPlacement | None`, optional, defaults to `null`). Sparse RL profiling mapping.
- `xprof` (`XProfServer | None`, optional, defaults to `null`). Sparse analysis server mapping.
- `metrics` (`MetricsTuning | None`, optional, defaults to `null`). Sparse metrics mapping.
- `goodput` (`Goodput | None`, optional, defaults to `null`). Sparse goodput mapping.
- `diagnostics` (`ManagedMLDiagnostics | None`, optional, defaults to `null`). Sparse managed diagnostics mapping.
- `monitoring` (`GcpMonitoring | None`, optional, defaults to `null`). Sparse workload monitoring mapping.
- `xla_flags` (`str | None`, optional, defaults to `null`). Value for `XLA_FLAGS` when the environment does not define it.

</details>
