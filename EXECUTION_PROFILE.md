# Configure and launch execution

Run the conditional wizard to write a reusable YAML profile.

```sh
jaxcoord config --output jaxcoord.yaml
jaxcoord launch --config-file jaxcoord.yaml train.py
```

The wizard asks about hardware, node placement, local GPU devices, supported controller
arrangement, basic parallelism, compute dtype, recomputation, host offload, distributed
logging and DiLoCo. Enabling DiLoCo adds synchronization period, outer learning rate and
outer momentum questions. The file contains selected settings rather than every default.
The wizard preserves an existing file by refusing to overwrite it.

Launch once on each node. Each launcher replaces itself with one Python process using all
selected local devices. It does not spawn one process per GPU, provision machines or install
environments. Script arguments follow the script path.

```sh
jaxcoord launch --config-file jaxcoord.yaml --node-rank 1 train.py --dataset local.jsonl
```

CPU, GPU and TPU placement share this process model. TPU single-controller execution is
available for one controller node. Multi-node profiles require a coordinator IPv4 address
reachable from every node. Physical GPU IDs select CUDA visibility; JAX receives the
corresponding local device indices.

## YAML structure

This CPU profile is a small starting point.

```yaml
version: 1
process:
  hardware: cpu
  num_nodes: 1
  node_rank: 0
  worker_arrangement: one_process_per_node
precision:
  dtype: float32
remat:
  remat_policy: none
```

Expert settings can be added manually. Section and field names are validated. Omitted
fields retain the selected source recipe's defaults. Explicit `false`, zero and empty
values remain explicit. YAML null follows the declared field type; it is not a general
instruction to discard a setting.

| Section | Fields and owner |
|---|---|
| `process` | `hardware`, `num_nodes`, `node_rank`, `coordinator_ip`, `coordinator_port`, `gpu_ids`, `worker_arrangement` in [process placement](src/lilax/launch/_process.py). |
| `mesh` | Physical topology, scan layout, context strategy, custom mesh, partitioner and PRNG in [HardwareAndMesh](src/lilax/configuration/domains/execution_mesh.py). |
| `ici`, `dcn`, `pipeline` | Exact ICI/DCN axis degrees and pipeline scheduling fields in [parallelism declarations](src/lilax/configuration/domains/execution_parallelism.py). |
| `sharding` | Logical-axis rules, data sharding, FSDP placement and internal offline compilation mechanics in [LayoutAndSharding](src/lilax/configuration/domains/execution_sharding.py). |
| `precision` | Compute, gradient and stored-weight dtypes and matmul precision in [DataTypes](src/lilax/configuration/domains/model_data_types.py). Model quantization recipes remain Python/model-owned. |
| `remat` | Basic and per-tensor recomputation/offload in [RematAndOffload](src/lilax/configuration/domains/execution_remat.py). |
| `diloco` | Common DiLoCo selection and detailed transport, fragmentation and overlap fields in [DilocoParams](src/lilax/configuration/domains/data_diloco.py). |
| `execution` | `use_tunix_gradient_accumulation`, `max_inflight_computations`, `block_state_before_run_start`, `warm_input_reshard_before_run_start`, `warm_eval_input_reshard_before_run_start`, `colocated_python_data_input`. |
| `kernels`, `splash` | `ragged_block_size`, `force_q_layout`, and [SplashAttention](src/lilax/configuration/domains/model_splash.py) tuning. |
| `transport` | `replicate_quant_scale`, `te_comm_gemm_overlap`. These do not select a quantization recipe. |
| `compilation` | Saved train-step artifact and offline target topology in [AOT](src/lilax/configuration/domains/data_aot.py). |
| `debugging`, `hlo` | Initialization timeout/controller/subslice settings, cache, checkify, JAX logs and IR dumps in [debugging declarations](src/lilax/configuration/domains/data_debugging.py). |
| `storage` | [OrbaxStorage](src/lilax/configuration/domains/model_checkpointing.py) fields plus `enable_single_replica_ckpt_restoring`, `checkpoint_todelete_subdir`, `checkpoint_todelete_full_path`, `colocated_python_checkpointing`. |
| `recovery`, `replication` | [ElasticTraining](src/lilax/configuration/domains/execution_training.py) and [EmergencyCheckpointing](src/lilax/configuration/domains/model_checkpointing.py). |
| `profiling` | Exact step-window and accelerator profiler fields in [Profiling](src/lilax/configuration/domains/data_profiling.py). |
| `xprof` | `launch_server` and `port` select the existing TensorBoard/XProf UI for training. |
| `rl_profiling` | `profiler_max_num_hosts` selects Pathways RL profiler allocation and is consumed only by the RL runtime. |
| `metrics` | `record_internal_nn_metrics`, `prometheus_port`, `enable_checkpoint_cloud_logger`. |
| `goodput`, `diagnostics`, `monitoring` | The corresponding focused [monitoring declarations](src/lilax/configuration/domains/execution_monitoring.py). |
| `xla_flags` | Launcher XLA flags. An existing `XLA_FLAGS` environment value takes precedence. |

Basic parallelism questions set ICI degrees within a hardware slice. A single `-1` axis
receives remaining devices under the existing mesh resolver. Use manual `ici` and `dcn`
allocation for multislice placement and DiLoCo replica axes. A generic degree is not
silently reinterpreted as both ICI and DCN allocation.

`internal_compile` is an offline topology mechanic. It is not a generic switch that
enables JIT. Ordinary training retains its backend compilation behavior and does not run
the expensive batch-size or recomputation estimator automatically.

## Configuration precedence

Model loading and Trainer construction consume the profile before materialization.
Source recipe defaults are followed by profile values. Explicit Python model settings,
explicit checkpoint metadata and explicit Trainer execution controls take precedence over
profile defaults. Trainer algorithm settings own optimization, duration, batches and loss.
Launcher process flags override the YAML process section. Hardware placement comes from
the launcher.

Native model checkpoints store effective focused model configuration and separate
`scan_layers` and `param_scan_axis` metadata. Those checkpoint layout values retain the
saved weight layout. A profile does not relabel an existing model graph or mesh. Attempts
to change initialized mesh placement or incompatible graph precision are rejected.

Quantization recipes, Qwix QAT rules, model identity/revision, processors, dataset sources,
formatting, collators, checkpoint save frequency and resume selection remain in Python.
The YAML schema has no model quantization selector. Model precision in a checkpoint is
preserved; selecting an execution profile does not silently replace its quantization.

## Evaluation data

`SyntheticDataset` is an unbounded training source. Selecting it for startup, periodic or
manual evaluation raises a clear error requiring finite evaluation data. Finite Hugging
Face iterable datasets exhaust normally without a known length. Evaluation has no batch
cap and synthetic data has no length argument.
