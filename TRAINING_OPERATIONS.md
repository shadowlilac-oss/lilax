# Training operations

This guide describes the current source, not end-to-end verified behavior. Platform support,
installation, and interactions between these options have not been verified at runtime.

## Container installation

`Dockerfile.cuda` uses a pinned plain NVIDIA JAX image for amd64 and runs `install_ngc.py` to
guard observed JAX, Transformer Engine, and native paths during installation. The guard does
not cover every system library or dynamic-loader binding, and nested directory symlinks can
cause it to fail. The base retains a CUTLASS cu12 helper package; its presence does not establish
that a CUDA 12 runtime is active.

`Dockerfile.tpu` targets amd64. TPU support on ARM is not established: the [JAX platform
table](https://docs.jax.dev/en/latest/installation.html#supported-platforms) lists Google Cloud
TPU on Linux aarch64 as unavailable, and the [libtpu 0.0.51 release](https://pypi.org/project/libtpu/0.0.51/)
lists x86_64 wheels. The CUDA recipe has been built and used for the limited run in the README;
the TPU recipe has not been built or its dependency installation resolved.

## Multi-node GPU launch

`jaxcoord launch` starts one Python process on the node where it is invoked. It does not
connect to other nodes, start remote processes over SSH, or activate a virtual environment.
Start it separately on every node from the intended environment, with the script and its
dependencies available on each node. The launcher replaces itself with the Python interpreter
associated with `jaxcoord`.

For two nodes, pass `--num-nodes 2` on both, `--node-rank 0` on one and `--node-rank 1` on
the other, and the same `--main-process-ip` and `--main-process-port` on both. The IP must be
an IPv4 address reachable from both nodes; loopback and unspecified addresses are rejected
for multi-node launches. The port defaults to `12355`. Coordinate the launches and ensure the
coordinator address and port are reachable between nodes.

`--gpu-ids` optionally sets `CUDA_VISIBLE_DEVICES` to a comma-separated list of distinct,
nonnegative physical GPU IDs on each node. Omit it to leave that variable unchanged. The
launcher sets the node count, rank, coordinator address and GPU hardware environment variables;
MaxText then initializes JAX with one distributed process per node. Arguments after the script
are passed to that script. This launch path has not been execution-verified.

## Quantization

`AutoModelForCausalLM.from_pretrained` and `from_config` accept a `quantization_config` for
model-level Qwix quantization-aware training. Use `QwixQATConfig` and `QwixQATRule` from
`lilax.configuration`: the config requires `rules` and selects `provider="odml"`
by default or `provider="qt"`. Rules specify fields including `module_path`, `op_names`,
`weight_qtype`, `act_qtype`, calibration settings, and optional `tile_size`. ODML rejects
`tile_size`; Qt passes the rules to `qwix.QtProvider` and rejects non-default ODML-only options
(`disable_per_channel_weights`, `fixed_range_for_inputs`, `fixed_range_for_outputs`, and `strict`).
The model is quantized when materialized. An existing Qwix
precision recipe cannot be combined with this model-level QAT configuration.

Training precision is configured separately through `TrainingArguments` and its derived
configs. For example, `quantization="te_nvfp4_no_rht"` selects a Transformer Engine NVFP4
block-scaling recipe with `disable_rht=True`; it is not a Qwix QAT provider. These settings do
not establish hardware or model compatibility.

## Metrics and profiling

`report_to` accepts `"trackio"`, `"wandb"`, `"tensorboard"`, a list of those names,
`"none"`, or `None`. Explicit selections override the native TensorBoard and W&B flags;
`None` leaves existing configuration in place. Trackio and W&B reporting is wired into native
pre-training, native SFT, Tunix SFT and DPO, distillation, GRPO, and experimental native GRPO.
TensorBoard uses the native metrics path for pre-training, native SFT, and experimental native
GRPO, and the reporter writer for distillation and GRPO. Tunix SFT and DPO do not create that
reporter writer; an explicit `"tensorboard"` selection enables the native MetricLogger writer
on process zero, except for decoupled training. Their default Tunix metrics logger is configured
only when `report_to` is `None`. This explicit reporting path has not been runtime-verified. Reporting
runs on process zero; backend availability and emitted metrics depend on the training path.
For GRPO, `"none"` or `[]` disables selected reporting backends; managed ML Diagnostics retains
its backend. `None` retains the original defaults.

`launch_xprof=True` starts TensorBoard for the duration of the training attempt on process zero,
using `tensorboard_dir` and `xprof_port` (default `8791`). It binds only to `127.0.0.1` and
does not expose the UI remotely. For remote viewing, use a secure port-forward to the training
node; protect access to the forwarded UI and its potentially sensitive profile data. Starting
the UI does not itself enable trace capture. Configure profiling separately, for example with
the `profiler`, `skip_first_n_steps_for_profiler`, `profiler_steps`, and
`profile_periodically_period` configuration fields.

The native pre-training and experimental native GRPO loops wire completed XPlane captures to
Trackio/W&B artifacts when the relevant reporter is active. The callback runs for completed
local JAX XPlane traces, not managed ML Diagnostics captures; it is not a general artifact
export for every trainer or profiler. Experimental native GRPO also queues inference captures
under its source-specific conditions. Artifacts contain the capture log directory and stop-step
metadata; `gs://` log directories are staged locally under `~/tmp/lilax` while uploading.
Tunix SFT/DPO also has an authored artifact path for cooperative, exclusive recording: other
tools must not control the trace during the run. Capture stops at the next data fetch or training
exit; incomplete or error-exit captures are not uploaded. XProf UI discovery of the nested
capture directories is unverified.
Capture, upload, and remote access have not been runtime-verified.
