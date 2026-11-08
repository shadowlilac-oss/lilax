# Launch a Python training script

Use `lilax config` and `lilax launch` to set up execution for your Python workflow. Keep model loading, datasets, and trainer calls in your script.

## Create a profile

From the project environment described in [environment setup](../../getting-started/environment.md), run:

```bash
lilax config --output jaxcoord.yaml
```

Answer the hardware, node placement, parallelism, precision, and memory questions. The command writes a new YAML file and refuses to overwrite an existing file. Choose another output path to keep both profiles.

Alternatively, save this minimal single-node TPU profile as `jaxcoord.yaml`:

```yaml
version: 1
process:
  hardware: tpu
```

For GPU execution, set `hardware: gpu`. Optional `gpu_ids: [0, 1]` selects those physical local GPUs. Without `gpu_ids`, the launcher leaves device visibility unchanged.

## Launch your script

Save the Python training workflow from the [SFT guide](../training/sft.md) as `train.py`. The following command executes that script, including its training and checkpoint writes:

```bash
lilax launch --config-file jaxcoord.yaml train.py
```

The launcher uses the current environment's Python interpreter and runs one process on this node. Arguments after `train.py` go to the script. It does not submit remote jobs.

## Launch on multiple nodes

Install the same environment and place `train.py` on every node. Save this two-node TPU profile on each node. Replace `10.0.0.10` with the coordinator's IPv4 address reachable from both nodes:

```yaml
version: 1
process:
  hardware: tpu
  num_nodes: 2
  node_rank: 0
  coordinator_ip: 10.0.0.10
  coordinator_port: 12355
```

These commands execute the training script on each node. Run the first on node zero and the second on node one:

```bash
lilax launch --config-file jaxcoord.yaml --node-rank 0 train.py
lilax launch --config-file jaxcoord.yaml --node-rank 1 train.py
```

Use the same coordinator address, port, and node count everywhere. Each node needs a distinct rank below `num_nodes`. The launcher does not start processes on the other nodes for you.

## Select parallelism separately from the model

Add this section to allocate remaining devices within a hardware slice to FSDP:

```yaml
ici:
  ici_data_parallelism: 1
  ici_fsdp_parallelism: -1
  ici_tensor_parallelism: 1
```

An axis degree of `1` leaves that axis unpartitioned. `-1` lets runtime resolution assign the remaining devices to that axis. Use positive degrees that fit your topology when combining axes. `ici` describes degrees within a slice, while `dcn` describes degrees across slices. Node count alone does not specify either layout.

Keep model architecture in model configuration and training choices in trainer arguments. Use [Python distributed controls](runtime.md) for sharding behavior. Physical placement and parallelism degrees belong in this reusable execution profile.

See [execution-profile reference](../../reference/execution/execution-profile.md) for options, defaults, and validation rules. The workflow follows the configuration-then-launch organization of [Accelerate's launch guide](https://huggingface.co/docs/accelerate/basic_tutorials/launch), using Lilax's own options.
