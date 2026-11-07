"""Common, conditional questions for a reusable execution profile."""

from ._profile import ExecutionProfile, save_profile


def _ask(question, default, convert=str, choices=None):
  while True:
    answer = input(f"{question} [{default}]: ").strip()
    try:
      value = convert(answer if answer else str(default))
      if choices is not None and value not in choices:
        raise ValueError(f"Choose one of {', '.join(map(str, choices))}")
      return value
    except ValueError as error:
      print(error)


def _yes(question, default=False):
  return _ask(question + " (yes/no)", "yes" if default else "no", choices=("yes", "no")) == "yes"


def configure(output):
  print("Configure execution on this node. Run the launcher once on each node.")
  hardware = _ask("Hardware (cpu/gpu/tpu)", "gpu", choices=("cpu", "gpu", "tpu"))
  process = {"hardware": hardware}
  if _yes("Use multiple nodes"):
    process["num_nodes"] = _ask("Number of nodes", 2, int)
    process["node_rank"] = _ask("This node's rank", 0, int)
    process["coordinator_ip"] = _ask("Coordinator IPv4 address", "")
    process["coordinator_port"] = _ask("Coordinator port", 12355, int)
  if hardware == "gpu":
    devices = _ask("Physical GPU IDs, comma-separated, or all", "all")
    if devices != "all":
      process["gpu_ids"] = [int(device.strip()) for device in devices.split(",")]
  if hardware == "tpu" and process.get("num_nodes", 1) == 1:
    process["worker_arrangement"] = _ask(
        "Controller arrangement (one_process_per_node/single_controller)", "one_process_per_node",
        choices=("one_process_per_node", "single_controller"),
    )
  else:
    process["worker_arrangement"] = "one_process_per_node"
    print("Worker arrangement uses one process per node and all selected local devices.")

  settings = {"version": 1, "process": process}
  strategy = _ask("Parallelism (auto/data/fsdp/hybrid)", "auto", choices=("auto", "data", "fsdp", "hybrid"))
  if strategy != "auto":
    print("Degrees apply within each hardware slice. -1 assigns remaining devices to one axis.")
    ici = {"ici_data_parallelism": 1, "ici_fsdp_parallelism": 1}
    if strategy == "data":
      ici["ici_data_parallelism"] = _ask("Data parallel degree", -1, int)
    elif strategy == "fsdp":
      ici["ici_fsdp_parallelism"] = _ask("FSDP degree", -1, int)
    else:
      ici["ici_data_parallelism"] = _ask("Data parallel degree", 1, int)
      ici["ici_fsdp_parallelism"] = _ask("FSDP degree", -1, int)
      ici["ici_tensor_parallelism"] = _ask("Tensor parallel degree", 1, int)
      ici["ici_context_parallelism"] = _ask("Context parallel degree", 1, int)
    if _yes("Enable pipeline parallelism"):
      ici["ici_pipeline_parallelism"] = _ask("Pipeline parallel degree", 1, int)
    settings["ici"] = ici

  precision = _ask("Compute dtype (bfloat16/float16/float32)", "bfloat16",
                   choices=("bfloat16", "float16", "float32"))
  settings["precision"] = {"dtype": precision}
  settings["remat"] = {
      "remat_policy": _ask("Activation recomputation (full/minimal/none)", "full",
                           choices=("full", "minimal", "none")),
  }
  if hardware != "cpu":
    settings["remat"]["parameter_memory_host_offload"] = _yes("Offload model parameters to host memory")
    settings["remat"]["optimizer_memory_host_offload"] = _yes("Offload optimizer state to host memory")
  if _yes("Enable distributed debug logging"):
    settings["debugging"] = {"jax_debug_log_modules": "jax._src.distributed"}
  if _yes("Enable DiLoCo"):
    settings["diloco"] = {
        "enable_diloco": True, "diloco_sync_period": _ask("DiLoCo synchronization period", 36, int),
        "diloco_outer_lr": _ask("DiLoCo outer learning rate", 0.3, float),
        "diloco_outer_momentum": _ask("DiLoCo outer momentum", 0.9, float),
    }
    print("Set replica placement in the YAML ici/dcn diloco axes for your hardware topology.")
  profile = ExecutionProfile.model_validate(settings)
  save_profile(profile, output)
  print(f"Wrote {output}. Launch with jaxcoord launch --config-file {output} your_script.py")


def add_config_command(commands):
  command = commands.add_parser("config", help="write an execution profile through common setup questions")
  command.add_argument("--output", default="jaxcoord.yaml", help="new YAML profile path (default: jaxcoord.yaml)")
  command.set_defaults(handler=lambda args: configure(args.output))
