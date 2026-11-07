"""Launch one Python process per node."""

import argparse
import os
import sys

from pydantic import ValidationError
import yaml

from ._process import ProcessConfig
from ._profile import ExecutionProfile, load_profile


def add_launch_command(commands):
  launch = commands.add_parser("launch", help="launch one Python process per node")
  launch.add_argument("--config-file", help="YAML execution profile written by lilax config")
  launch.add_argument("--hardware", choices=("cpu", "gpu", "tpu"))
  launch.add_argument("--num-nodes", type=int)
  launch.add_argument("--node-rank", type=int)
  launch.add_argument("--main-process-ip", help="IPv4 address of the coordinator (default: 127.0.0.1)")
  launch.add_argument("--main-process-port", type=int, help="coordinator port (default: 12355)")
  launch.add_argument("--gpu-ids", help="comma-separated physical GPU IDs")
  launch.add_argument("script", help="Python script to execute")
  launch.add_argument("script_args", nargs=argparse.REMAINDER)
  launch.set_defaults(handler=run, launch_parser=launch)


def run(args):
  launch = args.launch_parser
  try:
    profile = load_profile(args.config_file) if args.config_file else ExecutionProfile()
    process = profile.process.model_dump()
    for name, value in (
        ("hardware", args.hardware), ("num_nodes", args.num_nodes), ("node_rank", args.node_rank),
        ("coordinator_ip", args.main_process_ip), ("coordinator_port", args.main_process_port),
    ):
      if value is not None:
        process[name] = value
    if args.gpu_ids is not None:
      gpu_ids = args.gpu_ids.split(",")
      if any(not device.isascii() or not device.isdecimal() for device in gpu_ids):
        raise ValueError("--gpu-ids must be comma-separated nonnegative physical GPU IDs")
      process["gpu_ids"] = list(map(int, gpu_ids))
    profile.process = ProcessConfig.model_validate(process)
    profile.source_values()
  except (OSError, ValueError, ValidationError, yaml.YAMLError) as error:
    launch.error(str(error))
  os.environ.update(profile.process.environment())
  os.environ["LILAX_EXECUTION_PROFILE"] = profile.model_dump_json(exclude_unset=True)
  if profile.xla_flags is not None:
    os.environ.setdefault("XLA_FLAGS", profile.xla_flags)
  os.execv(sys.executable, [sys.executable, args.script, *args.script_args])
