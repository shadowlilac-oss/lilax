"""Launch one Python process per GPU node."""

import argparse
import ipaddress
import os
import sys


def main():
  parser = argparse.ArgumentParser(prog="jaxcoord")
  commands = parser.add_subparsers(dest="command", required=True)
  launch = commands.add_parser("launch", help="launch one Python process per node")
  launch.add_argument("--num-nodes", type=int, default=1)
  launch.add_argument("--node-rank", type=int, default=0)
  launch.add_argument("--main-process-ip", help="IPv4 address of the coordinator (default: 127.0.0.1)")
  launch.add_argument("--main-process-port", type=int, default=12355, help="coordinator port (default: 12355)")
  launch.add_argument("--gpu-ids", help="comma-separated physical GPU IDs")
  launch.add_argument("script", help="Python script to execute")
  launch.add_argument("script_args", nargs=argparse.REMAINDER)
  args = parser.parse_args()

  if args.num_nodes < 1 or not 0 <= args.node_rank < args.num_nodes:
    launch.error("--num-nodes must be positive and --node-rank must be in range")
  if not 1 <= args.main_process_port <= 65535:
    launch.error("--main-process-port must be between 1 and 65535")
  if args.num_nodes > 1 and args.main_process_ip is None:
    launch.error("--main-process-ip is required for multiple nodes")
  try:
    coordinator = ipaddress.IPv4Address(args.main_process_ip or "127.0.0.1")
  except ipaddress.AddressValueError:
    launch.error("--main-process-ip must be an IPv4 address")
  if args.num_nodes > 1 and (coordinator.is_loopback or coordinator.is_unspecified):
    launch.error("--main-process-ip must be reachable from every node")

  if args.gpu_ids is not None:
    gpu_ids = args.gpu_ids.split(",")
    if any(not gpu_id.isascii() or not gpu_id.isdecimal() for gpu_id in gpu_ids):
      launch.error("--gpu-ids must be comma-separated nonnegative physical GPU IDs")
    if len(set(map(int, gpu_ids))) != len(gpu_ids):
      launch.error("--gpu-ids must not contain duplicates")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

  os.environ.update({
      "M_HARDWARE": "gpu",
      "NNODES": str(args.num_nodes),
      "NODE_RANK": str(args.node_rank),
      "JAX_COORDINATOR_IP": str(coordinator),
      "JAX_COORDINATOR_PORT": str(args.main_process_port),
      "LILAX_JAXCOORD": "1",
  })
  os.execv(sys.executable, [sys.executable, args.script, *args.script_args])
