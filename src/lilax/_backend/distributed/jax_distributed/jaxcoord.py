"""Explicit controller placement from the jaxcoord launcher."""

import os

import jax


def initialize_jaxcoord(raw_keys):
  visible = os.environ.get("CUDA_VISIBLE_DEVICES") if raw_keys["hardware"] == "gpu" else None
  devices = list(range(len(visible.split(",")))) if visible and visible != "-1" else None
  jax.distributed.initialize(
      coordinator_address=f'{os.environ["JAX_COORDINATOR_IP"]}:{os.environ["JAX_COORDINATOR_PORT"]}',
      num_processes=int(os.environ["NNODES"]), process_id=int(os.environ["NODE_RANK"]),
      local_device_ids=devices, cluster_detection_method="deactivate",
      initialization_timeout=raw_keys["jax_distributed_initialization_timeout"],
  )
