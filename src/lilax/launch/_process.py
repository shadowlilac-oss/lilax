"""Process placement for a lilax launch."""

import ipaddress
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProcessConfig(BaseModel):
  """One controller process per node, with all selected local devices."""

  model_config = ConfigDict(extra="forbid")
  hardware: Literal["cpu", "gpu", "tpu"] = "gpu"
  num_nodes: int = Field(1, ge=1)
  node_rank: int = Field(0, ge=0)
  coordinator_ip: str = "127.0.0.1"
  coordinator_port: int = Field(12355, ge=1, le=65535)
  gpu_ids: list[int] | None = None
  worker_arrangement: Literal["one_process_per_node", "single_controller"] = "one_process_per_node"

  @model_validator(mode="after")
  def validate_placement(self):
    if self.node_rank >= self.num_nodes:
      raise ValueError("node_rank must be smaller than num_nodes")
    address = ipaddress.IPv4Address(self.coordinator_ip)
    if self.num_nodes > 1 and (address.is_loopback or address.is_unspecified):
      raise ValueError("coordinator_ip must be reachable from every node")
    if self.gpu_ids is not None:
      if self.hardware != "gpu":
        raise ValueError("gpu_ids is supported only for GPU execution")
      if not self.gpu_ids or any(device < 0 for device in self.gpu_ids):
        raise ValueError("gpu_ids must contain nonnegative physical GPU IDs")
      if len(set(self.gpu_ids)) != len(self.gpu_ids):
        raise ValueError("gpu_ids must not contain duplicates")
    if self.worker_arrangement == "single_controller" and (self.hardware != "tpu" or self.num_nodes != 1):
      raise ValueError("single_controller requires one TPU controller node")
    return self

  def environment(self):
    values = {
        "M_HARDWARE": self.hardware, "NNODES": str(self.num_nodes), "NODE_RANK": str(self.node_rank),
        "JAX_COORDINATOR_IP": self.coordinator_ip, "JAX_COORDINATOR_PORT": str(self.coordinator_port),
        "LILAX_JAXCOORD": "1",
    }
    if self.gpu_ids is not None:
      values["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))
    if self.hardware == "cpu":
      values["JAX_PLATFORMS"] = "cpu"
    return values
