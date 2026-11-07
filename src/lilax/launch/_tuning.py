"""Focused execution-profile settings drawn from the backend declarations."""

from pydantic import BaseModel, ConfigDict

from lilax._backend.configs.inert_datasets import DatasetGeneral
from lilax.configuration.domains.data_rl import RL
from lilax.configuration.domains.execution_optimizer import Optimizer
from lilax.configuration.domains.execution_monitoring import Metrics
from lilax.configuration.domains.execution_training import TrainingLoop
from lilax.configuration.domains.model_attention import Attention
from lilax.configuration.domains.model_checkpointing import Checkpointing, OrbaxStorage
from lilax.configuration.domains.model_quantization import Quantization


class ExecutionTuning(BaseModel):
  model_config = ConfigDict(extra="forbid")
  use_tunix_gradient_accumulation: bool = Optimizer.model_fields["use_tunix_gradient_accumulation"]
  max_inflight_computations: int = TrainingLoop.model_fields["max_inflight_computations"]
  block_state_before_run_start: bool = TrainingLoop.model_fields["block_state_before_run_start"]
  warm_input_reshard_before_run_start: bool = TrainingLoop.model_fields["warm_input_reshard_before_run_start"]
  warm_eval_input_reshard_before_run_start: bool = TrainingLoop.model_fields["warm_eval_input_reshard_before_run_start"]
  colocated_python_data_input: bool = DatasetGeneral.model_fields["colocated_python_data_input"]


class MetricsTuning(BaseModel):
  model_config = ConfigDict(extra="forbid")
  record_internal_nn_metrics: int = Metrics.model_fields["record_internal_nn_metrics"]
  prometheus_port: int = Metrics.model_fields["prometheus_port"]
  enable_checkpoint_cloud_logger: bool = Metrics.model_fields["enable_checkpoint_cloud_logger"]


class AttentionTuning(BaseModel):
  model_config = ConfigDict(extra="forbid")
  ragged_block_size: int = Attention.model_fields["ragged_block_size"]
  force_q_layout: bool = Attention.model_fields["force_q_layout"]


class CheckpointStorage(OrbaxStorage):
  model_config = ConfigDict(extra="forbid")
  enable_single_replica_ckpt_restoring: bool = Checkpointing.model_fields["enable_single_replica_ckpt_restoring"]
  checkpoint_todelete_subdir: str | None = Checkpointing.model_fields["checkpoint_todelete_subdir"]
  checkpoint_todelete_full_path: str | None = Checkpointing.model_fields["checkpoint_todelete_full_path"]
  colocated_python_checkpointing: bool = Checkpointing.model_fields["colocated_python_checkpointing"]


class QuantizationTransport(BaseModel):
  model_config = ConfigDict(extra="forbid")
  replicate_quant_scale: bool = Quantization.model_fields["replicate_quant_scale"]
  te_comm_gemm_overlap: str = Quantization.model_fields["te_comm_gemm_overlap"]


class XProfServer(BaseModel):
  model_config = ConfigDict(extra="forbid")
  launch_server: bool = False
  port: int = 8791


class RLProfilingPlacement(BaseModel):
  model_config = ConfigDict(extra="forbid")
  profiler_max_num_hosts: int = RL.model_fields["profiler_max_num_hosts"]
