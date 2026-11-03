def resolve_mesh_parallelism(self):
  for prefix in ("ici", "dcn"):
    axis_map = {
        "diloco": getattr(self, f"{prefix}_diloco_parallelism"),
        "data": getattr(self, f"{prefix}_data_parallelism"),
        "stage": getattr(self, f"{prefix}_pipeline_parallelism"),
        "fsdp": getattr(self, f"{prefix}_fsdp_parallelism"),
        "fsdp_transpose": getattr(self, f"{prefix}_fsdp_transpose_parallelism"),
        "sequence": getattr(self, f"{prefix}_sequence_parallelism"),
        "context": getattr(self, f"{prefix}_context_parallelism"),
        "context_usp_ulysses": getattr(self, f"{prefix}_context_usp_ulysses_parallelism"),
        "context_autoregressive": getattr(self, f"{prefix}_context_autoregressive_parallelism"),
        "tensor": getattr(self, f"{prefix}_tensor_parallelism"),
        "tensor_sequence": getattr(self, f"{prefix}_tensor_sequence_parallelism"),
        "model": getattr(self, f"{prefix}_tensor_parallelism"),
        "expert": getattr(self, f"{prefix}_expert_parallelism"),
        "autoregressive": getattr(self, f"{prefix}_autoregressive_parallelism"),
        "attn_dp": 1, "attn_dp_expert": 1, "dcp": 1, "pcp": 1,
    }
    configured = getattr(self, f"{prefix}_parallelism")
    if configured and len(configured) == len(self.mesh_axes) and any(axis not in axis_map for axis in self.mesh_axes):
      parallelism = list(configured)
    else:
      parallelism = [axis_map.get(axis, 1) for axis in self.mesh_axes]
    setattr(self, f"{prefix}_parallelism", parallelism)
