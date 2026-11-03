import os

from jinja2 import Environment, TemplateSyntaxError

from lilax._backend.utils.globals import MAXTEXT_ASSETS_ROOT
from lilax.configuration.domain_types import TokenizerType
from lilax.configuration.domains.execution_mesh import CustomRule


def resolve_inert_defaults(self):
  if self.custom_mesh_and_rule == CustomRule.DEFAULT and self.custom_mesh_and_rule_for_eval == CustomRule.DEFAULT:
    self.logical_axis_rules_for_eval = self.logical_axis_rules
  if (
      self.tokenizer_path and not os.path.exists(self.tokenizer_path) and not self.tokenizer_path.startswith("gs://")
      and not self.colocated_python_data_input
  ):
    self.tokenizer_path = next(
        filter(os.path.exists, (
            os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", os.path.basename(self.tokenizer_path)),
            os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", self.tokenizer_path),
        )),
        self.tokenizer_path,
    )
  if self.tokenizer_path and "tokenizer_llama3.tiktoken" in self.tokenizer_path and self.tokenizer_type != "tiktoken":
    self.tokenizer_type = TokenizerType.TIKTOKEN
  if self.chat_template:
    try:
      Environment().parse(self.chat_template)
    except TemplateSyntaxError as error:
      raise ValueError(f"Specified chat_template is invalid: {error}") from error
  if self.learning_rate_schedule_steps == -1:
    self.learning_rate_schedule_steps = self.steps
  if self.steps == -1:
    self.steps = self.learning_rate_schedule_steps
  if self.eval_per_device_batch_size == 0.0:
    self.eval_per_device_batch_size = self.per_device_batch_size
  if not self.mu_dtype:
    self.mu_dtype = self.weight_dtype
  if self.attn_logits_soft_cap == 0.0:
    self.attn_logits_soft_cap = None
  if self.final_logits_soft_cap == 0.0:
    self.final_logits_soft_cap = None
  if self.remat_policy == "custom":
    tensors = [
        "decoder_layer_input", "indexer_cutoff_threshold", "context", "mlpwi", "moe_mlpwi_0", "moe_x_sorted",
        "moe_routing_maps", "moe_tc_routing_checkpoint", "moe_mlpwi_1", "moe_mlpwo", "te_moe_dispatch_remat_location",
        "mlpwi_0", "mlpwi_1", "mlpwo", "query_proj", "key_proj", "value_proj", "kv_proj", "query_wa_proj",
        "kv_wa_proj", "mla_kv", "mla_q", "qkv_proj", "attention_out", "out_proj",
    ]
    self.tensors_on_device = [tensor for tensor in tensors if getattr(self, tensor) == "device"]
    self.tensors_to_offload = [tensor for tensor in tensors if getattr(self, tensor) == "offload"]
  for local_name, global_name in (
      ("local_sa_block_q", "sa_block_q"),
      ("local_sa_block_kv", "sa_block_kv"),
      ("local_sa_block_kv_compute", "sa_block_kv_compute"),
      ("eval_local_sa_block_q", "eval_sa_block_q"),
      ("eval_local_sa_block_kv", "eval_sa_block_kv"),
      ("eval_local_sa_block_kv_compute", "eval_sa_block_kv_compute"),
      ("local_sa_block_q_dkv", "sa_block_q_dkv"),
      ("local_sa_block_kv_dkv", "sa_block_kv_dkv"),
      ("local_sa_block_kv_dkv_compute", "sa_block_kv_dkv_compute"),
      ("local_sa_block_q_dq", "sa_block_q_dq"),
      ("local_sa_block_kv_dq", "sa_block_kv_dq"),
      ("local_sa_use_fused_bwd_kernel", "sa_use_fused_bwd_kernel"),
      ("local_sa_q_layout", "sa_q_layout"),
      ("local_sa_k_layout", "sa_k_layout"),
      ("local_sa_v_layout", "sa_v_layout"),
      ("eval_local_sa_q_layout", "eval_sa_q_layout"),
      ("eval_local_sa_k_layout", "eval_sa_k_layout"),
      ("eval_local_sa_v_layout", "eval_sa_v_layout"),
      ("local_use_splash_scheduler", "use_splash_scheduler"),
      ("local_sa_fuse_reciprocal", "sa_fuse_reciprocal"),
      ("local_sa_use_base2_exp", "sa_use_base2_exp"),
  ):
    if getattr(self, local_name) is None:
      setattr(self, local_name, getattr(self, global_name))
  return self
