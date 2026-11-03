from lilax._backend.integration.tunix.weight_mapping.deepseek3 import DEEPSEEK_VLLM_MAPPING
from lilax._backend.integration.tunix.weight_mapping.gemma3 import GEMMA3_VLLM_MAPPING
from lilax._backend.integration.tunix.weight_mapping.gpt_oss import GPT_OSS_VLLM_MAPPING
from lilax._backend.integration.tunix.weight_mapping.llama3 import LLAMA3_VLLM_MAPPING
from lilax._backend.integration.tunix.weight_mapping.qwen2 import QWEN2_VLLM_MAPPING
from lilax._backend.integration.tunix.weight_mapping.qwen3 import QWEN3_VLLM_MAPPING


class StandaloneVllmWeightMapping:
  def __getattr__(self, name):
    if name.startswith("llama3.1"):
      return LLAMA3_VLLM_MAPPING
    elif name.startswith("qwen2"):
      return QWEN2_VLLM_MAPPING
    elif name.startswith("qwen3"):
      return QWEN3_VLLM_MAPPING
    elif name.startswith("gemma3"):
      return GEMMA3_VLLM_MAPPING
    elif name.startswith("deepseek3"):
      return DEEPSEEK_VLLM_MAPPING
    elif name.startswith("gpt-oss"):
      return GPT_OSS_VLLM_MAPPING
    raise ValueError(f"{name} vLLM weight mapping not found.")

  def __getitem__(self, key):
    return getattr(self, key)

  @classmethod
  def __class_getitem__(cls, key):
    instance = cls()
    return instance[key]
