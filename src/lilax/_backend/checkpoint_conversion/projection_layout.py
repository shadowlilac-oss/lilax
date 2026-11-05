"""Select source-established projection layout adapters."""

from lilax._backend.checkpoint_conversion.model_families.llama31 import fused as fused_llama31
from lilax._backend.checkpoint_conversion.model_families.llama4 import fused as fused_llama4
from lilax._backend.checkpoint_conversion.model_families.gemma2.param_mapping import GEMMA2_MAXTEXT_TO_HF_PARAM_MAPPING
from lilax._backend.checkpoint_conversion.model_families.gemma3.param_mapping import GEMMA3_MAXTEXT_TO_HF_PARAM_MAPPING
from lilax._backend.checkpoint_conversion.model_families.gemma4 import fused as fused_gemma
from lilax._backend.checkpoint_conversion.model_families.gemma4.param_mapping import (
    GEMMA4_MAXTEXT_TO_HF_PARAM_MAPPING, GEMMA4_SMALL_MAXTEXT_TO_HF_PARAM_MAPPING,
)
from lilax._backend.checkpoint_conversion.model_families.llama31.param_mapping import LLAMA31_MAXTEXT_TO_HF_PARAM_MAPPING
from lilax._backend.checkpoint_conversion.model_families.param_mapping_registry import PARAM_MAPPING
from lilax._backend.checkpoint_conversion.model_families.gpt_oss import fused as fused_gpt_oss
from lilax._backend.checkpoint_conversion.model_families.gpt_oss.param_mapping import GPT_OSS_MAXTEXT_TO_HF_PARAM_MAPPING
from lilax._backend.checkpoint_conversion.model_families.mixtral import fused as fused_mixtral
from lilax._backend.checkpoint_conversion.model_families.mixtral.param_mapping import MIXTRAL_MAXTEXT_TO_HF_PARAM_MAPPING
from lilax._backend.checkpoint_conversion.model_families.olmo3 import fused as fused_olmo3
from lilax._backend.checkpoint_conversion.model_families.olmo3.param_mapping import OLMO3_MAXTEXT_TO_HF_PARAM_MAPPING
from lilax._backend.checkpoint_conversion.model_families.qwen import fused as fused_qwen
from lilax._backend.checkpoint_conversion.model_families.qwen.param_mapping import QWEN_MAXTEXT_TO_HF_PARAM_MAPPING
from lilax._backend.checkpoint_conversion.model_families.qwen3_next import fused as fused_qwen3_next
from lilax._backend.checkpoint_conversion.model_families.qwen3_next.param_mapping import QWEN3_NEXT_MAXTEXT_TO_HF_PARAM_MAPPING
from lilax._backend.checkpoint_conversion.model_families.qwen3_5 import fused as fused_qwen3_5
from lilax._backend.checkpoint_conversion.model_families.qwen3_5.param_mapping import QWEN3_5_MAXTEXT_TO_HF_PARAM_MAPPING
from lilax._backend.checkpoint_conversion.model_families.deepseek import fused as fused_deepseek
from lilax._backend.checkpoint_conversion.model_families.deepseek.param_mapping import DEEPSEEK_MAXTEXT_TO_HF_PARAM_MAPPING
from lilax._backend.checkpoint_conversion.model_families.deepseek_v4.param_mapping import DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_MAPPING
from lilax._backend.checkpoint_conversion.model_families.gpt3 import fused as fused_gpt3
from lilax._backend.checkpoint_conversion.model_families.qwen3_vl import fused as fused_qwen3_vl
from lilax._backend.checkpoint_conversion.model_families.qwen3_vl.param_mapping import QWEN3_VL_MAXTEXT_TO_HF_PARAM_MAPPING
from lilax._backend.checkpoint_conversion.model_families.qwen3_omni import fused as fused_qwen3_omni
from lilax._backend.checkpoint_conversion.model_families.qwen3_omni.param_mapping import QWEN3_OMNI_MOE_MAXTEXT_TO_HF_PARAM_MAPPING
from lilax._backend.common.common_types import is_fp8_dtype


def mlp_only_adapter(config):
  mapping = PARAM_MAPPING.get(config.model_name)
  decoder = getattr(config.decoder_block, "value", config.decoder_block)
  if config.use_m3_model:
    return None
  if mapping is QWEN3_NEXT_MAXTEXT_TO_HF_PARAM_MAPPING and decoder == "qwen3_next":
    return fused_qwen3_next
  if mapping is QWEN3_5_MAXTEXT_TO_HF_PARAM_MAPPING and decoder == "qwen3_5":
    return fused_qwen3_5
  if ((mapping is DEEPSEEK_MAXTEXT_TO_HF_PARAM_MAPPING and decoder == "deepseek")
      or (mapping is DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_MAPPING and decoder == "deepseek4")):
    return fused_deepseek
  return None


def supports_deepseek_fp8_mlp(config):
  block_size = getattr(config, "weight_block_size", None)
  if isinstance(block_size, int):
    return block_size > 0 and block_size != 2
  return isinstance(block_size, (list, tuple)) and bool(block_size) and all(
      isinstance(block, int) and block > 0 for block in block_size
  )


def validate_deepseek_layout(config, *, fused_qkv=None, fused_mlp=None):
  if config.fused_qkv or fused_qkv:
    raise NotImplementedError("DeepSeek MLA does not support fused QKV conversion.")
  if not (config.fused_mlp or fused_mlp):
    return
  if config.use_batch_split_schedule:
    raise NotImplementedError("DeepSeek fused MLP conversion does not support batch-split scheduling.")
  if config.ici_pipeline_parallelism > 1 or config.dcn_pipeline_parallelism > 1:
    raise NotImplementedError("DeepSeek fused MLP conversion does not support pipeline parallelism.")
  if getattr(config, "qwix_qat_config", None) is not None:
    raise NotImplementedError("DeepSeek fused MLP conversion does not support Qwix QAT.")
  if (is_fp8_dtype(getattr(config.weight_dtype, "value", config.weight_dtype))
      and not supports_deepseek_fp8_mlp(config)):
    raise NotImplementedError("DeepSeek fused FP8 MLP requires matching non-scalar block scale grids.")


def validate_hybrid_qwen_layout(config, *, fused_qkv=None, fused_mlp=None):
  if (config.fused_qkv or config.fused_mlp or fused_qkv or fused_mlp) and (config.attention_bias or config.mlp_bias):
    raise NotImplementedError("Hybrid Qwen fused projection conversion does not support bias flags.")
  vision = getattr(config.vision_encoder_block, "value", config.vision_encoder_block)
  if ((config.fused_qkv or fused_qkv) and
      ((config.use_multimodal and vision != "none" and not (mlp_only_adapter(config) is fused_qwen3_5 and vision == "qwen3_5"))
       or config.use_audio)):
    raise NotImplementedError("Hybrid Qwen fused QKV would also fuse unsupported vision or audio projections.")
  if config.use_audio and (config.fused_mlp or fused_mlp):
    raise NotImplementedError("Hybrid Qwen fused MLP would also fuse unsupported audio projections.")


def validate_mlp_only_layout(config, *, fused_qkv=None, fused_mlp=None):
  mapping = PARAM_MAPPING.get(config.model_name)
  decoder = getattr(config.decoder_block, "value", config.decoder_block)
  vision = getattr(config.vision_encoder_block, "value", config.vision_encoder_block)
  if ((config.fused_qkv or fused_qkv) and is_fp8_dtype(getattr(config.weight_dtype, "value", config.weight_dtype))
      and ((config.use_multimodal and vision != "none") or config.use_audio)):
    raise NotImplementedError("Multimodal fused QKV conversion does not support FP8 weights.")
  if (decoder in ("gemma", "gemma2", "gemma3", "gemma4", "gemma4_small") and config.use_multimodal
      and (config.attention_bias or config.mlp_bias) and (config.fused_qkv or config.fused_mlp or fused_qkv or fused_mlp)):
    raise NotImplementedError("Multimodal Gemma fused projection conversion with bias flags is not supported.")
  if (decoder in ("gemma", "gemma2", "gemma3", "gemma4", "gemma4_small")
      and (config.fused_qkv or config.fused_mlp or fused_qkv or fused_mlp)):
    vision = getattr(config.vision_encoder_block, "value", config.vision_encoder_block)
    if config.use_audio or (config.use_multimodal and vision not in ("none", "gemma3", "gemma4")):
      raise NotImplementedError("Gemma projection conversion does not support the selected multimodal towers.")
  if mapping in (QWEN3_VL_MAXTEXT_TO_HF_PARAM_MAPPING, QWEN3_OMNI_MOE_MAXTEXT_TO_HF_PARAM_MAPPING):
    vision = getattr(config.vision_encoder_block, "value", config.vision_encoder_block)
    expected_vision = "qwen3_vl" if mapping is QWEN3_VL_MAXTEXT_TO_HF_PARAM_MAPPING else "qwen3_omni"
    if (config.fused_qkv or config.fused_mlp or fused_qkv or fused_mlp) and (
        (config.use_multimodal and vision not in ("none", expected_vision))
        or (config.use_audio and mapping is QWEN3_VL_MAXTEXT_TO_HF_PARAM_MAPPING)
    ):
      raise NotImplementedError("This Qwen projection adapter does not support the selected multimodal towers.")
    if (decoder == "qwen3_moe" and (config.fused_mlp or fused_mlp)
        and not (mapping is QWEN3_OMNI_MOE_MAXTEXT_TO_HF_PARAM_MAPPING and config.use_audio)):
      raise NotImplementedError("Fused Qwen routed-expert MLP conversion is not supported.")
  if decoder == "qwen3_custom_moe" and (config.fused_mlp or fused_mlp):
    raise NotImplementedError("Qwen3 custom routed-expert MLP conversion is not supported.")
  if decoder == "llama4" and (config.num_experts > 1 or config.shared_experts) and (config.fused_mlp or fused_mlp):
    raise NotImplementedError("Llama 4 shared-expert fused MLP conversion is not supported.")
  new_native = decoder in ("gpt3", "llama2", "mistral", "llama4", "envy", "qwen3_custom_moe")
  qwen_multimodal = mapping in (QWEN3_VL_MAXTEXT_TO_HF_PARAM_MAPPING, QWEN3_OMNI_MOE_MAXTEXT_TO_HF_PARAM_MAPPING)
  if (new_native or qwen_multimodal) and (config.fused_qkv or config.fused_mlp or fused_qkv or fused_mlp):
    if config.ici_pipeline_parallelism > 1 or config.dcn_pipeline_parallelism > 1:
      raise NotImplementedError("This projection adapter does not support pipeline parallelism.")
    if new_native and config.use_multimodal and decoder != "llama4":
      raise NotImplementedError("This native projection adapter does not support multimodal towers.")
    if decoder == "llama4" and (config.use_audio or (config.use_multimodal and
        getattr(config.vision_encoder_block, "value", config.vision_encoder_block) not in ("none", "llama4"))):
      raise NotImplementedError("Llama 4 projection conversion does not support the selected multimodal towers.")
  if (PARAM_MAPPING.get(config.model_name) is QWEN_MAXTEXT_TO_HF_PARAM_MAPPING
      and getattr(config.decoder_block, "value", config.decoder_block) in ("qwen2", "qwen3", "qwen3_moe")
      and config.num_experts > 1 and (config.fused_mlp or fused_mlp)):
    raise NotImplementedError("Fused Qwen routed-expert MLP conversion is not supported.")
  adapter = mlp_only_adapter(config)
  if adapter is None:
    return
  if adapter is fused_deepseek:
    validate_deepseek_layout(config, fused_qkv=fused_qkv, fused_mlp=fused_mlp)
  if adapter in (fused_qwen3_next, fused_qwen3_5):
    validate_hybrid_qwen_layout(config, fused_qkv=fused_qkv, fused_mlp=fused_mlp)
  if fused_mlp and supports_native_hybrid_fp8_qkv(config):
    raise NotImplementedError("Fused hybrid Qwen MLP conversion does not support FP8.")
  if (config.fused_qkv or fused_qkv or config.fused_mlp or fused_mlp) and projection_layout_adapter(config) is None:
    raise NotImplementedError("This projection layout is unsupported for the selected backend or quantization, including FP8.")


def supports_hybrid_fp8_qkv(config):
  return (mlp_only_adapter(config) is fused_qwen3_5
          and str(getattr(config.weight_dtype, "value", config.weight_dtype)) == "float8_e4m3fn"
          and bool(getattr(config, "weight_block_size", None)) and not config.fused_mlp)


def supports_native_hybrid_fp8_qkv(config):
  return (mlp_only_adapter(config) in (fused_qwen3_5, fused_qwen3_next)
          and str(getattr(config.weight_dtype, "value", config.weight_dtype)) == "float8_e4m3fn"
          and bool(getattr(config, "weight_block_size", None)) and not config.fused_mlp)


def validate_huggingface_projection_layout(config, *, hf_config=None):
  mapping = PARAM_MAPPING.get(config.model_name)
  decoder = getattr(config.decoder_block, "value", config.decoder_block)
  vision = getattr(config.vision_encoder_block, "value", config.vision_encoder_block)
  quantization = hf_config.to_dict().get("quantization_config") if hf_config is not None else None
  if (config.fused_qkv and ((config.use_multimodal and vision != "none") or config.use_audio)
      and isinstance(quantization, dict) and quantization.get("quant_method") == "fp8"):
    raise NotImplementedError("Multimodal fused QKV conversion does not support FP8 weights.")
  if (mlp_only_adapter(config) in (fused_qwen3_next, fused_qwen3_5)
      and (config.ici_pipeline_parallelism > 1 or config.dcn_pipeline_parallelism > 1)):
    raise NotImplementedError("Hugging Face conversion of pipeline hybrid Qwen is not supported.")
  if (decoder in ("gpt3", "llama2", "mistral", "llama4", "envy", "qwen3_custom_moe")
      and mapping is not LLAMA31_MAXTEXT_TO_HF_PARAM_MAPPING):
    raise NotImplementedError(f"Hugging Face projection conversion of {decoder} is not supported.")
  if decoder in ("gemma", "gemma2", "gemma3", "gemma4", "gemma4_small"):
    if decoder == "gemma":
      raise NotImplementedError("Hugging Face conversion of Gemma v1 is not supported.")
    if ((config.fused_qkv or config.fused_mlp)
        and (config.ici_pipeline_parallelism > 1 or config.dcn_pipeline_parallelism > 1)):
      raise NotImplementedError("Hugging Face conversion of pipeline Gemma is not supported.")
    vision = getattr(config.vision_encoder_block, "value", config.vision_encoder_block)
    expected_vision = "gemma3" if decoder == "gemma3" else "gemma4"
    if (config.use_multimodal and (config.fused_qkv or config.fused_mlp)
        and (decoder not in ("gemma3", "gemma4", "gemma4_small")
             or vision not in ("none", expected_vision) or config.use_audio)):
      raise NotImplementedError("Hugging Face fused projection conversion of multimodal Gemma is not supported.")
  if mapping is OLMO3_MAXTEXT_TO_HF_PARAM_MAPPING and config.attention_bias:
    raise NotImplementedError("Hugging Face conversion of OLMo 3 attention biases is not supported.")
  if (mlp_only_adapter(config) is fused_qwen3_next
      and str(getattr(config.weight_dtype, "value", config.weight_dtype)) == "float8_e4m3fn"):
    raise NotImplementedError("Hugging Face conversion of Qwen3-Next FP8 weights is not supported.")
  if mlp_only_adapter(config) is fused_deepseek:
    if (is_fp8_dtype(getattr(config.weight_dtype, "value", config.weight_dtype))
        or (isinstance(quantization, dict) and quantization.get("quant_method") == "fp8")):
      raise NotImplementedError("Hugging Face conversion of DeepSeek FP8 weights is not supported.")


def projection_layout_adapter(config):
  mapping = PARAM_MAPPING.get(config.model_name)
  vision = getattr(config.vision_encoder_block, "value", config.vision_encoder_block)
  if (config.fused_qkv and is_fp8_dtype(getattr(config.weight_dtype, "value", config.weight_dtype))
      and ((config.use_multimodal and vision != "none") or config.use_audio)):
    raise NotImplementedError("Multimodal fused QKV conversion does not support FP8 weights.")
  if mapping is LLAMA31_MAXTEXT_TO_HF_PARAM_MAPPING:
    return fused_llama31
  decoder = getattr(config.decoder_block, "value", config.decoder_block)
  if not config.use_m3_model:
    if mapping in (QWEN3_VL_MAXTEXT_TO_HF_PARAM_MAPPING, QWEN3_OMNI_MOE_MAXTEXT_TO_HF_PARAM_MAPPING):
      validate_mlp_only_layout(config)
      return fused_qwen3_vl if mapping is QWEN3_VL_MAXTEXT_TO_HF_PARAM_MAPPING else fused_qwen3_omni
    if decoder == "gpt3":
      return fused_gpt3
    if decoder == "llama4":
      validate_mlp_only_layout(config)
      return fused_llama4
    if decoder in ("llama2", "mistral", "envy", "qwen3_custom_moe"):
      validate_mlp_only_layout(config)
      return fused_llama31
  adapter = mlp_only_adapter(config)
  if adapter is not None:
    weight_dtype = getattr(config.weight_dtype, "value", config.weight_dtype)
    quantization = getattr(config.quantization, "value", config.quantization)
    if adapter is fused_deepseek:
      validate_deepseek_layout(config)
    if adapter in (fused_qwen3_next, fused_qwen3_5):
      validate_hybrid_qwen_layout(config)
    deepseek_fp8 = (adapter is fused_deepseek and is_fp8_dtype(weight_dtype)
                    and supports_deepseek_fp8_mlp(config))
    if ((not config.use_multimodal or adapter is fused_qwen3_5)
        and (adapter is not fused_deepseek or (config.ici_pipeline_parallelism <= 1 and config.dcn_pipeline_parallelism <= 1))
        and not config.use_batch_split_schedule and (adapter is fused_deepseek or not config.mlp_bias) and not quantization
        and not config.checkpoint_is_quantized and not config.use_qwix_quantization
        and (str(weight_dtype) in ("float32", "float16", "bfloat16") or supports_native_hybrid_fp8_qkv(config)
             or deepseek_fp8)
        and getattr(config, "qwix_qat_config", None) is None):
      return adapter
    return None
  if (mapping is QWEN_MAXTEXT_TO_HF_PARAM_MAPPING and decoder in ("qwen2", "qwen3", "qwen3_moe")
      and not config.use_m3_model and not config.use_multimodal):
    if config.num_experts > 1 and config.fused_mlp:
      raise NotImplementedError("Fused Qwen routed-expert MLP conversion is not supported.")
    return fused_qwen
  gemma_mappings = {
      "gemma2": GEMMA2_MAXTEXT_TO_HF_PARAM_MAPPING,
      "gemma3": GEMMA3_MAXTEXT_TO_HF_PARAM_MAPPING,
      "gemma4": GEMMA4_MAXTEXT_TO_HF_PARAM_MAPPING,
      "gemma4_small": GEMMA4_SMALL_MAXTEXT_TO_HF_PARAM_MAPPING,
  }
  if (not config.use_m3_model
      and ((decoder == "gemma" and config.model_name in ("gemma-2b", "gemma-7b"))
           or (mapping is not None and mapping is gemma_mappings.get(decoder)))):
    validate_mlp_only_layout(config)
    if config.use_multimodal and (config.attention_bias or config.mlp_bias):
      if config.fused_qkv or config.fused_mlp:
        raise NotImplementedError("Multimodal Gemma fused projection conversion with bias flags is not supported.")
      return None
    if config.fused_qkv and (config.share_kv_projections or config.num_kv_shared_layers):
      raise ValueError("Shared KV projections or layers are incompatible with fused QKV.")
    if (decoder == "gemma4_small" and config.scan_layers and config.use_multimodal
        and config.ici_pipeline_parallelism <= 1 and config.dcn_pipeline_parallelism <= 1):
      raise NotImplementedError("Scanned multimodal Gemma 4 small projection conversion is not supported.")
    return fused_gemma
  if not config.use_m3_model and not config.use_multimodal:
    if mapping is OLMO3_MAXTEXT_TO_HF_PARAM_MAPPING and decoder == "olmo3":
      return fused_olmo3
    if mapping is MIXTRAL_MAXTEXT_TO_HF_PARAM_MAPPING and decoder == "mixtral":
      return fused_mixtral
    if mapping is GPT_OSS_MAXTEXT_TO_HF_PARAM_MAPPING and decoder == "gpt_oss":
      return fused_gpt_oss
  return None


def supports_mlp_layout(adapter):
  return adapter not in (fused_mixtral, fused_gpt_oss)


def attention_container(adapter):
  if adapter is fused_olmo3:
    return "attention"
  if adapter is fused_gpt_oss:
    return "GptOssAttention"
  return "self_attention"


def projection_containers(adapter, *, mlp):
  if mlp and adapter in (fused_qwen3_next, fused_qwen3_5):
    return ("mlp", "shared_expert")
  if not mlp and adapter in (fused_qwen3_next, fused_qwen3_5):
    return ("attention",)
  if mlp and adapter is fused_deepseek:
    return ("mlp", "shared_experts")
  if adapter is fused_gemma:
    return ("mlp", "mlp_local", "mlp_global", "shared_experts") if mlp else (
        "self_attention", "self_attention_local", "self_attention_global"
    )
  return ("mlp",) if mlp else (attention_container(adapter),)


def is_hybrid_qwen_adapter(adapter):
  return adapter in (fused_qwen3_next, fused_qwen3_5)
