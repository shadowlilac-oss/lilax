"""Decoder family selection."""

from lilax._backend.common.common_types import DecoderBlockType
from lilax._backend.models import (
    deepseek, deepseek4, envy, gemma, gemma2, gemma3, gemma4, gemma4_small, gpt3, gpt_oss, hy4,
    llama2, llama4, ming, mistral, mixtral, olmo3, qwen2, qwen3, qwen3_5, qwen3_custom, simple_layer,
)
from lilax._backend.models.weaver import decoder_layer as weaver_decoder_layer


def get_decoder_layers(cfg, default_layer):
  """Retrieves decoder layer classes based on config using a dictionary lookup."""

  def get_scannable(normal_cls, scannable_cls):
    return [scannable_cls] if cfg.scan_layers else [normal_cls]

  def get_deepseek():
    return [deepseek.DeepSeekDenseLayer, deepseek.DeepSeekMoELayer]

  layer_map = {
      DecoderBlockType.DEFAULT: [default_layer],
      DecoderBlockType.LLAMA2: [llama2.LlamaDecoderLayer],
      DecoderBlockType.MISTRAL: [mistral.MistralDecoderLayer],
      DecoderBlockType.MIXTRAL: [mixtral.MixtralDecoderLayer],
      DecoderBlockType.GEMMA: [gemma.GemmaDecoderLayer],
      DecoderBlockType.GEMMA2: [gemma2.Gemma2DecoderLayer],
      DecoderBlockType.GEMMA3: [gemma3.Gemma3DecoderLayer],
      DecoderBlockType.GEMMA4: get_scannable(gemma4.Gemma4DecoderLayer, gemma4.Gemma4ScannableBlock),
      DecoderBlockType.GEMMA4_SMALL: [gemma4_small.Gemma4SmallDecoderLayer],
      DecoderBlockType.GPT3: [gpt3.Gpt3DecoderLayer],
      DecoderBlockType.QWEN2: [qwen2.Qwen2DecoderLayer],
      DecoderBlockType.QWEN3: [qwen3.Qwen3DecoderLayer],
      DecoderBlockType.QWEN3_MOE: [qwen3.Qwen3MoeDecoderLayer],
      DecoderBlockType.QWEN3_CUSTOM_MOE: [qwen3_custom.Qwen3CustomMoeDecoderLayer],
      DecoderBlockType.SIMPLE: [simple_layer.SimpleDecoderLayer],
      DecoderBlockType.SIMPLE_MLP: [simple_layer.SimpleMlpDecoderLayer],
      DecoderBlockType.DEEPSEEK: get_deepseek(),
      DecoderBlockType.DEEPSEEK4: get_scannable(deepseek4.DeepSeek4DecoderLayer, deepseek4.DeepSeek4ScannableBlock),
      DecoderBlockType.GPT_OSS: get_scannable(gpt_oss.GptOssDecoderLayer, gpt_oss.GptOssScannableBlock),
      DecoderBlockType.QWEN3_NEXT: get_scannable(qwen3.Qwen3NextDecoderLayer, qwen3.Qwen3NextScannableBlock),
      DecoderBlockType.QWEN3_5: get_scannable(qwen3_5.Qwen3_5DecoderLayer, qwen3_5.Qwen3_5ScannableBlock),
      DecoderBlockType.LLAMA4: get_scannable(llama4.Llama4DecoderLayer, llama4.Llama4ScannableBlock),
      DecoderBlockType.OLMO3: get_scannable(olmo3.Olmo3DecoderLayer, olmo3.Olmo3ScannableBlock),
      DecoderBlockType.ENVY: get_scannable(envy.EnvyDecoderLayer, envy.EnvyScannableBlock),
      DecoderBlockType.WEAVER: [weaver_decoder_layer.WeaverMoTDecoderLayer],
      DecoderBlockType.MING: get_scannable(ming.MingDecoderLayer, ming.MingScannableBlock),
      DecoderBlockType.HY4: get_scannable(hy4.Hy4DecoderLayer, hy4.Hy4ScannableBlock),
  }

  if cfg.decoder_block not in layer_map:
    raise ValueError(f"Incorrect decoder_block name {cfg.decoder_block.value=}")

  return layer_map[cfg.decoder_block]
