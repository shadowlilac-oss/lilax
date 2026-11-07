"""FLOPs accounting for Engram memory layers."""


def calculate_engram_tflops(config):
  B = config.per_device_batch_size
  S = config.max_target_length
  G = config.mhc_expansion_rate
  D = config.emb_dim
  k = config.engram_kernel_size
  num_ngram_orders = config.engram_max_ngram_size - 1
  engram_dim = config.engram_num_heads * config.engram_head_dim * num_ngram_orders
  key_flops = 2 * (B * S) * engram_dim * (G * D)
  value_flops = 2 * (B * S) * engram_dim * D
  attention_flops = 2 * (B * S) * G * D
  total_channels = G * D
  conv_flops = 2 * (B * S) * k * total_channels
  num_layers = len(config.engram_layers)
  learnable_tflops = num_layers * (key_flops + value_flops + conv_flops) * 3 / 1e12
  attention_tflops = num_layers * attention_flops * 3 / 1e12
  return learnable_tflops, attention_tflops
