"""Resolves the Tokamax Splash Attention kernel and mask modules across Tokamax releases."""

try:  # pragma: no cover
  from tokamax._src.ops.splash_attention import splash_attention_kernel as tokamax_splash_kernel
  from tokamax._src.ops.splash_attention import splash_attention_mask as tokamax_splash_mask
except ImportError:  # pragma: no cover
  from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as tokamax_splash_kernel
  from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as tokamax_splash_mask
