"""Feature cache state for the Wan autoencoder."""

from flax import nnx
from jax import tree_util

from .autoencoder_causal_conv_wan import WanCausalConv3d

CACHE_T = 2


def _update_cache(cache, idx, value):
  if cache is None:
    return None
  return cache[:idx] + (value,) + cache[idx + 1 :]


class RepSentinel:
  def __eq__(self, other):
    return isinstance(other, RepSentinel)


RepSentinel.__module__ = "lilax._backend.m3.models.wan.autoencoder_kl_wan"
tree_util.register_pytree_node(RepSentinel, lambda x: ((), None), lambda _, __: RepSentinel())


class AutoencoderKLWanCache:
  def __init__(self, module):
    self.module = module

    def _count_conv3d(module):
      count = 0
      node_types = nnx.graph.iter_graph([module])
      for _, value in node_types:
        if isinstance(value, WanCausalConv3d):
          count += 1
      return count

    self._conv_num = _count_conv3d(self.module.decoder)
    self._enc_conv_num = _count_conv3d(self.module.encoder)
    self.init_cache()

  def init_cache(self):
    self._feat_map = (None,) * self._conv_num
    self._enc_feat_map = (None,) * self._enc_conv_num


AutoencoderKLWanCache.__module__ = "lilax._backend.m3.models.wan.autoencoder_kl_wan"


def _wan_cache_flatten(cache):
  return (cache._feat_map, cache._enc_feat_map), (cache._conv_num, cache._enc_conv_num)


def _wan_cache_unflatten(aux, children):
  conv_num, enc_conv_num = aux
  feat_map, enc_feat_map = children
  obj = AutoencoderKLWanCache.__new__(AutoencoderKLWanCache)
  obj._conv_num = conv_num
  obj._enc_conv_num = enc_conv_num
  obj._feat_map = feat_map
  obj._enc_feat_map = enc_feat_map
  obj.module = None
  return obj


tree_util.register_pytree_node(AutoencoderKLWanCache, _wan_cache_flatten, _wan_cache_unflatten)
