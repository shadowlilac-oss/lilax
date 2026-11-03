# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Typed ``WeaverConfig`` dataclass and its MaxText config adapter."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax.numpy as jnp
from lilax._backend.models.weaver.dtypes import _resolve_jax_dtype
from lilax._backend.models.weaver.yaml_defaults import _load_weaver_yaml_defaults


@dataclasses.dataclass
class WeaverConfig:
  """Configuration for Weaver Mixture-of-Transformers blocks and backbone."""

  hidden_size: int = 4096
  head_dim: int = 128
  num_attention_heads: int = 32
  num_key_value_heads: int = 8
  intermediate_size: int = 12288
  num_hidden_layers: int = 36
  vocab_size: int = 151936
  latent_channels: int = 48
  patch_size: int = 2
  time_embed_in_channels: int = 256
  timestep_scale: float = 0.001
  timestep_max_period: int = 10000
  rms_norm_eps: float = 1e-6
  hidden_act: str = "silu"
  attention_bias: bool = False
  qk_norm_for_text: bool = True
  qk_norm_for_diffusion: bool = True
  use_und_k_norm_for_gen: bool = False
  attention_kernel: str = "dot_product"
  use_ragged_ops: bool = False
  block_q: int = 128
  block_kv: int = 128
  mrope_section: tuple[int, int, int] = (24, 20, 20)
  rope_theta: float = 5_000_000.0
  scan_layers: bool = False
  param_scan_axis: int = 0
  remat_policy: str = "none"
  embed_kernel_axes: tuple[None | str, ...] = ("vocab", "embed")
  proj_in_kernel_axes: tuple[None | str, ...] = ("patch", "embed")
  time_mlp_in_kernel_axes: tuple[None | str, ...] = ("time_embed", "embed")
  time_mlp_out_kernel_axes: tuple[None | str, ...] = ("embed", "embed")
  proj_out_kernel_axes: tuple[None | str, ...] = ("embed", "patch")
  q_kernel_axes: tuple[None | str, ...] = ("embed", "heads")
  kv_kernel_axes: tuple[None | str, ...] = ("embed", "kv_heads")
  o_kernel_axes: tuple[None | str, ...] = ("heads", "embed")
  qk_norm_kernel_axes: tuple[None | str, ...] = ()
  mlp_up_kernel_axes: tuple[None | str, ...] = ("embed", "mlp")
  mlp_down_kernel_axes: tuple[None | str, ...] = ("mlp", "embed")
  norm_kernel_axes: tuple[None | str, ...] = ("norm",)
  dtype: Any = jnp.float32
  weight_dtype: Any = jnp.float32

  def __post_init__(self) -> None:
    if self.num_attention_heads % self.num_key_value_heads != 0:
      raise ValueError(
          "num_attention_heads must be divisible by num_key_value_heads, "
          f"got {self.num_attention_heads=} {self.num_key_value_heads=}"
      )
    self.dtype = _resolve_jax_dtype(self.dtype)
    self.weight_dtype = _resolve_jax_dtype(self.weight_dtype)

  @classmethod
  def from_maxtext_config(cls, config: Any) -> WeaverConfig:
    """Translates a MaxText ``pyconfig`` object into a ``WeaverConfig`` dataclass.

    Clarification on ``WeaverConfig.from_maxtext_config`` vs ``Module.from_config``:
    * ``WeaverConfig.from_maxtext_config`` (this method) is a **dataclass config
      adapter**: it converts a MaxText ``pyconfig.HyperParameters`` / ``MaxTextConfig``
      object (or a partial config stub in unit tests) into a self-contained
      ``WeaverConfig`` dataclass instance.
    * ``WeaverMoTDecoderLayer.from_config`` and ``WeaverOmniTransformer.from_config``
      are **Flax NNX module factory classmethods**: they instantiate and return the
      neural network module itself from a ``WeaverConfig`` or MaxText ``pyconfig``
      object (delegating config normalization to ``from_maxtext_config``).

    Why ``getattr(config, ..., default)`` is used here:
    1. Bridges MaxText's attribute names (e.g. ``emb_dim`` / ``base_emb_dim``,
       ``num_query_heads``, ``mlp_dim``, ``num_decoder_layers``,
       ``normalization_layer_epsilon``, ``rope_max_timescale``) to the
       self-contained ``WeaverConfig`` field names.
    2. Reads Weaver-specific diffusion and pathway defaults (``latent_channels``,
       ``patch_size``, ``time_embed_in_channels``, ``timestep_scale``,
       ``timestep_max_period``, ``qk_norm_for_text``, ``qk_norm_for_diffusion``,
       ``use_und_k_norm_for_gen``, ``weaver_block_q``, ``weaver_block_kv``) from
       the isolated ``src/maxtext/configs/models/weaver.yml`` file when not
       present on ``config``, avoiding the need to register Weaver-only fields in
       core ``base.yml`` or ``types.py``.
    3. Allows callers and unit tests to pass partial/mock config objects without
       requiring every MaxText field to be populated.
    """
    if isinstance(config, cls):
      return config

    yml = _load_weaver_yaml_defaults()

    mlp_acts = getattr(config, "mlp_activations", yml.get("mlp_activations", ["silu", "linear"]))
    hidden_act = "relu2" if "relu2" in mlp_acts else "silu"

    raw_attn = str(getattr(config, "attention", "dot_product"))
    attn_kernel = raw_attn if raw_attn in ("dot_product", "flash", "splash") else "dot_product"

    mrope_raw = getattr(config, "mrope_section", yml.get("mrope_section", (24, 20, 20)))
    mrope_tuple = (int(mrope_raw[0]), int(mrope_raw[1]), int(mrope_raw[2]))

    # Core architectural fields: check derived MaxText names (e.g. emb_dim) first,
    # then base MaxText names (e.g. base_emb_dim), then isolated weaver.yml defaults.
    return cls(
        hidden_size=int(getattr(config, "emb_dim", getattr(config, "base_emb_dim", yml.get("base_emb_dim", 4096)))),
        head_dim=int(getattr(config, "head_dim", yml.get("head_dim", 128))),
        num_attention_heads=int(
            getattr(
                config,
                "num_query_heads",
                getattr(config, "base_num_query_heads", yml.get("base_num_query_heads", 32)),
            )
        ),
        num_key_value_heads=int(
            getattr(config, "num_kv_heads", getattr(config, "base_num_kv_heads", yml.get("base_num_kv_heads", 8)))
        ),
        intermediate_size=int(
            getattr(config, "mlp_dim", getattr(config, "base_mlp_dim", yml.get("base_mlp_dim", 12288)))
        ),
        num_hidden_layers=int(
            getattr(
                config,
                "num_decoder_layers",
                getattr(config, "base_num_decoder_layers", yml.get("base_num_decoder_layers", 36)),
            )
        ),
        vocab_size=int(getattr(config, "vocab_size", yml.get("vocab_size", 151936))),
        # Weaver-specific diffusion & pathway fields default to src/maxtext/configs/models/weaver.yml.
        latent_channels=int(getattr(config, "latent_channels", yml.get("latent_channels", 48))),
        patch_size=int(getattr(config, "patch_size", yml.get("patch_size", 2))),
        time_embed_in_channels=int(getattr(config, "time_embed_in_channels", yml.get("time_embed_in_channels", 256))),
        timestep_scale=float(getattr(config, "timestep_scale", yml.get("timestep_scale", 0.001))),
        timestep_max_period=int(getattr(config, "timestep_max_period", yml.get("timestep_max_period", 10000))),
        rms_norm_eps=float(getattr(config, "normalization_layer_epsilon", yml.get("normalization_layer_epsilon", 1e-6))),
        hidden_act=hidden_act,
        attention_bias=bool(getattr(config, "attention_bias", False)),
        qk_norm_for_text=bool(
            getattr(
                config,
                "qk_norm_for_text",
                getattr(config, "use_qk_norm", yml.get("qk_norm_for_text", True)),
            )
        ),
        qk_norm_for_diffusion=bool(getattr(config, "qk_norm_for_diffusion", yml.get("qk_norm_for_diffusion", True))),
        use_und_k_norm_for_gen=bool(getattr(config, "use_und_k_norm_for_gen", yml.get("use_und_k_norm_for_gen", False))),
        attention_kernel=attn_kernel,
        use_ragged_ops=bool(getattr(config, "use_ragged_attention", False)),
        block_q=int(getattr(config, "weaver_block_q", yml.get("weaver_block_q", 128))),
        block_kv=int(getattr(config, "weaver_block_kv", yml.get("weaver_block_kv", 128))),
        mrope_section=mrope_tuple,
        rope_theta=float(getattr(config, "rope_max_timescale", yml.get("rope_max_timescale", 5_000_000.0))),
        scan_layers=bool(getattr(config, "scan_layers", False)),
        param_scan_axis=int(getattr(config, "param_scan_axis", 0)),
        remat_policy=str(getattr(config, "remat_policy", "none")),
        dtype=_resolve_jax_dtype(getattr(config, "dtype", jnp.float32)),
        weight_dtype=_resolve_jax_dtype(getattr(config, "weight_dtype", jnp.float32)),
    )
