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

"""Self-contained Weaver Mixture-of-Transformers architecture.

The Weaver architecture is implemented without routing through a shared
``Attention`` or ``Decoder`` class, split by domain:

* ``yaml_defaults``: isolated ``weaver.yml`` default loading.
* ``dtypes``: string/dtype resolution.
* ``config``: ``WeaverConfig`` dataclass and MaxText config adaptation.
* ``leaf_layers``: pure ``flax.nnx`` leaf layer builders with logical axes.
* ``packing``: ``WeaverPackingMetadata`` and ``build_weaver_packing_metadata``.
* ``streams``: ``unpack_streams`` and ``reinterleave_streams``.
* ``masks``: sample-fenced causal & bidirectional masks + Pallas Splash masks.
* ``rotary``: 3D M-RoPE position ids and rotary position embeddings.
* ``attention_kernels``: dual-pathway attention kernels.
* ``attention``: ``WeaverDualAttention`` module.
* ``mlp``: ``WeaverMLP`` feed-forward block (``silu`` SwiGLU and ``relu2``).
* ``decoder_layer``: ``WeaverMoTDecoderLayer`` dual-pathway decoder block.
* ``latents``: latent patchify / unpatchify.
* ``timestep``: sinusoidal timestep embedding and ``WeaverTimeEmbedder``.
* ``transformer``: ``WeaverOmniTransformer`` joint multimodal diffusion backbone.
"""
