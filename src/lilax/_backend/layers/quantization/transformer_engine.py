"""TransformerEngine quantization recipes, collective GEMM overlap and MoE block quantizer sets."""

import functools
from typing import Tuple

import jax.numpy as jnp

import flax.linen as nn

from lilax._backend.configs.types import TeCommGemmOverlapPolicy
from lilax._backend.layers.quantization.base import Quantization


class TransformerEngineQuantization(Quantization):
  """Class for TransformerEngine quantization recipes."""

  def __init__(self, config):
    """Initialize TransformerEngine quantization."""

    self.quant_mode = "train"

    if not config.quantization.startswith("te_"):
      raise ValueError(f"Invalid TransformerEngine quantization config: {config.quantization}")

    self._recipe = TransformerEngineQuantization._get_recipe(config.quantization)

    self._te_comm_gemm_overlap_policy = config.te_comm_gemm_overlap

  def __hash__(self):
    return hash((self.quant_mode, self._recipe))

  def __eq__(self, other):
    if not isinstance(other, TransformerEngineQuantization):
      return False
    return (self.quant_mode, self._recipe) == (other.quant_mode, other._recipe)

  @staticmethod
  def _get_recipe(recipe_name: str):
    """Get the TransformerEngine recipe based on the name."""
    from transformer_engine.common import recipe  # pylint: disable=import-outside-toplevel # pytype: disable=import-error

    RECIPES = {
        "te_no_quant": lambda: None,
        "te_fp8_delayedscaling": recipe.DelayedScaling,
        "te_fp8_currentscaling": recipe.Float8CurrentScaling,
        "te_mxfp8": recipe.MXFP8BlockScaling,
        "te_nvfp4": recipe.NVFP4BlockScaling,  # pytype: disable=module-attr
        "te_nvfp4_no_rht": functools.partial(recipe.NVFP4BlockScaling, disable_rht=True),  # pytype: disable=module-attr
    }
    if recipe_name not in RECIPES:
      raise ValueError(f"Invalid TransformerEngine recipe: {recipe_name}")
    return RECIPES[recipe_name]()

  @property
  def needs_apply_rngs(self) -> bool:
    """Whether this recipe draws RNGs at apply time.

    Only NVFP4 does, and only while stochastic rounding is on: TE draws ``sr_rng`` for
    the DGRAD quantizer. The other recipes take their scales from the tensors.
    """
    from transformer_engine.common import recipe  # pylint: disable=import-outside-toplevel # pytype: disable=import-error

    if not isinstance(self._recipe, recipe.NVFP4BlockScaling):  # pytype: disable=module-attr
      return False
    return not self._recipe.disable_stochastic_rounding

  def get_block_size(self):
    """Get the block size for quantization for recipes that require blocks.

    If there is no block requirement for the current recipe, returns 1.
    """
    from transformer_engine.common import recipe  # pylint: disable=import-outside-toplevel # pytype: disable=import-error

    if isinstance(self._recipe, recipe.MXFP8BlockScaling):
      return 32
    if isinstance(self._recipe, recipe.NVFP4BlockScaling):  # pytype: disable=module-attr
      return 128  # TODO(set this to 16 when unfused RHT is supported)
    return 1

  def _wrap(self, f, name=None):
    """Wraps the given function `f` to support TransformerEngine quantization.

    This method does a couple things:


    1. Wraps the given function in a context that specifies MaxText's physical mesh axes to
    TransformerEngine. This ensures our collective operations in TransformerEngine are using
    the correct axes.

    2. Wraps the given function in a Flax linen module. This module does not store any Flax
    parameters but can store Flax variables for quantizers if required by the recipe.

    3. When the wrapper is called, it provides two additional arguments to the given function `f`,
    'generate_quantizer_set' as the first argument and 'generate_collective_op_set' as the second argument.
    'generate_quantizer_set' is a function that can be called to generate a TransformerEngine/JAX quantizer
    set object used in TransformerEngine/JAX APIs based on the recipe of this TransformerEngineQuantizer
    object. Similarly, 'generate_collective_op_set' is a function that can be called to generate a
    TransformerEngine/JAX collective operation set object used in TransformerEngine/JAX APIs based
    the kernel's mesh axes.

    Args:
      f: The function to wrap. The first argument must be 'generate_quantizer_set'.
      name: The name of this wrapped operation. If unspecified, will use `f.__name__`.

    Returns:
      A Flax linen module that wraps the given function.
    """

    import transformer_engine.jax  # pylint: disable=import-outside-toplevel # pytype: disable=import-error
    import transformer_engine.jax.cpp_extensions as tex  # pylint: disable=import-outside-toplevel # pytype: disable=import-error
    from transformer_engine.common import recipe  # pylint: disable=import-outside-toplevel # pytype: disable=import-error

    default_recipe = self._recipe
    overlap_policy = self._te_comm_gemm_overlap_policy

    class TEWrapper(transformer_engine.jax.flax.module.TransformerEngineBase):
      """Wrapper module for TransformerEngine quantization."""

      def generate_quantizer_set(
          self,
          postfix: str = "",
          variable_collection: str | None = None,
          quantization_checkpoint_name: str | None = None,
          fp8_recipe: recipe.Recipe | None = default_recipe,
          n_groups: int | None = None,
      ):
        """Route quantizer-set generation through the OVERWRITE_WITH_GRADIENT collection."""
        OVERWRITE_WITH_GRADIENT = "_overwrite_with_gradient"
        return super().generate_quantizer_set(  # pytype: disable=wrong-keyword-args
            postfix=postfix,
            variable_collection=OVERWRITE_WITH_GRADIENT,
            quantization_checkpoint_name="quantization",
            fp8_recipe=fp8_recipe,
            n_groups=n_groups,  # pyrefly: ignore[bad-argument-type]
        )

      def generate_collective_op_set(self, mesh_axes: Tuple[str, ...] = ()):
        """Inspect the kernel's mesh axes to determine the type of collective operation to use for collective GEMM."""
        if overlap_policy == TeCommGemmOverlapPolicy.DISABLED:
          return tex.noop_collective_op_set

        if len(mesh_axes) >= 1:
          # CGEMM in MLP layer (up projection, down projection)
          if mesh_axes[0] == "embed" and mesh_axes[-1] == "mlp":
            return tex.CollectiveOpSet.create(tex.CollectiveOp.ALL_GATHER)
          elif mesh_axes[0] == "mlp" and mesh_axes[-1] in ("embed", "embed_attn"):
            # 'embed_attn' covers the flattened attention output projection of Qwen3 hybrid models.
            return tex.CollectiveOpSet.create(tex.CollectiveOp.REDUCE_SCATTER)
          elif overlap_policy == TeCommGemmOverlapPolicy.FULL:
            # CGEMM also in Attention layer (QKV projection, output projection)
            if mesh_axes[0] == "embed_attn" and mesh_axes[-1].startswith("kv"):
              return tex.CollectiveOpSet.create(tex.CollectiveOp.ALL_GATHER)
            elif mesh_axes[0] == "heads" and mesh_axes[-1] == "embed_attn":
              return tex.CollectiveOpSet.create(tex.CollectiveOp.REDUCE_SCATTER)

        return tex.noop_collective_op_set

      @nn.compact
      def __call__(self, *args, **kwargs):
        return f(self.generate_quantizer_set, self.generate_collective_op_set, *args, **kwargs)

    TEWrapper.__name__ = f"TEWrapper_{name if name else f.__name__}"

    return TEWrapper

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    """Placeholder for dot_general implementation in subclasses."""
    import transformer_engine.jax  # pylint: disable=import-outside-toplevel # pytype: disable=import-error
    import transformer_engine.jax.cpp_extensions as tex  # pylint: disable=import-outside-toplevel # pytype: disable=import-error

    def te_dot_general(generate_quantizer_set, generate_collective_op_set, x, kernel, dims, **kwargs):
      contracting_dims, batch_dims = dims
      assert batch_dims == ((), ()), "Batch dimensions must be empty for TransformerEngine dot."

      quantizer_set = generate_quantizer_set()
      collective_op_set = (
          generate_collective_op_set(mesh_axes)
          if self._te_comm_gemm_overlap_policy != TeCommGemmOverlapPolicy.DISABLED
          else tex.noop_collective_op_set
      )
      return transformer_engine.jax.dense.dense(
          x,
          kernel,
          contracting_dims=contracting_dims,
          quantizer_set=quantizer_set,
          collective_op_set=collective_op_set,
      )

    return self._wrap(te_dot_general, "dot_general")

  def einsum(self, *args, **kwargs):
    """Placeholder for einsum implementation in subclasses."""
    # quant.einsum is only required for MoE or for inference with KVCache.
    raise ValueError("Einsum is not yet supported for TransformerEngine quantization.")

  def get_moe_block_quantizer_set(
      self,
      te_gmm_quantization_recipe_name: str,
      n_token_groups: int,
      n_expert_groups: int,
  ):
    """Create a TransformerEngine quantizer set for TE MoEBlock grouped GEMMs."""
    from transformer_engine.jax.quantize import (  # pylint: disable=import-outside-toplevel
        QuantizerFactory,
        QuantizerSet,
        ScalingMode,
    )

    if te_gmm_quantization_recipe_name == "te_no_quant":
      return QuantizerFactory.create_set(scaling_mode=ScalingMode.NO_SCALING, is_2x2x=False)
    if te_gmm_quantization_recipe_name != "te_mxfp8":
      raise ValueError(f"Invalid TransformerEngine GMM quantization recipe name: {te_gmm_quantization_recipe_name}")
    if n_token_groups <= 0 or n_expert_groups <= 0:
      raise ValueError(
          "TE MoEBlock quantizer group counts must be positive, got "
          f"n_token_groups={n_token_groups}, n_expert_groups={n_expert_groups}."
      )

    token_quantizer_set = QuantizerFactory.create_set(
        scaling_mode=ScalingMode.MXFP8_1D_SCALING,
        fwd_dtype=jnp.float8_e4m3fn,
        bwd_dtype=jnp.float8_e4m3fn,
        is_2x2x=True,
        n_groups=n_token_groups,
    )
    expert_quantizer_set = QuantizerFactory.create_set(
        scaling_mode=ScalingMode.MXFP8_1D_SCALING,
        fwd_dtype=jnp.float8_e4m3fn,
        bwd_dtype=jnp.float8_e4m3fn,
        is_2x2x=True,
        n_groups=n_expert_groups,
    )
    return QuantizerSet(
        x=token_quantizer_set.x,
        kernel=expert_quantizer_set.kernel,
        dgrad=token_quantizer_set.dgrad,
    )

  def get_moe_block_quantizer_sets(
      self,
      te_gmm_quantization_recipe_name: str,
      n_token_groups: int,
      n_expert_groups: int,
  ):
    """Create independent FC1 and FC2 quantizer sets for TE's MoE custom VJP."""
    return tuple(
        self.get_moe_block_quantizer_set(
            te_gmm_quantization_recipe_name,
            n_token_groups=n_token_groups,
            n_expert_groups=n_expert_groups,
        )
        for _ in range(2)
    )
