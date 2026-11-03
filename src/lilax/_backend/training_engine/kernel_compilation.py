"""Jitting, lowering and XLA options for the engine's train kernels."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
from typing import Any

import jax
from lilax._backend.compiler.xla_options import libtpu_flags

KERNEL_NAMES = ("fwd_bwd", "fwd_bwd_accum", "update")


def to_aval(value: Any) -> Any:
  if not hasattr(value, "shape") or not hasattr(value, "dtype"):
    return value
  return jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=getattr(value, "sharding", None))


def xla_options(config: Any, compiler_options: dict[str, Any] | None) -> dict[str, Any] | None:
  if compiler_options is None and getattr(config, "compile_xla_flags", ""):
    return libtpu_flags.parse_libtpu_flags_to_dict(config.compile_xla_flags)
  return compiler_options


@dataclasses.dataclass(frozen=True)
class TrainKernelShardings:
  """`in_shardings`/`out_shardings` of the three train kernels; all None leaves placement to JAX."""

  first_in: Any = None
  accum_in: Any = None
  fwd_bwd_out: Any = None
  update_in: Any = None
  update_out: Any = None


def train_kernel_shardings(
    mesh: jax.sharding.Mesh, state_shardings: Any, params_shardings: Any, rest_shardings: Any, batch_shardings: Any,
    grad_shardings: Any,
) -> TrainKernelShardings:
  replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  first_in = (params_shardings, rest_shardings, batch_shardings)
  return TrainKernelShardings(
      first_in=first_in,
      accum_in=first_in + (grad_shardings, replicated),
      fwd_bwd_out=(None, None, rest_shardings, grad_shardings, replicated),
      update_in=(state_shardings, grad_shardings, replicated, None),
      update_out=(state_shardings, None, None),
  )


def jit_train_kernels(
    fwd_bwd: Callable[..., Any], fwd_bwd_accum: Callable[..., Any], update: Callable[..., Any],
    shardings: TrainKernelShardings,
) -> dict[str, Any]:
  return {
      "fwd_bwd": jax.jit(fwd_bwd, in_shardings=shardings.first_in, out_shardings=shardings.fwd_bwd_out),
      "fwd_bwd_accum": jax.jit(
          fwd_bwd_accum, in_shardings=shardings.accum_in, out_shardings=shardings.fwd_bwd_out, donate_argnums=(3, 4)
      ),
      "update": jax.jit(
          update, in_shardings=shardings.update_in, out_shardings=shardings.update_out, donate_argnums=(0,)
      ),
  }


def lower_train_kernels(
    jitted: dict[str, Any], state_aval: Any, params_aval: Any, rest_aval: Any, batch_aval: Any, mean_loss_aval: Any
) -> dict[str, jax.stages.Lowered]:
  fwd_bwd = jitted["fwd_bwd"].lower(params_aval, rest_aval, batch_aval)
  _, _, _, grads_aval, denominator_aval = fwd_bwd.out_info
  return {
      "fwd_bwd": fwd_bwd,
      "fwd_bwd_accum": jitted["fwd_bwd_accum"].lower(params_aval, rest_aval, batch_aval, grads_aval, denominator_aval),
      "update": jitted["update"].lower(state_aval, grads_aval, denominator_aval, mean_loss_aval),
  }
