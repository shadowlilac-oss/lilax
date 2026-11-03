"""Matrix multiplication data type benchmark."""

from dataclasses import dataclass

import jax

from lilax._backend.inference.mlperf.matmul import timing_util


@dataclass
class MatmulDtypesConfig:
  num_matmuls: int = 250
  matrix_size: int = 2048
  enable_profile: bool = False


def run_matmul_dtypes(config: MatmulDtypesConfig):
  def f(X, Y):
    return jax.lax.batch_matmul(X, Y)

  f_jit = jax.jit(f)
  for dtypeA, dtypeB in [
      (jax.numpy.int4, jax.numpy.int4),
      (jax.numpy.int4, jax.numpy.int8),
      (jax.numpy.int8, jax.numpy.int4),
      (jax.numpy.int8, jax.numpy.int8),
      (jax.numpy.int8, jax.numpy.bfloat16),
      (jax.numpy.bfloat16, jax.numpy.bfloat16),
  ]:
    A = jax.numpy.ones((config.num_matmuls, config.matrix_size, config.matrix_size), dtype=dtypeA)
    B = jax.numpy.ones((config.num_matmuls, config.matrix_size, config.matrix_size), dtype=dtypeB)
    print(f"A, B shape is {f(A, B).shape}. A dtype is {A.dtype}, B dtype is {B.dtype} and prod type is {f(A, B).dtype}")
    timing_util.simple_timeit(f_jit, A, B, task="matmul_" + str(config.matrix_size), enable_profile=config.enable_profile)
