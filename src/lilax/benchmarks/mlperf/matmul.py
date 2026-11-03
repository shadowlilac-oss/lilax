"""MLPerf matrix multiplication operations."""

from lilax._backend.inference.mlperf.matmul.matmul_dtypes import MatmulDtypesConfig, run_matmul_dtypes
from lilax._backend.inference.mlperf.matmul.matmul_sharding import (
    MatmulShardingConfig, matmul, matmuls, run_matmul_sharding,
)
from lilax._backend.inference.mlperf.matmul.timing_util import simple_timeit

__all__ = [
    "MatmulDtypesConfig", "MatmulShardingConfig", "matmul", "matmuls", "run_matmul_dtypes",
    "run_matmul_sharding", "simple_timeit",
]
