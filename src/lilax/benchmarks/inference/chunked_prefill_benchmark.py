"""Python entry point for the chunked-prefill microbenchmark."""

from lilax._backend.inference.mlperf.microbenchmarks.benchmark_chunked_prefill import run_chunked_prefill_benchmark

__all__ = ["run_chunked_prefill_benchmark"]
