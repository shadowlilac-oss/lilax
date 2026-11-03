"""Registry mapping benchmark names to dataset classes."""

from lilax._backend.eval.datasets.base import BenchmarkDataset
from lilax._backend.eval.datasets.mlperf import MlperfOpenOrcaDataset

DATASET_REGISTRY: dict[str, type[BenchmarkDataset]] = {
    "mlperf_openorca": MlperfOpenOrcaDataset,
    "openorca": MlperfOpenOrcaDataset,
}


def get_dataset(benchmark_name: str) -> BenchmarkDataset:
  key = benchmark_name.lower()
  if key not in DATASET_REGISTRY:
    raise KeyError(
        f"No dataset registered for benchmark '{benchmark_name}'. "
        f"Available: {sorted(DATASET_REGISTRY)}. "
        "For MMLU, GPQA, MATH etc. use lm_eval_runner or evalchemy_runner instead."
    )
  return DATASET_REGISTRY[key]()


def register_dataset(benchmark_name: str, dataset_cls: type[BenchmarkDataset]) -> None:
  DATASET_REGISTRY[benchmark_name.lower()] = dataset_cls
