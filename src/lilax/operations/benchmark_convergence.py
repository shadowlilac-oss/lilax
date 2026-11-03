"""Python configuration for the original convergence benchmarks."""

from lilax._backend.benchmarks.convergence.c4_exp import (
    DEEPSEEK_V3_LIBTPU_INIT_ARGS,
    c4_deepseek_v3_ep_256_v5p_512_gbs_1024,
    c4_en_hp,
    c4_mlperf_hp,
    c4_mutil_hp,
    deepseek_671b_hp,
    llama3_405b_hp,
)
from lilax._backend.benchmarks.convergence.convergence_utils import (
    ConvHParams, DatasetHParams, load_checkpoint, setup_convergence_configs, setup_dataset, setup_model_convergence,
)
from lilax.configuration import MaxTextConfig


def c4_deepseek_v3_config(**overrides) -> MaxTextConfig:
  """Build the C4 DeepSeek V3 convergence configuration for pretraining."""
  settings = c4_deepseek_v3_ep_256_v5p_512_gbs_1024()
  settings.update(overrides)
  return MaxTextConfig(model_name="deepseek3-671b", **settings)
