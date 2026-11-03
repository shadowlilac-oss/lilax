"""Python entry points for native Omni supervised fine-tuning and evaluation."""

from lilax._backend.experimental.omni_pipeline.eval_sft_omni import evaluate_sft_omni
from lilax._backend.experimental.omni_pipeline.train_sft_omni import train_sft_omni

__all__ = ["evaluate_sft_omni", "train_sft_omni"]
