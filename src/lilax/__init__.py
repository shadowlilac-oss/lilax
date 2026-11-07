"""Inert configuration and supported model capabilities for Lilax."""

from .configuration import AutoConfig, MaxTextConfig
from .modeling.hub import (
    HF_EXPORT_MODELS, HF_IMPORT_MODELS, HF_REPOS, MODEL_CONFIG_NAMES, NATIVE_MODELS,
    ModelCapabilities, get_model_capabilities,
)
from .version import __version__
from .processing import AutoProcessor, AutoTokenizer
from .modeling.models import AutoModelForCausalLM, MaxTextModel
from .generation.configuration import GenerationConfig
from .trainers import (
    Trainer, TrainingArguments, SFTConfig, SFTTrainer, DPOConfig, DPOTrainer, DistillationConfig, DistillationTrainer,
    GRPOConfig, GRPOTrainer,
)

__all__ = [
    "AutoConfig", "MaxTextConfig", "ModelCapabilities", "get_model_capabilities",
    "HF_EXPORT_MODELS", "HF_IMPORT_MODELS", "HF_REPOS", "MODEL_CONFIG_NAMES", "NATIVE_MODELS", "__version__",
    "AutoProcessor", "AutoTokenizer", "AutoModelForCausalLM", "MaxTextModel", "GenerationConfig", "Trainer",
    "TrainingArguments", "SFTConfig", "SFTTrainer", "DPOConfig", "DPOTrainer", "DistillationConfig", "DistillationTrainer",
    "GRPOConfig", "GRPOTrainer",
]
