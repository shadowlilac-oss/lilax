"""Console logging defaults, applied before importing Lilax dependencies."""

import logging
import os
import warnings

from absl import logging as absl_logging


for variable in ("TRANSFORMERS_VERBOSITY", "DATASETS_VERBOSITY", "HF_HUB_VERBOSITY"):
    os.environ.setdefault(variable, "warning")

for module in (r"torch\.cuda", r"google\.api_core\._python_package_support", r"google\.auth\.transport\.grpc"):
    warnings.filterwarnings("ignore", category=FutureWarning, module=f"^{module}$", append=True)

if absl_logging.FLAGS["verbosity"].using_default_value:
    if logging.getLogger().handlers:
        absl_logging.get_absl_logger().setLevel(logging.WARNING)
    else:
        absl_logging.set_verbosity(absl_logging.WARNING)

logger = logging.getLogger("lilax")
if (not logger.handlers and logger.level == logging.NOTSET and logger.propagate
        and logging.getLogger().level == logging.WARNING
        and (not logger.hasHandlers() or (
            absl_logging.FLAGS["verbosity"].using_default_value
            and logging.getLogger().handlers == [absl_logging.get_absl_handler()]
        ))):
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
