"""Ahead-of-time inference executable cache."""

from lilax._backend.inference.aot_cache import cached_jit
from lilax._backend.inference.aot_cache import install
from lilax._backend.inference.aot_cache import wait_for_loads
from lilax._backend.inference.aot_cache import save_pending
from lilax._backend.inference.aot_cache import warmup_mode
from lilax._backend.inference.aot_cache import in_warmup
from lilax._backend.inference.aot_cache import real_execution
