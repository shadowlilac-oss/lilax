"""Per-shape ahead-of-time compiled executable cache."""

from __future__ import annotations

import contextlib
import glob
import hashlib
import inspect
import json
import os
import pickle
import re
import threading
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax.experimental import serialize_executable

from lilax._backend.utils import max_logging

_FORMAT_VERSION = 1


def _metadata_fingerprint(meta: dict[str, Any]) -> str:
  serialized = json.dumps(meta, sort_keys=True, default=str)
  return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def _dynamic_signature(args: tuple, kwargs: dict) -> str:
  leaves_with_paths = jax.tree_util.tree_flatten_with_path((args, kwargs))[0]
  parts = []
  for path, leaf in leaves_with_paths:
    if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
      desc = f"{tuple(leaf.shape)}:{leaf.dtype}"
    else:
      desc = re.sub(r"0x[0-9a-fA-F]+", "@", repr(leaf))
    parts.append(f"{jax.tree_util.keystr(path)}={desc}")
  return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def _ambient_mesh() -> jax.sharding.Mesh | None:
  mesh = jax.sharding.get_mesh()
  return None if mesh.empty else mesh


class _AotEntry:
  """Executables for one wrapped function, keyed by dynamic input signature."""

  def __init__(self, name: str, fn: Callable, static_argnames: tuple):
    self.name = name
    self.fn = fn
    self.static_argnames = tuple(static_argnames)
    self.py_signature = inspect.signature(fn)
    self.jitted = jax.jit(fn, static_argnames=static_argnames or None)
    self._compiled: dict[str, Any] = {}
    self._out_specs: dict[str, Any] = {}
    self._pending: dict[str, tuple] = {}
    self._adapters: dict[str, Any] = {}
    self._on_disk: set[str] = set()
    self._lock = threading.Lock()

  def _zeros_output(self, signature: str):
    spec = self._out_specs.get(signature)
    if spec is None:
      return None
    out_treedef, shapes_dtypes, out_shardings = spec
    zeros = [
        jax.device_put(jnp.zeros(shape, dtype), sharding)
        for (shape, dtype), sharding in zip(shapes_dtypes, out_shardings)
    ]
    return jax.tree_util.tree_unflatten(out_treedef, zeros)

  def _adapter_for(self, signature: str, treedef: Any, static: dict):
    adapter_jit = self._adapters.get(signature)
    if adapter_jit is None:

      def adapter(flat, _treedef=treedef, _static=static):
        return self.fn(**jax.tree_util.tree_unflatten(_treedef, flat), **_static)

      adapter_jit = jax.jit(adapter)
      with self._lock:
        self._adapters.setdefault(signature, adapter_jit)
        adapter_jit = self._adapters[signature]
    return adapter_jit

  def _canonicalize(self, args: tuple, kwargs: dict) -> tuple[dict, dict]:
    bound = self.py_signature.bind(*args, **kwargs)
    dynamic, static = {}, {}
    for name, val in bound.arguments.items():
      (static if name in self.static_argnames else dynamic)[name] = val
    return dynamic, static

  def _compile_and_record(self, signature: str, leaves: list, treedef: Any, static: dict):
    lowered = self._adapter_for(signature, treedef, static).lower(leaves)
    compiled = lowered.compile()
    info_leaves, out_treedef = jax.tree_util.tree_flatten(lowered.out_info)
    shapes_dtypes = [(tuple(x.shape), jnp.dtype(x.dtype)) for x in info_leaves]
    out_shardings = jax.tree_util.tree_leaves(compiled.output_shardings)
    with self._lock:
      self._compiled[signature] = compiled
      self._out_specs[signature] = (out_treedef, shapes_dtypes, out_shardings)
    return compiled

  def __call__(self, *args, **kwargs):
    if not _STATE.enabled:
      return self.jitted(*args, **kwargs)
    dynamic, static = self._canonicalize(args, kwargs)
    leaves, treedef = jax.tree_util.tree_flatten(dynamic)
    if any(isinstance(leaf, jax.core.Tracer) for leaf in leaves):
      return self.jitted(**dynamic, **static)
    signature = _dynamic_signature((), {**dynamic, **static})
    if _STATE.warmup_only:
      if signature not in self._compiled:
        self._compile_and_record(signature, leaves, treedef, static)
        with self._lock:
          if signature not in self._on_disk:
            self._pending[signature] = (leaves, treedef, static, _ambient_mesh())
      zeros = self._zeros_output(signature)
      if zeros is not None:
        return zeros
    compiled = self._compiled.get(signature)
    if compiled is not None:
      flat = self._align_inputs(compiled, leaves)
      if flat is not None:
        return compiled(flat)
      try:
        return compiled(leaves)
      except Exception as e:
        max_logging.log(f"[aot] {self.name}: compiled call failed ({e}); using jit")
    with self._lock:
      if signature not in self._pending and signature not in self._compiled:
        self._pending[signature] = (leaves, treedef, static, _ambient_mesh())
    return self._adapter_for(signature, treedef, static)(leaves)

  def _align_inputs(self, compiled: Any, leaves: list):
    try:
      flat_expected = jax.tree_util.tree_leaves(compiled.input_shardings)
      if len(flat_expected) != len(leaves):
        return None
      aligned = []
      for leaf, expected in zip(leaves, flat_expected):
        if not hasattr(leaf, "shape"):
          leaf = jnp.asarray(leaf)
        sharding = getattr(leaf, "sharding", None)
        if sharding is not None and sharding.is_equivalent_to(expected, leaf.ndim):
          aligned.append(leaf)
        else:
          aligned.append(jax.device_put(leaf, expected))
      return aligned
    except Exception as e:
      max_logging.log(f"[aot] {self.name}: cannot align inputs ({e}); using jit")
      return None

  def _path_for(self, signature: str) -> str:
    return os.path.join(_STATE.cache_dir, f"{self.name}-{_STATE.fingerprint}-{signature}.aotx")

  def load_from_disk(self) -> None:
    """Deserialize matching executables from a trusted cache directory."""
    pattern = os.path.join(_STATE.cache_dir, f"{self.name}-{_STATE.fingerprint}-*.aotx")
    for path in glob.glob(pattern):
      try:
        with open(path, "rb") as f:
          blob = pickle.load(f)
        if blob["format_version"] != _FORMAT_VERSION:
          continue
        execution_devices = list(_STATE.mesh.devices.flatten()) if _STATE.mesh is not None else None
        compiled = serialize_executable.deserialize_and_load(
            blob["payload"], blob["in_tree"], blob["out_tree"], execution_devices=execution_devices
        )
        signature = blob["dynamic_signature"]
        with self._lock:
          self._compiled[signature] = compiled
          self._on_disk.add(signature)
          if "out_shapes_dtypes" in blob:
            self._out_specs[signature] = (
                blob["out_tree"],
                [(tuple(shape), jnp.dtype(dtype)) for shape, dtype in blob["out_shapes_dtypes"]],
                jax.tree_util.tree_leaves(compiled.output_shardings),
            )
        max_logging.log(f"[aot] {self.name}: loaded {os.path.basename(path)} ({len(blob['payload']) / 1e6:.1f}MB)")
      except Exception as e:
        max_logging.log(f"[aot] {self.name}: load failed for {os.path.basename(path)} ({e}); will re-jit")

  def save_pending(self) -> int:
    """Lower and serialize recorded signatures; return the number saved."""
    saved = 0
    with self._lock:
      pending, self._pending = self._pending, {}
    for signature, (leaves, treedef, static, mesh) in pending.items():
      if signature in self._on_disk:
        continue
      try:
        compiled = self._compiled.get(signature)
        if compiled is None:
          mesh_ctx = contextlib.nullcontext() if mesh is None else jax.set_mesh(mesh)
          with mesh_ctx:
            compiled = self._compile_and_record(signature, leaves, treedef, static)
        payload, in_tree, out_tree = serialize_executable.serialize(compiled)
        blob = {
            "format_version": _FORMAT_VERSION,
            "payload": payload,
            "in_tree": in_tree,
            "out_tree": out_tree,
            "dynamic_signature": signature,
        }
        out_spec = self._out_specs.get(signature)
        if out_spec is not None:
          blob["out_shapes_dtypes"] = [(list(shape), str(dtype)) for shape, dtype in out_spec[1]]
        path = self._path_for(signature)
        tmp_path = f"{path}.tmp.{os.getpid()}"
        with open(tmp_path, "wb") as f:
          pickle.dump(blob, f)
        os.replace(tmp_path, path)
        with self._lock:
          self._compiled[signature] = compiled
          self._on_disk.add(signature)
        saved += 1
        max_logging.log(f"[aot] {self.name}: serialized {os.path.basename(path)} ({len(payload) / 1e6:.1f}MB)")
      except Exception as e:
        max_logging.log(f"[aot] {self.name}: serialize failed ({e}); shape stays on jit")
    return saved


class _State:
  def __init__(self):
    self.enabled = False
    self.cache_dir = ""
    self.fingerprint = ""
    self.mesh = None
    self.warmup_only = False


_STATE = _State()
_REGISTRY: list[_AotEntry] = []
_LOAD_THREADS: list[threading.Thread] = []


def cached_jit(fn: Callable, static_argnames: tuple = ()) -> Callable:
  """Wrap a function with JIT and the optional installed AOT cache."""
  name = f"{fn.__module__.rsplit('.', 1)[-1]}.{fn.__name__}"
  entry = _AotEntry(name, fn, static_argnames)
  _REGISTRY.append(entry)
  return entry


def install(cache_dir: str, meta: dict[str, Any], mesh: Any) -> None:
  """Install the cache and load matching executables in background threads.

  Only use a trusted cache directory: loading its pickle and executable can run code.
  """
  wait_for_loads()
  _STATE.cache_dir = ""
  _STATE.fingerprint = ""
  _STATE.mesh = None
  _STATE.enabled = False
  for entry in _REGISTRY:
    with entry._lock:
      entry._compiled.clear()
      entry._out_specs.clear()
      entry._pending.clear()
      entry._adapters.clear()
      entry._on_disk.clear()
  if not cache_dir:
    return
  os.makedirs(cache_dir, exist_ok=True)
  _STATE.cache_dir = cache_dir
  _STATE.fingerprint = _metadata_fingerprint(meta)
  _STATE.mesh = mesh
  _STATE.enabled = True
  for entry in _REGISTRY:
    thread = threading.Thread(target=entry.load_from_disk, name=f"aot-load-{entry.name}", daemon=True)
    thread.start()
    _LOAD_THREADS.append(thread)


def wait_for_loads() -> None:
  """Wait for background deserialization to finish."""
  for thread in _LOAD_THREADS:
    thread.join()
  _LOAD_THREADS.clear()


def save_pending() -> int:
  """Synchronously serialize all recorded shapes and return the number saved."""
  if not _STATE.enabled:
    return 0
  return sum(entry.save_pending() for entry in _REGISTRY)


@contextlib.contextmanager
def warmup_mode():
  """Compile wrapped functions and return zero outputs without executing them."""
  _STATE.warmup_only = _STATE.enabled
  try:
    yield
  finally:
    _STATE.warmup_only = False


def in_warmup() -> bool:
  """Return whether zero-execution warmup is active."""
  return _STATE.warmup_only


@contextlib.contextmanager
def real_execution():
  """Temporarily suspend zero-execution warmup."""
  previous = _STATE.warmup_only
  _STATE.warmup_only = False
  try:
    yield
  finally:
    _STATE.warmup_only = previous
