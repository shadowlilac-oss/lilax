# AOT executable cache

Import from `lilax.generation.aot`. See [cache guide](../../guides/components/aot-cache.md).

## cached_jit

```python
cached_jit(fn: Callable, static_argnames: tuple = ()) -> Callable
```

Registers a JIT callable. Accepts the plain decorator form or a function as its first argument.

Returns the registered callable.

<details>
<summary>Parameters</summary>

- `fn` (`Callable`). Function to compile.
- `static_argnames` (`tuple`, optional, defaults to `()`). Names of arguments treated as static by JAX.

</details>

## install

```python
install(cache_dir: str, meta: dict[str, Any], mesh: Any) -> None
```

Enables the process-global cache and loads matching executables on background threads. Empty `cache_dir` disables it.
Cache directories must be trusted because deserialization can execute code. Installation does not establish compatibility across JAX versions or device topologies.

Returns `None`.

<details>
<summary>Parameters</summary>

- `cache_dir` (`str`). The trusted cache directory, created when needed.
- `meta` (`dict[str, Any]`). Metadata for JSON fingerprinting.
- `mesh` (`Any`). The execution mesh for deserialization.

</details>

## wait_for_loads

```python
wait_for_loads() -> None
```

Joins background loader threads and clears their registrations.

Returns `None`.

## save_pending

```python
save_pending() -> int
```

Synchronously serializes pending executables.

Returns the number saved, or zero when disabled.

## warmup_mode

```python
warmup_mode()
```

Returns a context manager that compiles registered functions and returns zero outputs while caching is enabled.

## in_warmup

```python
in_warmup() -> bool
```

Returns the current global warmup flag.

## real_execution

```python
real_execution()
```

Returns a context manager that temporarily suspends warmup and restores the previous flag on exit.
