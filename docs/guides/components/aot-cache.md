# Cache compiled inference functions

Use the AOT cache for JAX-compatible functions whose compiled shapes you want to save and reload. Use only a trusted cache directory. Loading cached pickle and executable data can run code.

```python
import jax.numpy as jnp
from lilax.generation.aot import cached_jit, install, save_pending, wait_for_loads, warmup_mode

@cached_jit
def double(values):
	return values * 2

install("./trusted-aot-cache", {"workload": "double-example"}, mesh=None)
wait_for_loads()
inputs = jnp.ones((4,))
with warmup_mode():
	double(inputs)
saved_count = save_pending()
result = double(inputs)
```

Warmup compiles shapes and returns zeros without running the wrapped computation. Consume the normal call's result, not the warmup output.

## Include your workload metadata

`meta` determines the cache fingerprint. Supply metadata appropriate to the functions and environment you cache. For distributed functions, supply the JAX mesh used for deserialization.

`install` changes process-global cache state and clears prior registrations' compiled entries. An empty cache directory disables caching.

See [AOT reference](../../reference/components/aot-cache.md).
