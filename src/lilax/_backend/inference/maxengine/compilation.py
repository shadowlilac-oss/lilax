"""Ahead-of-time generation compilation and array layout."""

from jax.experimental.layout import Format
import jax

if jax.__version_info__ >= (0, 6, 3):
  from jax.experimental.layout import Layout as DLL
else:
  from jax.experimental.layout import DeviceLocalLayout as DLL


def abstract_params(params):
  """Describe parameter arrays and their shardings for compilation."""
  return jax.tree.map(
      lambda x: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype, sharding=x.sharding)
      if isinstance(x, jax.Array)
      else None,
      params,
  )


def compile_init_decode_state(init_decode_state, decode_state_layouts, rng_shape):
  """Compile decode-state initialization with the selected output layouts."""
  return (
      jax.jit(init_decode_state, in_shardings=(None), out_shardings=decode_state_layouts)
      .lower(rng_shape)
      .compile()
  )


def compile_generate_and_get_layouts(generate_aot, params, decode_state, rng_shape, xla_flags=None):
  """Compile generation and return its selected input and output layouts."""
  param_layout = Format(DLL.AUTO)
  decode_state_layout = Format(DLL.AUTO)
  compiled_generate = (
      jax.jit(
          generate_aot,
          in_shardings=(param_layout, decode_state_layout, None),
          out_shardings=(Format(DLL.AUTO), Format(DLL.AUTO)),
          donate_argnames=("decode_state",),
      ).lower(params, decode_state, rng_shape)
  ).compile(compiler_options=xla_flags)

  arg_layouts, _ = compiled_generate.input_formats
  generate_out_layouts, _ = compiled_generate.output_formats
  return compiled_generate, arg_layouts[0], arg_layouts[1], generate_out_layouts


def iterated_layout(identity, arrays, layouts, xla_flags=None):
  """Lay out one tensor at a time to prevent OOMs."""
  def _layout(x, s, l):
    if x.format == l:
      return x
    dll = (
        (l.layout if jax.__version_info__ >= (0, 6, 3) else l.device_local_layout)
        if isinstance(l, Format)
        else l
    )
    f = jax.jit(identity, out_shardings=Format(dll, s)).lower(x).compile(compiler_options=xla_flags)
    y = f(x)
    jax.tree.map(lambda z: z.delete(), x)
    jax.block_until_ready(y)
    return y

  shardings = jax.tree.map(lambda x: x.sharding, arrays)
  arrays = jax.tree.map(_layout, arrays, shardings, layouts)
  return arrays
