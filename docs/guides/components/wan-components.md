# Work with Wan diffusion components

Wan exposes model, autoencoder, text encoder, scheduler, and sharding components. Assemble these explicitly for advanced diffusion workflows.

## Load component settings

```python
from lilax.generation.wan import DiffusionConfig, WanSharding, load_wan_config

raw = load_wan_config()
config = DiffusionConfig.from_dict(raw)
sharding = WanSharding.from_rules(raw["logical_axis_rules"])
```

This reads bundled Wan 2.1 14B settings without loading weights or initializing a runtime. `process_config(raw)` additionally resolves runtime values and initializes JAX.

## Construct components

The following constructs the bundled large architecture with initialized weights. It requires resources for that architecture and does not load a pretrained checkpoint.

```python
from flax import nnx
from lilax.generation.wan import WanModel

transformer = WanModel(rngs=nnx.Rngs(0), **raw["transformer_config"])
```

`AutoencoderKLWan` and `UMT5EncoderModel` expose separate constructors. Supply their architecture settings and RNGs. There is no Lilax pretrained Wan pipeline factory.

## Create a denoising schedule

Take one step with same-shaped synthetic sample and prediction arrays. This example uses the scheduler defaults, including 100 inference steps.

```python
import jax.numpy as jnp
from lilax.generation.wan import FlaxFlowMatchScheduler

scheduler, state = FlaxFlowMatchScheduler.from_config({})
state = scheduler.set_timesteps(state)
sample = jnp.ones((1,), dtype=scheduler.dtype)
model_output = jnp.ones_like(sample)
output = scheduler.step(state, model_output, state.timesteps[0], sample)
sample, state = output.prev_sample, output.state
print(sample.shape, bool(jnp.all(sample < 1)), state.num_inference_steps)
```

Expected output:

```text
(1,) True 100
```

`from_config` returns both scheduler and state. Keep both `output.prev_sample` and `output.state` for the next step, and pass the next entry in `state.timesteps`. The flow-match scheduler returns the schedule state unchanged; it does not advance a timestep cursor.

## Encode and decode a video

Construct the VAE with the bundled architecture settings and initialized weights. This requires memory for the full VAE even though the input below is small. It does not load pretrained weights.

```python
from flax import nnx
import jax.numpy as jnp
from lilax.generation.wan import AutoencoderKLWan, AutoencoderKLWanCache, load_wan_config

raw = load_wan_config()
vae = AutoencoderKLWan(rngs=nnx.Rngs(0), **raw["vae_config"])
cache = AutoencoderKLWanCache(vae)
spatial_factor = 2 ** len(vae.temperal_downsample)
video = jnp.zeros((1, 1, spatial_factor, spatial_factor, 3), dtype=jnp.float32)
posterior = vae.encode(video, cache).latent_dist
latents = posterior.mode()
decoded = vae.decode(latents, cache).sample
print(video.shape, latents.shape, decoded.shape)
```

Expected output:

```text
(1, 1, 8, 8, 3) (1, 1, 1, 1, 16) (1, 1, 8, 8, 3)
```

The input is one RGB frame that encodes to one latent spatial position. Arrays use batch, time, height, width, and channel order. `mode()` selects the posterior mean without sampling. With initialized weights, the decoded values are not a trained reconstruction. Decoding clips values to `[-1, 1]`. Encoding and decoding each reset and update the supplied feature cache.

The default `vae_encode_chunk=4` is a multiple of the bundled temporal downsampling factor of 4. A different positive encode chunk must remain a multiple of that factor; `-1` processes the remaining frames together. Both encode and decode chunks must be positive or `-1`, otherwise construction raises `AssertionError`.

See [Wan components](../../reference/components/wan-components.md) and [diffusion schedulers](../../reference/components/diffusion-schedulers.md) for signatures and shapes.
