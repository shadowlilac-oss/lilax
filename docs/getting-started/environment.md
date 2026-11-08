# Set up your environment

Lilax requires Python 3.12 or later and a JAX environment for your accelerator. Keep dependencies in a container or project virtual environment.

Public imports, including inference imports, currently load both JetStream and Tunix. Installing only `.[tpu,generation]` leaves Tunix missing. Use the shipped container recipe below for TPU setup.

## Set up a TPU container

Use an x86_64 Linux TPU host with Docker and access to the TPU devices. From your checkout's `lilax` directory, where `Dockerfile.tpu` and `pyproject.toml` live, build the supplied image:

```bash
docker build --platform linux/amd64 -f Dockerfile.tpu -t lilax-tpu .
```

The image includes Lilax's TPU training dependencies, CPU Torch, Tunix, and JetStream. It supplies the
dependencies needed for public imports, so a separate `generation` install is unnecessary inside this image.
The [Dockerfile](../../Dockerfile.tpu) specifies the package versions and build tools.

On the TPU host, open a shell in the image:

```bash
docker run --rm --privileged --network host -it lilax-tpu bash
```

The working directory is `/opt/lilax`. The device access and host networking follow Google's [TPU container instructions](https://docs.cloud.google.com/tpu/docs/run-in-container). Configure multi-host execution separately through [launch configuration](../guides/execution/launch.md).

## TPU vLLM workflows

Before using `create_vllm_llm`, `VllmScorer`, vLLM GRPO rollouts, or a Lilax evaluation endpoint, install the supplied TPU-vLLM stack. `Dockerfile.tpu` does not install it. The `vllm` extra installs plain `vllm` and does not supply the required `tpu-inference` integration or adapter registration.

Continue inside the TPU container above, which supplies the build tools and Python build dependencies.
Install workflow extras before the pinned stack. The [pinned upstream requirements](https://github.com/vllm-project/tpu-inference/blob/b67ae5f8f234fd559cf4b376840e7bb9ae6d3275/requirements.txt)
conflict with Lilax's Qwix and FastAPI bounds; the supplied installer uses dependency overrides. The commands
below preserve those overrides with `--no-deps`. They include serving and evaluation dependencies:

```bash
apt-get update
apt-get install -y --no-install-recommends git
export VLLM_TARGET_DEVICE=tpu
export UV_TORCH_BACKEND=cpu
export VLLM_VERSION_OVERRIDE=0.30.0+tpu
python -m uv pip install --system -e '.[serving,evaluation]'
python -m uv pip install --system \
    -r src/lilax/_backend/assets/dependencies/extra_deps/post_train_github_deps.txt \
    --no-deps --no-build-isolation
python -m uv pip install --system src/lilax/_backend/assets/integration/vllm --no-deps
```

The requirements file pins these exact revisions:

| Package | Revision |
| --- | --- |
| `google-tunix` | `385551df4643ee3ac3205e813a93c2ff217dffa5` |
| `tpu-inference` | `b67ae5f8f234fd559cf4b376840e7bb9ae6d3275` |
| `vllm` | `d626108b1841888ec90aced33367149a6bbc7e4b` |

The final command installs Lilax's vLLM plugin metadata. Keep `--no-deps` and `--no-build-isolation` on the
pinned install to retain the prepared dependencies. The [upstream quickstart](https://github.com/vllm-project/tpu-inference/blob/b67ae5f8f234fd559cf4b376840e7bb9ae6d3275/docs/getting_started/quickstart.md)
describes TPU deployment. The NVIDIA container below plus a plain `vllm` extra does not provide this TPU stack.

## Set up an NVIDIA GPU container

Use the checkout's `Dockerfile.cuda` for the supported container setup. It pins the NVIDIA JAX `26.10-py3` base image by digest and builds the pinned Grain wheel for CPython 3.12 on x86_64. The build installs Lilax with the `ngc-image` dependencies through the repository's `install_ngc.py` installer.

The build downloads dependencies and creates a container image. From the checkout's `lilax` directory, use that directory as the build context because the Dockerfile uses `COPY . .`.

```bash
docker build -f Dockerfile.cuda -t lilax-cuda .
```

The image's working directory is `/opt/lilax`. On a host with a compatible NVIDIA driver and [NVIDIA Container Toolkit configured for Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), start a GPU-enabled shell with:

```bash
docker run --rm --gpus all -it lilax-cuda bash
```

The `ngc` and `ngc-image` extras describe NVIDIA container dependencies. They do not replace the pinned base image and installer steps in `Dockerfile.cuda`.

## Select optional dependencies

Optional capabilities have separate extras:

| Workflow | Extra |
| --- | --- |
| Public model imports and generation | JetStream through `generation`, plus Tunix and accelerator dependencies as above |
| Serving | `serving`; for vLLM, also the [TPU-vLLM setup](#tpu-vllm-workflows) |
| Evaluation | `evaluation` |
| Audio and video | `omni` |
| Dataset exports | `data-operations` |
| Conversion agent | `conversion-agent` |
| TFDS | `tfds` |
| Diagnostics | `diagnostics` |
| Prompt optimization | `gepa` |
| MLPerf | `mlperf` |

Hub downloads use Hugging Face authentication. Pass `token` to loading factories when needed, or set `HF_TOKEN`. The Hub library gives `HF_TOKEN` precedence over its stored token. Lilax's configuration loader also uses `HF_TOKEN` when `hf_access_token` is empty. See the [Hub environment-variable reference](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hf_token) and [Lilax configuration overrides](../reference/models/models.md#environment-overrides) for precedence and `M_<KEY>` collisions.

Continue with the [quickstart](quickstart.md). For physical device placement and multi-node execution, [configure and launch your Python script](../guides/execution/launch.md).
