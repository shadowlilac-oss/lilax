# Prepare model-specific media prompts

Use the public processor for ordinary image, video, and audio input. The preprocessing namespace also exposes lower-level prompt formatting and initialization-shape helpers.

```python
from lilax import AutoConfig
from lilax.preprocessing.multimodal import get_dummy_image_shape_for_init, reformat_response

config = AutoConfig.from_pretrained("Qwen/Qwen3-VL-4B-Instruct", model_name="qwen3-vl-4b")
shape = get_dummy_image_shape_for_init(config.model_name, batch_size=1)
response = reformat_response("An image description.", config.model_name)
print(shape, response)
```

The shape is for model initialization rather than the shape of an arbitrary image. Response formatting adds family-specific end-of-turn tokens. Do not add them twice if another processor already supplied them.

For record conversion, preserve text and media together through [AutoProcessor](processing.md). See [media helper reference](../../reference/data/media-helpers.md).

## Process images with a model runtime

Use the model factory as the configuration producer when you own image preparation separately. Continue with the Qwen configuration above and supply `image_path` as your existing local image file. This opens the file and resolves the model runtime, including JAX initialization, before preprocessing a single RGB image. The wrapper does not load weights.

```python
from lilax import AutoModelForCausalLM
from lilax.preprocessing.multimodal import get_image_offsets, preprocess_image_for_training
from PIL import Image
import numpy as np

with Image.open(image_path) as image:
	pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
model = AutoModelForCausalLM.from_config(config)
runtime = model.runtime.resolve_config()
processed = preprocess_image_for_training(pixels, runtime.config)
offsets = get_image_offsets(runtime.config, processed)
```

This prepares media and token-offset information without running the model. The runtime supplies the resolved model-family settings. Unsupported image families raise `ValueError`.
