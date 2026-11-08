# Tokenize text and process media

## Encode and decode text

```python
from lilax import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
inputs = tokenizer("Explain tokenization.", return_tensors="np", return_token_type_ids=False)
print(tokenizer.decode(inputs["input_ids"][0]))
```

The tokenizer is a Hugging Face tokenizer. Its encode, decode, chat-template, and save methods remain available.

## Format a chat prompt

```python
messages = [{"role": "user", "content": "Explain tokenization."}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="np", return_token_type_ids=False)
```

For trainer-managed conversational SFT, the `messages` column selects chat preparation. Formatting uses `tokenizer.chat_template`. If the tokenizer has no template, assign the model's actual template to `tokenizer.chat_template` before constructing `SFTTrainer`. Keep any system message at index zero. See [SFT record layouts](../training/sft.md#choose-a-record-layout) and the [Hugging Face chat-template guide](https://huggingface.co/docs/transformers/chat_templating).

`tokenize`, `use_chat_template`, and `use_truncation` persist as Lilax processing preferences, but their effects depend on the caller. `use_chat_template` is read by [distillation request filtering](distillation-requests.md) and generation tokenizer consumers. It does not select conversational SFT. `use_truncation` does not enable tokenizer truncation in SFT. Set `SFTConfig.max_length` to control SFT sequence length.

## Keep text and media paired

Use the matching processor and model. Media calls take formatted text strings and separate media arguments. For Qwen3-VL, format a string containing the model's image marker before processing:

```python
from PIL import Image
from lilax import AutoModelForCausalLM, AutoProcessor

model_id = "Qwen/Qwen3-VL-8B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
image = Image.open("./image.jpg")
messages = [{"role": "user", "content": "<|vision_start|><|image_pad|><|vision_end|>Describe this image."}]
text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, images=image, return_tensors="np")
sequences = model.generate(**inputs, max_new_tokens=32)
print(processor.tokenizer.decode(sequences[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

This repository ID is registered as `weaver-max`. The processor expands the image marker and returns token, attention, position, and media arrays accepted by `generate`. The returned sequences include the prompt. The slice above decodes only new tokens.

## Generate from a video

With the Qwen3-VL model and processor above, supply one local video path for one text string:

```python
messages = [{"role": "user", "content": "<|vision_start|><|video_pad|><|vision_end|>Describe this clip."}]
text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, videos="./clip.mp4", return_tensors="np")
sequences = model.generate(**inputs, max_new_tokens=32)
print(processor.tokenizer.decode(sequences[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Video processing supports the Qwen3-VL, Qwen3-Omni, and Qwen3.5 vision families. It accepts file paths, not decoded frame arrays or message-content dictionaries.

## Prepare an audio input

Separate audio files are supported by Ming. Supply its audio marker in the formatted text:

```python
audio_processor = AutoProcessor.from_pretrained("inclusionAI/Ming-3.0-audio-tiny")
text = (
	"<role>SYSTEM</role>detailed thinking off<|role_end|>"
	"<role>HUMAN</role><|audio_start|><|audio_pad|><|audio_end|>Transcribe this recording.<|role_end|>"
	"<role>ASSISTANT</role>\n<think></think>"
)
audio_inputs = audio_processor(text=text, audio="./recording.wav", return_tensors="np")
print(audio_inputs["input_ids"].shape, audio_inputs["audio_values"].shape)
```

This prepares token IDs and audio features. Ming's current model scope is inference; this example does not imply audio SFT support. Qwen audio processing extracts audio from a video with `use_audio_in_video=True` on a matching audio-capable processor; it does not accept separate audio files.

For batched processor calls, supply one video or audio file path per text string. Supply one image group per text string, using nested image lists for batches. Every row must contain the same media fields with matching processed shapes. Text padding does not make different media shapes compatible. The expanded media tokens must fit the requested input length and model context.

Nonempty `images_kwargs`, `videos_kwargs`, and `audio_kwargs` are rejected. Set supported preferences such as `max_num_images_per_example` or `video_max_grid_t`, `video_max_grid_h`, and `video_max_grid_w` when loading the processor.

## Supply paired records for SFT

SFT takes explicit prompt and completion columns alongside the media column. With the Qwen3-VL model above, use image paths in the processor's selected `image_column`:

```python
from lilax import SFTConfig, SFTTrainer
from lilax.data import Dataset

processor = AutoProcessor.from_pretrained(model_id, image_column="image")
model = AutoModelForCausalLM.from_pretrained(model_id)
dataset = Dataset.from_list([
	{"prompt": "<|image|>Describe the scene.", "completion": "A red bicycle beside a wall.", "image": "./bicycle.jpg"},
])
args = SFTConfig(output_dir="./image-sft", max_steps=10, per_device_train_batch_size=1, packing=False)
trainer = SFTTrainer(model, args=args, train_dataset=dataset, processing_class=processor)
trainer.train()
```

Replace the image and caption with your paired training data. Unlike direct processor calls, SFT formats the prompt and replaces `<|image|>` before tokenization. For video SFT with this family, select `image_column="video"` and use a record with `prompt="<|video|>Describe the clip."`, its `completion`, and `video="./clip.mp4"`.

Transform records with `dataset.map` before constructing the trainer. Multimodal SFT rejects `formatting_func`, which loses media and completion boundaries. Numerical choices illustrate API use.

See the [data reference](../../reference/data/data.md) for processing options.
