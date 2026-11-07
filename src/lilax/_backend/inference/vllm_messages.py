"""Format the existing text and image prompts for vLLM chat decoding."""

from typing import Any

from lilax._backend.common.common_types import Config


def build_chat_messages(config: Config) -> list[dict[str, Any]]:
  """Build the chat messages, prepending a system prompt when set."""
  messages = []
  if config.system_prompt:
    messages.append({"role": "system", "content": config.system_prompt})
  if config.use_multimodal and config.image_path:
    content = [{"type": "image"} for _ in config.image_path.split(",")]
    content.append({"type": "text", "text": config.prompt})
    messages.append({"role": "user", "content": content})
  else:
    messages.append({"role": "user", "content": config.prompt})
  return messages
