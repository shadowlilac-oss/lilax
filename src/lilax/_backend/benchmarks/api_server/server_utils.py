"""Prompt and debug helpers for the benchmark API server."""

from datetime import datetime, timezone
import json
import logging
import os

from fastapi import HTTPException

from lilax._backend.benchmarks.api_server.encoding import encoding_dsv32
from lilax._backend.input_pipeline import encoding_dsv4

DEBUG_MODE = os.environ.get("MAXTEXT_SERVER_DEBUG", "0") == "1"
DEBUG_LOG_FILE = os.environ.get("MAXTEXT_DEBUG_LOG_FILE", "benchmarks/api_server/server_debug_log.jsonl")
DISABLE_DSV32_ENCODING = os.environ.get("DISABLE_DSV32_ENCODING", "0") == "1"
DISABLE_DSV4_ENCODING = os.environ.get("DISABLE_DSV4_ENCODING", "0") == "1"
logger = logging.getLogger(__name__)


def is_dsv32_encoding_enabled(model_name):
  return not DISABLE_DSV32_ENCODING and "deepseek3.2" in model_name.lower()


def is_dsv4_encoding_enabled(model_name):
  if DISABLE_DSV4_ENCODING:
    return False
  model_name_lower = model_name.lower()
  return "deepseek4" in model_name_lower or "deepseekv4" in model_name_lower or "deepseek-v4" in model_name_lower


def log_debug_event(request_id, event_type, content):
  if not DEBUG_MODE:
    return
  try:
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id, "event": event_type, "content": content,
    }
    with open(DEBUG_LOG_FILE, "a", encoding="utf8") as file:
      json.dump(log_entry, file)
      file.write("\n")
  except (OSError, TypeError) as error:
    logger.error("Error writing to debug log file '%s': %s", DEBUG_LOG_FILE, error)


def decode_one_prompt(prompt, llm):
  if isinstance(prompt, str):
    return prompt
  if isinstance(prompt, list) and (len(prompt) == 0 or isinstance(prompt[0], int)):
    try:
      return llm.tokenizer.decode(prompt)
    except (ValueError, TypeError):
      print("Return empty string on decoding error")
      return ""
  raise ValueError("Unsupported prompt item type")


def normalize_prompts(prompt, llm):
  if isinstance(prompt, str):
    return [prompt]
  if isinstance(prompt, list):
    if len(prompt) == 0:
      return []
    first = prompt[0]
    if isinstance(first, str):
      return [str(item) for item in prompt]
    if isinstance(first, int):
      return [decode_one_prompt(prompt, llm)]
    if isinstance(first, list):
      return [decode_one_prompt(item, llm) for item in prompt]
  raise HTTPException(status_code=400, detail="Unsupported prompt type for this API.")


def get_prompts_for_request(request, llm):
  if hasattr(request, "messages"):
    thinking_mode = getattr(request, "thinking_mode", "thinking")
    if is_dsv32_encoding_enabled(request.model):
      messages = [message.model_dump(exclude_none=True) for message in request.messages]
      return [encoding_dsv32.encode_messages(messages, thinking_mode=thinking_mode, drop_thinking=True,
                                             add_default_bos_token=True)]
    if is_dsv4_encoding_enabled(request.model):
      messages = [message.model_dump(exclude_none=True) for message in request.messages]
      return [encoding_dsv4.encode_messages(messages, thinking_mode=thinking_mode, drop_thinking=True,
                                            add_default_bos_token=True)]
    messages = [message.model_dump() for message in request.messages]
    return [llm.tokenizer.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
  return normalize_prompts(request.prompt, llm)
