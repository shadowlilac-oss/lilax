"""Qwen3 prompt and continuation rendering for tool rollouts."""

import json

import numpy as np
from tunix.rl.agentic.parser.chat_template_parser import parser


class QwenFactoryParser(parser.QwenChatTemplateParser):
  def __init__(self, tokenizer, expected_template):
    if tokenizer.chat_template != expected_template:
      raise ValueError("environment_factory currently requires the bundled Qwen3 chat template.")
    super().__init__(tokenizer, enable_thinking=True)
    self.end_token = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if self.end_token != tokenizer.eos_token_id:
      raise ValueError("environment_factory requires Qwen3's <|im_end|> EOS token.")

  def parse(self, messages, add_generation_prompt=False, is_first_msg=False):
    if is_first_msg and messages and "tools" in messages[0]:
      messages = [dict(message) for message in messages]
      tools = json.loads(messages[0].pop("tools"))
      return self.tokenizer.apply_chat_template(
          messages, tools=tools, tokenize=False, add_generation_prompt=add_generation_prompt,
      )
    return super().parse(messages, add_generation_prompt=add_generation_prompt, is_first_msg=is_first_msg)

  def _parse_message(self, message):
    if message["role"] != "tool":
      return super()._parse_message(message)
    group = message.get("tool_group")
    block = "<tool_response>\n" + message["content"] + "\n</tool_response>"
    prefix = "<|im_start|>user\n" if group in (None, "first") else ""
    suffix = "<|im_end|>" if group in (None, "last") else ""
    return prefix + block + suffix

  def update_assistant_end_tokens(self, tokens):
    if len(tokens) and tokens[-1] == self.end_token:
      return tokens, 0
    return np.append(tokens, self.end_token).astype(np.int32), 1
