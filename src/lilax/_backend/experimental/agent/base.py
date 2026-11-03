"""Google Gen AI client for checkpoint conversion agents."""

import os

from google import genai
from google.genai.types import GenerateContentConfig


class BaseAgent:
  """Generate text with the model used by the original conversion agent."""

  def __init__(self, model_id="gemini-2.5-pro"):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set.")
    self.client = genai.Client(api_key=api_key)
    self.model_id = model_id

  def generate_text(self, prompt, tools=None):
    config = GenerateContentConfig(tools=tools) if tools else None
    response = self.client.models.generate_content(model=self.model_id, contents=prompt, config=config)
    return response.text
