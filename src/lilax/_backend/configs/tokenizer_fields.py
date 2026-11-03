"""Native tokenizer and chat template configuration fields."""

from enum import Enum

from lilax._backend.configs.path_types import PathStr
from pydantic.fields import Field
from pydantic.main import BaseModel


class TokenizerType(str, Enum):
  """Supported tokenizer libraries."""

  SENTENCEPIECE = "sentencepiece"
  HUGGINGFACE = "huggingface"
  TIKTOKEN = "tiktoken"


class Tokenizer(BaseModel):
  """Configuration for the tokenizer."""

  vocab_size: int = Field(32_000, description="The size of the vocabulary.")
  tokenizer_path: None | PathStr = Field(
      None,
      description="Path to the tokenizer model file.",
  )
  tokenizer_type: TokenizerType = Field(TokenizerType.SENTENCEPIECE, description="The type of tokenizer.")
  use_chat_template: bool = Field(False, description="Whether to use the chat template for tokenization.")
  chat_template_path: str = Field("", description="Path to chat template json file.")
  chat_template: str = Field(
      "",
      description="Chat template to use with HF tokenizers. It should be a valid Jinja2-formatted template.",
  )
  data_template_path: str = Field(
      "",
      description="Path to a chat template file to be used when tokenizing the dataset.",
  )
  tokenize_train_data: bool = Field(True, description="If False, assumes the training dataset is pre-tokenized.")
  tokenize_eval_data: bool = Field(True, description="If False, assumes the evaluation dataset is pre-tokenized.")
  add_bos: bool = Field(True, description="Whether to add a beginning-of-sentence token.")
  add_eos: bool = Field(True, description="Whether to add an end-of-sentence token.")
  use_truncation: bool = Field(
      True,
      description="If False, use chunking for long sequences instead of truncation.",
  )
