"""Detection and combination of conversational dataset columns."""

import datasets


def combine_columns(example, columns, data_column):
  """Combine columns such as 'prompt' and 'completion' for sft training"""
  assert len(columns) > 1
  combined = []
  if len(columns) == 2:
    prompt, completion = (example[column] for column in columns)
    if len(prompt) != len(completion) or any(message["role"] != "user" for message in prompt):
      example[data_column] = prompt + completion
      return example
  for i in range(len(example[columns[0]])):
    for c in columns:
      combined.append(example[c][i])
  example[data_column] = combined
  return example


def is_conversational(features, data_columns):
  """Check if data is in a conversational format.
  Examples:

  features = {'prompt': [{'content': Value(dtype='string', id=None), 'role': Value(dtype='string', id=None)}],
              'completion': [{'content': Value(dtype='string', id=None), 'role': Value(dtype='string', id=None)}]}
  data_columns = ["prompt", "completion"]
  is_conversational(features, data_columns) return True.

  features = {'prompt': [Value(dtype='string', id=None)], 'completion': [Value(dtype='string', id=None)]}
  data_columns = ["prompt", "completion"]
  is_conversational(features, data_columns) returns False.
  """
  for column in data_columns:
    messages = features[column]
    if isinstance(messages, list):
      if messages and isinstance(messages[0], dict) and {"role", "content"}.issubset(messages[0]):
        return True
    if isinstance(messages, datasets.Sequence):
      if isinstance(messages.feature, dict) and "role" in messages.feature and "content" in messages.feature:
        return True

  return False
