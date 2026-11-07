"""Application text formatting before the existing SFT token and mask transforms."""

import warnings


def format_dataset(dataset, formatting_func, text_field):
  def format_batch(batch):
    texts = []
    for index in range(len(next(iter(batch.values())))):
      value = formatting_func({key: values[index] for key, values in batch.items()})
      if isinstance(value, str):
        texts.append(value)
      elif isinstance(value, list) and all(isinstance(text, str) for text in value):
        texts.extend(value)
      else:
        raise TypeError("formatting_func must return a string or a list of strings.")
    return {text_field: texts}

  columns = dataset.column_names
  if columns is None:
    columns = list(next(iter(dataset), {}))
  return dataset.map(format_batch, batched=True, remove_columns=columns)


def text_segments(example, text_field):
  return {text_field: [example[text_field]], "is_prompt": [False]}


def paired_segments(example, prompt_field, completion_field, tokenizer):
  prompt = example[prompt_field]
  tokens = prompt + example[completion_field]
  if tokenizer is not None:
    prompt = tokenizer(prompt)["input_ids"]
    tokens = tokenizer(tokens)["input_ids"]
  if tokens[:len(prompt)] != prompt:
    warnings.warn("The tokenized prompt differs from the prefix of prompt plus completion; check tokenizer boundaries.")
  return {prompt_field: [tokens[:len(prompt)], tokens[len(prompt):]], "is_prompt": [True, False]}
