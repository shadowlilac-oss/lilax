"""Per-learner references to environments awaiting batched rewards."""

import threading
import uuid


ENVIRONMENT_TOKEN = "_lilax_environment_token"


class EnvironmentRegistry:
  def __init__(self):
    self.lock = threading.Lock()
    self.entries = {}
    self.closed = False

  def add(self, environment):
    with self.lock:
      if self.closed:
        raise RuntimeError("The environment reward registry is closed.")
      token = uuid.uuid4().hex
      self.entries[token] = (environment, "open")
      return token

  def mark_closed(self, token):
    with self.lock:
      entry = self.entries.get(token)
      if entry is not None and entry[1] == "open":
        self.entries[token] = (entry[0], "closed")

  def mark_timeout(self, token):
    with self.lock:
      entry = self.entries.get(token)
      if entry is not None:
        self.entries[token] = (entry[0], "timed_out")

  def resolve(self, tokens):
    with self.lock:
      if len(set(tokens)) != len(tokens):
        raise RuntimeError("Duplicate environment reward tokens in a batch.")
      entries = [self.entries.get(token) for token in tokens]
      if any(entry is None or entry[1] != "closed" for entry in entries):
        raise RuntimeError("An environment is missing or its close did not complete before reward computation.")
      return [entry[0] for entry in entries]

  def release(self, tokens):
    with self.lock:
      for token in tokens:
        self.entries.pop(token, None)

  def shutdown(self):
    with self.lock:
      self.closed = True
      self.entries.clear()
