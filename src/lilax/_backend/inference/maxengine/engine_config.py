"""Configuration view for an individual inference engine."""


class MaxEngineConfig:
  """Engine specific config class to allow using multiple MaxEngine instances in an inference run.
  TODO: evaluate the need for this given the restructured pyconfig.py
  """

  def __init__(self, keys):
    self.__dict__["keys"] = keys

  def __getattr__(self, attr):
    if attr not in self.keys:
      raise ValueError(f"Requested key {attr}, not in config")
    return self.keys[attr]

  def __setattr__(self, attr, value):
    raise ValueError

  def get_keys(self):
    return self.keys
