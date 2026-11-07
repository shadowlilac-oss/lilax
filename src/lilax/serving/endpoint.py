"""Own the existing evaluation server's startup and shutdown around a model."""

from lilax._backend.eval.runner.server_manager import VllmServerManager
from .configuration import EvaluationServerConfig
from ._model_source import serving_model_source


class ServingEndpoint:
  def __init__(self, model, *, config: EvaluationServerConfig, processing_class=None):
    self.model = model
    self.config = config
    self.processing_class = processing_class
    self.model_name = model.config.model_name
    self._manager = None
    self._checkpoint_directory = None
    self._token = None

  @property
  def base_url(self):
    if self._manager is None:
      raise RuntimeError("Enter the ServingEndpoint context before evaluating it.")
    return self._manager.base_url

  @property
  def concurrency(self):
    if self._manager is None:
      raise RuntimeError("Enter the ServingEndpoint context before evaluating it.")
    return self._manager.concurrency

  def __enter__(self):
    if self._manager is not None:
      raise RuntimeError("This ServingEndpoint is already running.")
    config, self._checkpoint_directory = serving_model_source(self.model)
    options = self.config.model_dump()
    try:
      self._manager = VllmServerManager(
          model_path=config.tokenizer_path, checkpoint_path=config.load_parameters_path,
          maxtext_model_name=self.model_name, dtype=config.weight_dtype.name,
          env={"HF_TOKEN": self._token or config.hf_access_token} if self._token or config.hf_access_token else None,
          **options,
      )
      self._manager.start()
    except BaseException:
      self.__exit__(None, None, None)
      raise
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    try:
      if self._manager is not None:
        self._manager.stop()
    finally:
      self._manager = None
      if self._checkpoint_directory is not None:
        self._checkpoint_directory.cleanup()
        self._checkpoint_directory = None
