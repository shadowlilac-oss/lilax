"""Config-driven OpenAI-compatible benchmark server."""

import asyncio
import json
import logging
import os
import queue
import threading
import time
import uuid
from typing import Union

from fastapi import FastAPI, HTTPException
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
from openai_harmony import HarmonyEncodingName, load_harmony_encoding
import uvicorn

from lilax._backend.benchmarks.api_server import response_utils, server_utils
from lilax._backend.benchmarks.api_server.maxtext_generator import MaxTextGenerator
from lilax.serving.server_models import (
    ChatCompletionRequest, ChatCompletionResponse, CompletionRequest, CompletionResponse,
)

logger = logging.getLogger(__name__)
BATCH_TIMEOUT_S = float(os.environ.get("MAXTEXT_BATCH_TIMEOUT_S", "0.1"))
REQUEST_TIMEOUT_S = int(os.environ.get("MAXTEXT_REQUEST_TIMEOUT_S", "36000"))
MAX_REQUEST_SIZE = int(os.environ.get("MAXTEXT_REQUEST_SIZE", "655360"))


class MaxTextServer:
  """Own the generator, request queue, app, and multihost inference loop."""

  def __init__(self, config):
    logging.basicConfig(level=logging.WARNING)
    print("Initializing MaxTextGenerator and JAX distributed system...")
    self.llm = MaxTextGenerator(config)
    self.rank = jax.process_index()
    logger.setLevel(logging.INFO)
    logger.info("MaxTextGenerator initialization complete.")
    self.harmony_enc = None
    try:
      self.harmony_enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
      logger.info("Harmony encoding for gpt-oss loaded successfully.")
    except ImportError:
      logger.warning("openai_harmony not installed. GPT-OSS Harmony format will not be available.")
    except (RuntimeError, ValueError) as error:
      logger.error("Failed to load Harmony encoding: %s", error, exc_info=True)
    self.request_queue = queue.Queue()
    self.response_dict = {}
    self.response_lock = threading.Lock()
    self.app = FastAPI()
    self.app.add_api_route("/v1/completions", self.create_completion, methods=["POST"], response_model=CompletionResponse)
    self.app.add_api_route("/v1/chat/completions", self.create_chat_completion, methods=["POST"],
                           response_model=ChatCompletionResponse)
    self.app.add_api_route("/", self.health_check, methods=["GET"])

  async def _queue_and_wait_for_response(self, request: Union[CompletionRequest, ChatCompletionRequest]):
    request_id = f"req_{uuid.uuid4().hex}"
    self.request_queue.put((request_id, request))
    start_time = time.time()
    while time.time() - start_time < REQUEST_TIMEOUT_S:
      with self.response_lock:
        if request_id in self.response_dict:
          response_data = self.response_dict.pop(request_id)
          if "error" in response_data:
            raise HTTPException(status_code=500, detail=response_data["error"])
          return response_data
      await asyncio.sleep(0.05)
    raise HTTPException(status_code=504, detail="Request timed out.")

  async def create_completion(self, request: CompletionRequest):
    return await self._queue_and_wait_for_response(request)

  async def create_chat_completion(self, request: ChatCompletionRequest):
    return await self._queue_and_wait_for_response(request)

  def health_check(self):
    return {"status": "ok", "message": "MaxText API server is running."}

  def _collect_batched_requests(self):
    batched_items = []
    start_time = time.time()
    while len(batched_items) < self.llm.batch_size:
      remaining = BATCH_TIMEOUT_S - (time.time() - start_time)
      if remaining <= 0:
        break
      try:
        batched_items.append(self.request_queue.get(timeout=remaining))
      except queue.Empty:
        break
    return batched_items

  def _prepare_batch_for_broadcast(self, batched_items):
    _, first_request = batched_items[0]
    logprobs_param = None
    if isinstance(first_request, ChatCompletionRequest):
      if first_request.logprobs:
        logprobs_param = first_request.top_logprobs if first_request.top_logprobs is not None else 1
    else:
      logprobs_param = first_request.logprobs
    params = {
        "max_tokens": first_request.max_tokens, "logprobs": logprobs_param,
        "echo": getattr(first_request, "echo", False), "stop": first_request.stop,
        "temperature": first_request.temperature, "seed": first_request.seed,
        "top_k": first_request.top_k, "top_p": first_request.top_p,
    }
    all_prompts = []
    request_info_map = []
    for request_id, request in batched_items:
      is_chat = isinstance(request, ChatCompletionRequest)
      try:
        prompts = server_utils.get_prompts_for_request(request, self.llm)
        all_prompts.extend(prompts)
        request_info_map.append((request_id, request, is_chat, len(prompts)))
      except ValueError as error:
        logger.error("Failed to parse request %s: %s", request_id, error)
        with self.response_lock:
          self.response_dict[request_id] = {"error": f"Invalid request format or missing chat template: {error}"}
    if not all_prompts:
      return 0, b"", []
    payload_bytes = json.dumps({"prompts": all_prompts, "params": params}).encode("utf-8")
    payload_len = len(payload_bytes)
    if payload_len > MAX_REQUEST_SIZE:
      logger.error("Batched request is too large (%d bytes > %d)", payload_len, MAX_REQUEST_SIZE)
      for request_id, _, _, _ in request_info_map:
        with self.response_lock:
          self.response_dict[request_id] = {"error": "Batched request payload is too large."}
      return 0, b"", []
    return payload_len, payload_bytes, request_info_map

  def _process_results(self, completions, request_info_map, payload):
    logger.info("Batched generation finished. Processing %d completions.", len(completions))
    completion_index = 0
    for request_id, request, is_chat, num_prompts in request_info_map:
      completions_for_request = completions[completion_index:completion_index + num_prompts]
      prompts_for_request = payload["prompts"][completion_index:completion_index + num_prompts]
      completion_index += num_prompts
      response = response_utils.create_response(
          request, completions_for_request, prompts_for_request, is_chat, self.llm, self.harmony_enc,
      )
      with self.response_lock:
        self.response_dict[request_id] = response

  def main_loop(self):
    while True:
      payload_len, payload_bytes, request_info_map = 0, b"", []
      if jax.process_index() == 0:
        batched_items = self._collect_batched_requests()
        if batched_items:
          payload_len, payload_bytes, request_info_map = self._prepare_batch_for_broadcast(batched_items)
      data_to_broadcast = (
          jnp.array([payload_len], dtype=jnp.int32),
          jnp.pad(jnp.frombuffer(payload_bytes, dtype=jnp.uint8), (0, MAX_REQUEST_SIZE - payload_len)),
      )
      received_len_array, received_data_array = multihost_utils.broadcast_one_to_all(data_to_broadcast)
      received_len = int(received_len_array[0])
      if received_len == 0:
        time.sleep(0.01)
        continue
      payload = json.loads(received_data_array[:received_len].tobytes().decode("utf-8"))
      try:
        if jax.process_index() == 0:
          logger.info("Starting batched generation for %d prompts with params: %s", len(payload["prompts"]), payload["params"])
        completions = self.llm.generate_batch(prompts=payload["prompts"], **payload["params"])
        if jax.process_index() == 0:
          self._process_results(completions, request_info_map, payload)
      except (ValueError, RuntimeError) as error:
        logger.error("Inference failed for batch: %s", error, exc_info=True)
        if jax.process_index() == 0:
          for request_id, _, _, _ in request_info_map:
            with self.response_lock:
              self.response_dict[request_id] = {"error": f"Inference failed: {error}"}

  def run(self):
    server_thread = None
    server = None
    if jax.process_index() == 0:
      logging_config = {
          "version": 1,
          "disable_existing_loggers": False,
          "formatters": {
              "default": {"()": "uvicorn.logging.DefaultFormatter", "fmt": f"%(levelprefix)s RANK {self.rank}: %(message)s", "use_colors": None},
              "access": {"()": "uvicorn.logging.AccessFormatter", "fmt": f'%(levelprefix)s RANK {self.rank}: %(client_addr)s - "%(request_line)s" %(status_code)s'},
          },
          "handlers": {
              "default": {"formatter": "default", "class": "logging.StreamHandler", "stream": "ext://sys.stderr"},
              "access": {"formatter": "access", "class": "logging.StreamHandler", "stream": "ext://sys.stdout"},
          },
          "loggers": {
              "": {"handlers": ["default"], "level": "INFO"},
              "uvicorn.error": {"level": "INFO"},
              "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
          },
      }
      config = uvicorn.Config(self.app, host="0.0.0.0", port=8000, log_config=logging_config)
      server = uvicorn.Server(config)
      server_thread = threading.Thread(target=server.run)
      logger.info("Starting Uvicorn server in a background thread on coordinator process %d...", jax.process_index())
      server_thread.start()
    try:
      logger.info("Process %d is entering the main processing loop.", jax.process_index())
      self.main_loop()
    except KeyboardInterrupt:
      logger.info("Process %d received KeyboardInterrupt. Shutting down.", jax.process_index())
    finally:
      if jax.process_index() == 0 and server is not None and server_thread is not None:
        logger.info("Stopping Uvicorn server...")
        server.should_exit = True
        server_thread.join()
        logger.info("Uvicorn server stopped.")
    logger.info("Process %d has exited.", jax.process_index())
