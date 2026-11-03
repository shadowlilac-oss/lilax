"""Server warmup before evaluation."""

import asyncio
import logging

import aiohttp

from lilax._backend.eval.datasets.base import SampleRequest

logger = logging.getLogger(__name__)
_WARMUP_BUCKETS = [0, 16, 32, 64, 128, 256, 512, 1024]


def _sample_by_buckets(requests: list[SampleRequest]) -> list[SampleRequest]:
  sampled = []
  for start, end in zip(_WARMUP_BUCKETS[:-1], _WARMUP_BUCKETS[1:]):
    for request in requests:
      if start < len(request.prompt.split()) <= end:
        sampled.append(request)
        break
  return sampled


async def _send_warmup_requests(api_url: str, model: str, pairs: list[tuple[str, int]]) -> None:
  async def post(session: aiohttp.ClientSession, prompt: str, max_tokens: int) -> bool:
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens}
    try:
      async with session.post(api_url, json=payload) as response:
        return response.status == 200
    except aiohttp.ClientError:
      return False

  async with aiohttp.ClientSession() as session:
    results = await asyncio.gather(*[post(session, prompt, tokens) for prompt, tokens in pairs])
  failures = sum(1 for success in results if not success)
  if failures:
    logger.warning("Warmup: %d/%d requests failed.", failures, len(results))
  else:
    logger.info("Warmup complete (%d requests).", len(results))


def warmup_server(
    base_url: str, model: str, sample_requests: list[SampleRequest] | None = None, max_tokens: int = 16,
) -> None:
  if sample_requests:
    bucketed = _sample_by_buckets(sample_requests)
    logger.info("Running bucketed warmup: %d prompt-length buckets.", len(bucketed))
    pairs = [(request.prompt, max_tokens) for request in bucketed]
  else:
    logger.info("Running simple warmup (no dataset samples provided).")
    pairs = [("What is 1 + 1?", max_tokens)]
  asyncio.run(_send_warmup_requests(f"{base_url}/v1/completions", model, pairs))


async def _send_chat_warmup_batch(
    api_url: str, model: str, messages: list[dict[str, str]], concurrency: int,
    max_tokens: int, reasoning_effort: str | None,
) -> None:
  payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0}
  if reasoning_effort:
    payload["reasoning_effort"] = reasoning_effort
  timeout = aiohttp.ClientTimeout(total=600)
  async with aiohttp.ClientSession(timeout=timeout) as session:
    responses = await asyncio.gather(*[session.post(api_url, json=payload) for _ in range(concurrency)])
    failures = []
    for response in responses:
      if response.status != 200:
        failures.append(f"HTTP {response.status}: {(await response.text())[:500]}")
      response.release()
  if failures:
    raise RuntimeError(f"Chat warmup failed for {len(failures)}/{concurrency} requests: {failures[0]}")


def warmup_chat_server(
    base_url: str, model: str, concurrency: int, max_model_len: int, system_message: str | None,
    reasoning_effort: str | None = None, max_tokens: int = 16,
) -> None:
  max_words = max(8, (max_model_len - max_tokens) // 4)
  word_counts = sorted({min(size, max_words) for size in (32, 128, 512)})
  for word_count in word_counts:
    messages = []
    if system_message:
      messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": ("warmup " * word_count).strip()})
    logger.info("Running chat warmup (approx_words=%d concurrency=%d max_tokens=%d).", word_count, concurrency, max_tokens)
    asyncio.run(_send_chat_warmup_batch(
        f"{base_url}/v1/chat/completions", model, messages, concurrency, max_tokens, reasoning_effort,
    ))
