"""Concurrent client for the completions endpoint."""

import asyncio
import logging
import time
from dataclasses import dataclass, field

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
  text: str = ""
  prompt_tokens: int = 0
  completion_tokens: int = 0
  error: str = ""
  latency_s: float = field(default=0.0)


async def generate_batch_async(
    prompts: list[str], base_url: str, model: str, max_tokens: int = 1024, temperature: float = 0.0,
    concurrency: int = 64, request_timeout: int = 600, max_retries: int = 3,
) -> list[GenerationResult]:
  api_url = f"{base_url}/v1/completions"
  semaphore = asyncio.Semaphore(concurrency)
  timeout = aiohttp.ClientTimeout(total=request_timeout)

  async def generate_one(session: aiohttp.ClientSession, prompt: str) -> GenerationResult:
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
    start = time.monotonic()
    last_error = ""
    for attempt in range(max_retries + 1):
      retryable = True
      try:
        async with semaphore:
          async with session.post(api_url, json=payload) as response:
            if response.status != 200:
              body = await response.text()
              last_error = f"HTTP {response.status}: {body[:200]}"
              retryable = response.status in (408, 409, 429) or response.status >= 500
            else:
              data = await response.json()
              choice = data["choices"][0]
              usage = data.get("usage", {})
              return GenerationResult(
                  text=choice.get("text", ""), prompt_tokens=usage.get("prompt_tokens", 0),
                  completion_tokens=usage.get("completion_tokens", 0), latency_s=time.monotonic() - start,
              )
      except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        last_error = str(exc)
      if not retryable:
        return GenerationResult(error=last_error, latency_s=time.monotonic() - start)
      if attempt < max_retries:
        logger.warning("Request failed (attempt %d/%d): %s. Retrying.", attempt + 1, max_retries + 1, last_error)
        await asyncio.sleep(5.0 * (attempt + 1))
    return GenerationResult(error=last_error, latency_s=time.monotonic() - start)

  async with aiohttp.ClientSession(timeout=timeout) as session:
    return list(await asyncio.gather(*[generate_one(session, prompt) for prompt in prompts]))


def generate_batch(
    prompts: list[str], base_url: str, model: str, max_tokens: int = 1024, temperature: float = 0.0,
    concurrency: int = 64, request_timeout: int = 600, max_retries: int = 3,
) -> list[GenerationResult]:
  return asyncio.run(generate_batch_async(
      prompts=prompts, base_url=base_url, model=model, max_tokens=max_tokens, temperature=temperature,
      concurrency=concurrency, request_timeout=request_timeout, max_retries=max_retries,
  ))
