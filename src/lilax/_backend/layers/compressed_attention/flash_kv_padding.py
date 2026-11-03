"""Query and KV tile padding for compressed Flash Attention (DeepSeek-V4)."""


def compute_hca_padding(
    q_len: int,
    comp_len: int,
    block_q: int = 128,
    block_kv: int = 128,
    kv_len: int | None = None,
) -> tuple[int, int]:
  """Computes (pad_q, pad_kv_total) for HCA Flash Attention.

  Args:
    q_len: Unpadded query sequence length.
    comp_len: Number of compressed KV blocks appended after the local KV tokens.
    block_q: Query tile size the kernel is compiled for.
    block_kv: KV tile size the kernel is compiled for.
    kv_len: Unpadded local KV sequence length. Defaults to `q_len`, which is the
      self-attention case DeepSeek-V4 uses; pass it explicitly if the query and
      KV sequences can differ.

  Returns:
    A (pad_q, pad_kv_total) tuple.
  """
  if kv_len is None:
    kv_len = q_len
  pad_q = (block_q - (q_len % block_q)) % block_q
  total_kv_len = kv_len + comp_len
  pad_kv_total = (block_kv - (total_kv_len % block_kv)) % block_kv
  # Ensure padding KV columns exist whenever query padding is required
  if pad_q > 0 and pad_kv_total == 0:
    pad_kv_total = block_kv
  return pad_q, pad_kv_total
