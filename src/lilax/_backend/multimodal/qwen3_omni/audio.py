"""Audio feature preprocessing for Qwen3-Omni."""

import numpy as np
import jax

from lilax._backend.multimodal import utils as mm_utils

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
DITHER = 0.0


def _np_extract_fbank_features(waveform_batch: np.ndarray) -> np.ndarray:
  """
  Compute the log-mel spectrogram of the provided audio, gives similar results to Whisper's original torch
  implementation with 1e-5 tolerance.
  """
  log_spec_batch = []
  mel_filters = mm_utils.mel_filter_bank(
      num_frequency_bins=1 + N_FFT // 2,
      num_mel_filters=128,
      min_frequency=0.0,
      max_frequency=8000.0,
      sampling_rate=SAMPLE_RATE,
      norm="slaney",
      mel_scale="slaney",
  )
  for waveform in waveform_batch:
    log_spec = mm_utils.spectrogram(
        waveform,
        mm_utils.window_function(N_FFT, "hann"),
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
        power=2.0,
        dither=0.0,
        mel_filters=mel_filters,
        log_mel="log10",
    )
    log_spec = log_spec[:, :-1]
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    log_spec_batch.append(log_spec)
  log_spec_batch = np.array(log_spec_batch)
  return log_spec_batch


def pre_process_audio_qwen3_omni(audio_array):
  """Preprocess audio for Qwen3-Omni model."""
  chunk_samples = 16000  # hop_length (160) * chunk_size (100)
  remainder = len(audio_array) % chunk_samples
  if remainder > 0:
    padding_size = chunk_samples - remainder
    audio_array = np.pad(audio_array, (0, padding_size), mode="constant")

  audio_features = np.expand_dims(audio_array, axis=0)  # Add batch dimension
  audio_features = _np_extract_fbank_features(audio_features)
  audio_features_mask = np.ones((audio_features.shape[0], audio_features.shape[2]), dtype=np.int32)
  return audio_features, audio_features_mask


def _get_feat_extract_output_lengths(input_lengths: jax.Array) -> jax.Array:
  """Computes the output length of the audio encoder convolutional layers.

  The audio encoder processes audio in chunks of 100 samples, applying
  multiple convolutional layers with stride 2. Each full 100-sample chunk
  produces 13 output tokens, and remaining samples are processed separately.

  Args:
    input_lengths: Input audio sequence lengths. Shape: (batch,) or scalar.

  Returns:
    Output sequence lengths after audio encoding. Same shape as input.
  """
  input_lengths_leave = input_lengths % 100
  feat_lengths = (input_lengths_leave - 1) // 2 + 1
  output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
  return output_lengths
