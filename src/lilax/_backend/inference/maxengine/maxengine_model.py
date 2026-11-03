"""NNX model operation for MaxEngine inference."""

from flax import nnx


def run_model(
    abstract_model,
    graphdef,
    rest_state,
    params,
    cache_dict,
    decoder_input_tokens,
    decoder_positions,
    *,
    decoder_segment_ids=None,
    enable_dropout=False,
    model_mode,
    previous_chunk=None,
    true_length=None,
    slot=None,
    encoder_images=None,
    encoder_image_masks=None,
    encoder_videos=None,
    encoder_video_masks=None,
    encoder_video_grid_thw=None,
    encoder_audios=None,
):
  """Run an NNX model and return its logits and updated cache."""
  _, cache_state, _ = nnx.split(abstract_model, nnx.Cache, ...)
  nnx.replace_by_pure_dict(cache_state, cache_dict)
  model = nnx.merge(graphdef, params, cache_state, rest_state, copy=True)
  logits = model(
      decoder_input_tokens,
      decoder_positions,
      decoder_segment_ids=decoder_segment_ids,
      encoder_images=encoder_images,
      encoder_image_masks=encoder_image_masks,
      encoder_videos=encoder_videos,
      encoder_video_masks=encoder_video_masks,
      encoder_video_grid_thw=encoder_video_grid_thw,
      encoder_audios=encoder_audios,
      enable_dropout=enable_dropout,
      model_mode=model_mode,
      previous_chunk=previous_chunk,
      true_length=true_length,
      slot=slot,
  )
  new_cache = nnx.to_pure_dict(nnx.state(model, nnx.Cache))
  return logits, new_cache
