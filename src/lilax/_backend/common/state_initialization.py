"""Initialize training and decode state."""

from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np

from lilax._backend.multimodal import processor as mm_processor


def init_decode_state(apply_fn, params) -> TrainState:
  return TrainState(step=0, apply_fn=apply_fn, params=params, tx=None, opt_state={})


def init_training_state(apply_fn, params, tx):
  return TrainState.create(apply_fn=apply_fn, params=params, tx=tx)


def init_initial_state(model, tx, config, is_training, key):
  input_shape = (config.micro_batch_size_to_train_on, config.max_target_length)
  image_shape = mm_processor.get_dummy_image_shape_for_init(
      config.model_name, batch_size=config.micro_batch_size_to_train_on
  )
  audio_shape = mm_processor.get_dummy_audio_shape_for_init(config)
  params_key, dropout_key, aqt_key = jax.random.split(key, 3)
  model_vars = model.init(
      {"params": params_key, "dropout": dropout_key, "aqt": aqt_key},
      np.ones(input_shape, dtype=jnp.int32),
      np.ones(input_shape, dtype=jnp.int32),
      encoder_images=np.ones(image_shape, dtype=jnp.int32) if config.use_multimodal else None,
      encoder_audios=np.ones(audio_shape, dtype=jnp.float32) if config.use_audio else None,
  )
  if is_training:
    return init_training_state(model.apply, model_vars, tx)
  return init_decode_state(model.apply, model_vars)
