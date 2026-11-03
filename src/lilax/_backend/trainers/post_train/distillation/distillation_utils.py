"""Distillation strategies, data adapters, and checkpoint management."""

import abc
from typing import Any, Callable, Iterator, Literal, Sequence

import flax
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from orbax import checkpoint
from tunix.sft import peft_trainer

from lilax._backend.common.checkpoints import grain_utility
from lilax._backend.trainers.post_train.checkpoint_selection import TrainingCheckpointManager
from lilax._backend.utils import max_logging, maxtext_utils


@flax.struct.dataclass(frozen=True)
class DistillationForwardOutput:
  """Model outputs used for distillation."""

  logits: jax.Array
  out_projection_activations: jax.Array | None = None
  moe_lb_loss: jax.Array | None = None
  top_k_indices: jax.Array | None = None


@flax.struct.dataclass(frozen=True)
class MaxTextTrainingInput(peft_trainer.TrainingInput):
  """Tunix training input with MaxText-specific fields."""

  positions: jax.Array | None = None
  decoder_segment_ids: jax.Array | None = None
  targets: jax.Array | None = None
  targets_position: jax.Array | None = None
  targets_segmentation: jax.Array | None = None
  top_k_logits: jax.Array | None = None
  top_k_indices: jax.Array | None = None


class MaxTextToTunixIterator:
  """Adapt MaxText batch dictionaries to Tunix training inputs."""

  def __init__(self, maxtext_iterator: Iterator):
    self._iterator = maxtext_iterator

  def __iter__(self):
    return self

  def __next__(self) -> MaxTextTrainingInput:
    batch = next(self._iterator)
    if "inputs_segmentation" in batch:
      input_mask = batch["inputs_segmentation"] != 0
      seg_ids = batch["inputs_segmentation"]
    else:
      input_mask = jnp.ones_like(batch["inputs"], dtype=bool)
      seg_ids = None
    targets_segmentation = batch.get("targets_segmentation", jnp.ones_like(batch["targets"]))
    targets_position = batch.get("targets_position", batch.get("inputs_position"))
    return MaxTextTrainingInput(
        input_tokens=batch["inputs"], input_mask=input_mask, positions=batch["inputs_position"],
        decoder_segment_ids=seg_ids, targets=batch["targets"], targets_position=targets_position,
        targets_segmentation=targets_segmentation, top_k_logits=batch.get("top_k_logits"),
        top_k_indices=batch.get("top_k_indices"),
    )


_PPL_CE_CAP = 20.0
METRIC_TOTAL_LOSS = "distill/total_loss"
METRIC_HARD_LOSS = "distill/hard_loss"
METRIC_SOFT_LOSS = "distill/soft_loss"
METRIC_TEACHER_LOSS = "distill/teacher_loss"
METRIC_KL_DIV_T1 = "distill/kl_div_T1"
METRIC_KL_DIV_AT_T = "distill/kl_div_at_T"
METRIC_STUDENT_PERPLEXITY = "distill/student_perplexity"
METRIC_TEACHER_PERPLEXITY = "distill/teacher_perplexity"
METRIC_OUT_PROJ_FEATURE_LOSS = "distill/out_proj_feature_loss"
METRIC_MOE_LB_LOSS = "distill/moe_lb_loss"
METRIC_TEACHER_MOE_LB_LOSS = "distill/teacher_moe_lb_loss"
METRIC_TEMPERATURE = "distill/temperature"
METRIC_ALPHA = "distill/alpha"
METRIC_BETA_FEATURE = "distill/beta_feature"


def compute_schedule(
    step: jax.Array, max_steps: int, start_value: float, end_value: float | None, schedule_type: str,
) -> jax.Array:
  """Compute a constant, linear, or cosine scheduled value."""
  if end_value is None or schedule_type == "constant":
    return jnp.array(start_value, dtype=jnp.float32)
  progress = jnp.clip(step.astype(jnp.float32) / max_steps, 0.0, 1.0)
  if schedule_type == "linear":
    return start_value + (end_value - start_value) * progress
  elif schedule_type == "cosine":
    return end_value + (start_value - end_value) * 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
  else:
    raise ValueError(f"Unsupported schedule_type: {schedule_type!r}. Must be 'constant', 'linear', or 'cosine'.")


def weighted_mean(sum_count_pairs: Sequence[tuple[Any, Any]] | np.ndarray) -> float:
  """Aggregate sum and count pairs into a token-weighted mean."""
  arr = np.asarray(sum_count_pairs, dtype=np.float32)
  if arr.size == 0:
    return 0.0
  if arr.ndim == 1:
    arr = arr.reshape(1, -1)
  if arr.ndim != 2 or arr.shape[1] != 2:
    return 0.0
  total = float(arr[:, 1].sum())
  if total <= 0.0:
    return 0.0
  return float(arr[:, 0].sum() / total)


def calculate_distillation_tflops_per_device(student_config, teacher_config, is_offline: bool = False):
  """Return combined, student and teacher per-device TFLOPs per step."""
  student_tflops, _, _ = maxtext_utils.calculate_tflops_training_per_device(student_config, log=False)
  if is_offline or teacher_config is None:
    teacher_tflops = 0.0
  else:
    teacher_full, _, _ = maxtext_utils.calculate_tflops_training_per_device(teacher_config, log=False)
    teacher_tflops = teacher_full / 3.0
  return student_tflops + teacher_tflops, student_tflops, teacher_tflops


class DistillationStrategy(abc.ABC):
  """Base class for distillation strategies."""

  def __init__(self, student_forward_fn: Callable, teacher_forward_fn: Callable, vocab_size: int, pad_id: int = 0, **kwargs):
    self.student_forward_fn = student_forward_fn
    self.teacher_forward_fn = teacher_forward_fn
    self.vocab_size = vocab_size
    self.pad_id = pad_id

  @abc.abstractmethod
  def compute_loss(self, student_output, teacher_output, labels, step=None):
    """Compute the training loss and metrics."""
    raise NotImplementedError

  @abc.abstractmethod
  def compute_eval_loss(self, student_output, labels):
    """Compute the evaluation loss and metrics."""
    raise NotImplementedError

  @abc.abstractmethod
  def create_labels(self, targets, targets_segmentation=None, **kwargs):
    """Create labels for the loss."""
    raise NotImplementedError


class CombinedDistillationStrategy(DistillationStrategy):
  """Combine hard-label, distillation and optional feature losses."""

  def __init__(
      self, student_forward_fn: Callable[..., DistillationForwardOutput],
      teacher_forward_fn: Callable[..., DistillationForwardOutput], pad_id: int = 0,
      temperature: float = 2.0, alpha: float = 0.5, beta_feature: float = 0.0,
      layer_indices: list[int] | None = None,
      feature_loss_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array] | None = None,
      feature_loss_type: Literal["cosine", "l2"] = "cosine", cosine_distance_axis: int | tuple[int, ...] = -1,
      vocab_size: int = 0, alpha_end: float | None = None,
      alpha_schedule: Literal["constant", "linear", "cosine"] = "constant", temperature_end: float | None = None,
      temperature_schedule: Literal["constant", "linear", "cosine"] = "constant", beta_end: float | None = None,
      beta_schedule: Literal["constant", "linear", "cosine"] = "constant", max_steps: int = 1,
  ):
    super().__init__(student_forward_fn=student_forward_fn, teacher_forward_fn=teacher_forward_fn,
                     vocab_size=vocab_size, pad_id=pad_id)
    self.temperature = temperature
    self.alpha = alpha
    self.beta_feature = beta_feature
    self.layer_indices = jnp.array(layer_indices) if layer_indices is not None else None
    self.alpha_end = alpha_end
    self.alpha_schedule = alpha_schedule
    self.temperature_end = temperature_end
    self.temperature_schedule = temperature_schedule
    self.beta_end = beta_end
    self.beta_schedule = beta_schedule
    self.max_steps = max_steps
    if alpha_end is not None and not 0.0 <= alpha_end <= 1.0:
      raise ValueError(f"alpha_end must be in [0, 1], got {alpha_end}")
    if temperature_end is not None and temperature_end <= 0.0:
      raise ValueError(f"temperature_end must be > 0, got {temperature_end}")
    if beta_end is not None and beta_end < 0.0:
      raise ValueError(f"beta_end must be >= 0, got {beta_end}")
    if beta_feature == 0.0 and beta_end is not None and beta_end > 0.0:
      raise ValueError(
          f"distill_beta=0.0 but distill_beta_end={beta_end}. Feature extraction is disabled when "
          "distill_beta starts at 0.0 (the model does not sow intermediate activations). "
          "Set distill_beta to a small positive value (e.g., 1e-6) to enable feature extraction."
      )
    for param_name, schedule, end_value in [
        ("alpha", alpha_schedule, alpha_end), ("temperature", temperature_schedule, temperature_end),
        ("beta", beta_schedule, beta_end),
    ]:
      if schedule != "constant" and end_value is None:
        raise ValueError(
            f"{param_name}_schedule is '{schedule}' but {param_name}_end is None. "
            f"Set {param_name}_end to a target value or use schedule='constant'."
        )
    self.feature_loss_fn = feature_loss_fn
    if feature_loss_fn is None:
      if feature_loss_type == "cosine":
        def masked_cosine(student_features, teacher_features, mask):
          distance = optax.cosine_distance(student_features, teacher_features, axis=cosine_distance_axis, epsilon=1e-6)
          mask_b = mask.astype(distance.dtype)
          num_valid_terms = jnp.maximum(jnp.sum(mask_b), 1.0) * distance.shape[0]
          return jnp.sum(distance * mask_b[None, :, :]) / num_valid_terms
        self.feature_loss_fn = masked_cosine
      elif feature_loss_type == "l2":
        def masked_l2(student_features, teacher_features, mask):
          squared = jnp.mean(jnp.square(student_features - teacher_features), axis=-1)
          mask_b = mask.astype(squared.dtype)
          num_valid_terms = jnp.maximum(jnp.sum(mask_b), 1.0) * squared.shape[0]
          return jnp.sum(squared * mask_b[None, :, :]) / num_valid_terms
        self.feature_loss_fn = masked_l2
      else:
        raise ValueError(f"Unsupported feature_loss_type: {feature_loss_type!r}")

  def _get_scheduled_weights(self, step: jax.Array | None) -> tuple[jax.Array, jax.Array, jax.Array]:
    if step is None:
      return (jnp.array(self.alpha, dtype=jnp.float32), jnp.array(self.temperature, dtype=jnp.float32),
              jnp.array(self.beta_feature, dtype=jnp.float32))
    alpha = compute_schedule(step, self.max_steps, self.alpha, self.alpha_end, self.alpha_schedule)
    temperature = compute_schedule(step, self.max_steps, self.temperature, self.temperature_end, self.temperature_schedule)
    beta_feature = compute_schedule(step, self.max_steps, self.beta_feature, self.beta_end, self.beta_schedule)
    return alpha, temperature, beta_feature

  def compute_loss(self, student_output, teacher_output, labels, step=None):
    """Compute distillation loss and token-weighted metrics."""
    alpha, temperature, beta_feature = self._get_scheduled_weights(step)
    s_logits = student_output.logits.astype(jnp.float32)
    t_logits = teacher_output.logits.astype(jnp.float32)
    s_features = student_output.out_projection_activations
    t_features = teacher_output.out_projection_activations
    if (s_features is None or t_features is None) and self.beta_feature > 0.0:
      raise ValueError(
          "Features extracted from student or teacher model are None, but distill_beta > 0.0. "
          "Ensure the model architecture supports feature extraction (e.g., 'out_projection_activations' is sowed)."
      )
    mask = jnp.any(labels != 0, axis=-1).astype(jnp.float32)
    valid_count = jnp.sum(mask)
    safe_count = jnp.maximum(valid_count, 1.0)
    log_s_T_full = jax.nn.log_softmax(s_logits / temperature, axis=-1)
    log_s_1_full = jax.nn.log_softmax(s_logits, axis=-1)
    if getattr(teacher_output, "top_k_indices", None) is not None:
      t_p_T_sparse = jax.nn.softmax(t_logits / temperature, axis=-1)
      log_t_p_T_sparse = jax.nn.log_softmax(t_logits / temperature, axis=-1)
      s_logits_sparse = jnp.take_along_axis(s_logits, teacher_output.top_k_indices, axis=-1)
      log_s_T_sparse = jax.nn.log_softmax(s_logits_sparse / temperature, axis=-1)
      kl_softened_per_pos = jnp.sum(t_p_T_sparse * (log_t_p_T_sparse - log_s_T_sparse), axis=-1)
      ce_teacher_per_pos = jnp.zeros(s_logits.shape[:-1])
      kl_t1_sum = jnp.array(0.0)
    else:
      t_p_T = jax.nn.softmax(t_logits / temperature, axis=-1)
      t_p_1 = jax.nn.softmax(t_logits, axis=-1)
      kl_softened_per_pos = optax.kl_divergence(log_s_T_full, t_p_T)
      kl_t1_per_pos = optax.kl_divergence(log_s_1_full, t_p_1)
      ce_teacher_per_pos = optax.softmax_cross_entropy(logits=t_logits, labels=labels)
      kl_t1_sum = jnp.sum(kl_t1_per_pos * mask)
    kl_softened_sum = jnp.sum(kl_softened_per_pos * mask)
    soft_loss_sum_scaled = kl_softened_sum * (temperature**2)
    soft_loss_mean = soft_loss_sum_scaled / safe_count
    ce_student_per_pos = optax.softmax_cross_entropy(logits=s_logits, labels=labels)
    ce_student_sum = jnp.sum(ce_student_per_pos * mask)
    hard_loss_mean = ce_student_sum / safe_count
    ce_teacher_sum = jnp.sum(ce_teacher_per_pos * mask)
    base_logit_loss = (alpha * soft_loss_mean) + ((1.0 - alpha) * hard_loss_mean)
    feature_loss = jnp.array(0.0, dtype=jnp.float32)
    if self.beta_feature > 0.0:
      assert s_features is not None and t_features is not None
      if self.layer_indices is not None:
        s_features_sliced = jnp.take(s_features, self.layer_indices, axis=0)
        t_features_sliced = jnp.take(t_features, self.layer_indices, axis=0)
      else:
        s_features_sliced = s_features
        t_features_sliced = t_features
      s_features_sliced = s_features_sliced.astype(jnp.float32)
      t_features_sliced = t_features_sliced.astype(jnp.float32)
      feature_loss = beta_feature * self.feature_loss_fn(s_features_sliced, t_features_sliced, mask)
    total_loss = base_logit_loss + feature_loss
    moe_lb_loss = jnp.array(0.0)
    if student_output.moe_lb_loss is not None:
      moe_lb_loss = student_output.moe_lb_loss
      total_loss += moe_lb_loss
    teacher_moe_lb_loss = jnp.array(0.0)
    if teacher_output.moe_lb_loss is not None:
      teacher_moe_lb_loss = teacher_output.moe_lb_loss
    teacher_loss_mean = ce_teacher_sum / safe_count
    student_perplexity_step = jnp.exp(jnp.minimum(hard_loss_mean, _PPL_CE_CAP))
    teacher_perplexity_step = jnp.exp(jnp.minimum(teacher_loss_mean, _PPL_CE_CAP))
    one = jnp.array(1.0, dtype=jnp.float32)
    metrics = {
        METRIC_SOFT_LOSS: (soft_loss_sum_scaled, valid_count), METRIC_HARD_LOSS: (ce_student_sum, valid_count),
        METRIC_TEACHER_LOSS: (ce_teacher_sum, valid_count), METRIC_STUDENT_PERPLEXITY: (student_perplexity_step, one),
        METRIC_TEACHER_PERPLEXITY: (teacher_perplexity_step, one), METRIC_KL_DIV_AT_T: (kl_softened_sum, valid_count),
        METRIC_KL_DIV_T1: (kl_t1_sum, valid_count), METRIC_OUT_PROJ_FEATURE_LOSS: (feature_loss, one),
        METRIC_MOE_LB_LOSS: (moe_lb_loss, one), METRIC_TEACHER_MOE_LB_LOSS: (teacher_moe_lb_loss, one),
        METRIC_TOTAL_LOSS: (total_loss, one), METRIC_TEMPERATURE: (temperature, one), METRIC_ALPHA: (alpha, one),
        METRIC_BETA_FEATURE: (beta_feature, one),
    }
    return total_loss, metrics

  def compute_eval_loss(self, student_output, labels):
    """Compute the evaluation task loss and metrics."""
    s_logits = student_output.logits.astype(jnp.float32)
    mask = jnp.any(labels != 0, axis=-1).astype(jnp.float32)
    valid_count = jnp.sum(mask)
    safe_count = jnp.maximum(valid_count, 1.0)
    ce_per_pos = optax.softmax_cross_entropy(logits=s_logits, labels=labels)
    ce_sum = jnp.sum(ce_per_pos * mask)
    task_loss = ce_sum / safe_count
    metrics = {
        "eval/hard_loss": (ce_sum, valid_count),
        "eval/student_perplexity": (jnp.exp(jnp.minimum(task_loss, _PPL_CE_CAP)), jnp.array(1.0, dtype=jnp.float32)),
    }
    return task_loss, metrics

  def create_labels(self, targets, targets_segmentation=None, **kwargs):
    """Convert integer targets to masked one-hot labels."""
    del kwargs
    one_hot = jax.nn.one_hot(targets, self.vocab_size)
    mask = jnp.not_equal(targets, self.pad_id).astype(one_hot.dtype)[..., None]
    if targets_segmentation is not None:
      mask = mask * (targets_segmentation != 0)[..., None]
    return one_hot * mask


class MaxTextCheckpointManager(TrainingCheckpointManager):
  """Tunix checkpoint manager with MaxText Grain iterator state."""

  def __init__(self, raw_iterator: Any | None, root_directory: str | None, student_config: Any,
               options: checkpoint.CheckpointManagerOptions | None = None):
    super().__init__(root_directory=root_directory, options=options)
    self.student_config = student_config
    self._iterator = raw_iterator

  def save(self, step, model, optimizer=None, save_only_lora_params=False, force=False, custom_metadata=None):
    """Save model, optimizer, and Grain iterator state."""
    if self._checkpointer is None:
      return False
    if self.save_only_lora_params is not None:
      save_only_lora_params = self.save_only_lora_params
    target_model = getattr(model, "student_model", model)
    if save_only_lora_params:
      params = nnx.state(target_model, nnx.LoRAParam)
    else:
      params = nnx.state(target_model)
    checkpointables: dict[str, Any] = {"model_params": params}
    exclude_opt = self.student_config.learn_to_init_mode
    if optimizer is not None and not exclude_opt:
      checkpointables["optimizer_state"] = nnx.state(optimizer, nnx.optimizer.OptState)
    if self._iterator is not None:
      data_iterator = self._iterator
      if not isinstance(data_iterator, list):
        data_iterator = [data_iterator]
      grain_iters_to_save = []
      process_count_total = jax.process_count() * len(data_iterator)
      for index, data_iter in enumerate(data_iterator):
        process_index = jax.process_index() + index * jax.process_count()
        local_iter = data_iter.local_iterator if hasattr(data_iter, "local_iterator") else data_iter
        grain_iters_to_save.append((local_iter, process_index, process_count_total))
      checkpointables["iter"] = grain_utility.GrainCheckpointable(grain_iters_to_save)
    metadata = dict(custom_metadata or {})
    if self.training_graph is not None:
      metadata["maxtext_easy_training_graph"] = self.training_graph
    return self._save_checkpointables(step, checkpointables, force, metadata)

  def maybe_restore(self, model: Any, optimizer: Any = None, restore_only_lora_params: bool = False, step: int | None = None):
    """Restore the student model and optimizer through Tunix."""
    if self._checkpointer is None:
      if step is not None:
        raise ValueError("A checkpoint directory is required to restore a selected step.")
      return 0, {}
    target_model = getattr(model, "student_model", model)
    selected_step = step
    step, custom_metadata = super().maybe_restore(
        model=target_model, optimizer=optimizer, step=step, restore_only_lora_params=restore_only_lora_params,
    )
    if step == 0 and selected_step is None:
      return 0, {}
    max_logging.log(f"Restored from checkpoint step {step}.")
    return step, dict(custom_metadata or {})

  def restore_iterator(self, step: int | None = None):
    """Restore the iterator in place."""
    if self._checkpointer is None or self._iterator is None:
      return None
    selected_step = step
    if step is None:
      step = self.latest_step()
    if step is None:
      return None
    try:
      data_iter = self._iterator
      local_iter = data_iter.local_iterator if hasattr(data_iter, "local_iterator") else data_iter
      self._checkpointer.load_checkpointables(step, {"iter": grain_utility.GrainCheckpointable(local_iter)})
      return self._iterator
    except Exception as error:
      if selected_step is not None:
        raise
      max_logging.log(f"Warning: Could not restore input pipeline: {error}")
      return None

  def wait_until_finished(self):
    """Wait for pending checkpoint operations."""
    if self._checkpointer is not None:
      self._checkpointer.wait()

  def close(self) -> None:
    """Close the checkpoint manager."""
    super().close()
    if getattr(self, "_checkpointer", None) is not None:
      if hasattr(self._checkpointer, "close"):
        self._checkpointer.close()
