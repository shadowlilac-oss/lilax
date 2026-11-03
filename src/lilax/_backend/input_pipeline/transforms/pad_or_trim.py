"""Padding or trimming of token and image features to fixed lengths."""

import dataclasses

import grain.python as grain
import numpy as np

from lilax._backend.multimodal import processor as mm_processor
from lilax._backend.multimodal import utils as mm_utils


@dataclasses.dataclass
class PadOrTrimToMaxLength(grain.MapTransform):
  """Pads or trims each input to the specified length.
  And optionally add true length for the input."""

  def __init__(
      self,
      max_length: int,
      pad_id: int = 0,
      config=None,
      add_true_length: bool = False,
      max_num_images_per_example: int = -1,
  ):
    self.max_length = max_length
    self.pad_id = pad_id
    self.config = config
    self.add_true_length = add_true_length
    self.max_num_images_per_example = max_num_images_per_example

  def _pad_text(self, x: np.ndarray, max_length: int, pad_id: int) -> np.ndarray:
    pad_amount = max(max_length - x.shape[0], 0)
    pad_amount = [(0, pad_amount)] + [(0, 0)] * (len(x.shape) - 1)
    return np.pad(x, pad_amount, constant_values=pad_id)[: self.max_length]

  def _pad_image_and_mask(self, preprocessed_image: mm_utils.PreprocessorOutput) -> mm_utils.PreprocessorOutput:
    """Pads the input tensors (image and mask) of a PreprocessorOutput to a maximum number of items.

    This function unifies padding logic for image tensors (standard or tiled) and
    mask tensors. It determines the tensor type based on its dimensions and applies
    the appropriate padding along the first axis.

    The maximum number of items is calculated based on model constraints or a
    user-defined limit, ensuring that sequence length limits are respected while
    reserving space for at least one text token. If the input tensor has fewer
    items than this maximum, it is padded with zeros.

    Args:
        preprocessed_image (multimodal_utils.PreprocessorOutput): The input numpy arrays to pad.
            - For masks, the expected shape is (num_masks, num_tiles).
            - For standard images, the shape is (num_images, H, W, C).
            - For tiled images, the shape is (num_images, num_tiles, H, W, C).

    Returns:
        np.ndarray: The tensor, padded with zeros up to the maximum number of
        items along the first axis.

    Raises:
        ValueError: If the input tensor's dimension is not 2, 4, or 5.
        ValueError: If the number of items in the input tensor exceeds the
        allowed maximum.

    Notes:
      - The computation of maximum images ensures that space is reserved in the sequence
        for at least one text token.
      - The dummy images used for padding are based on the image shape for initialization
        of this model (ignoring batch size).
    """
    if not isinstance(preprocessed_image, mm_utils.PreprocessorOutput):
      raise TypeError(f"Input must be multimodal_utils.PreprocessorOutput, but got {type(preprocessed_image)}")

    if getattr(preprocessed_image, "video_values", None) is not None:
      return preprocessed_image

    if preprocessed_image.pixel_values is None:
      raise ValueError("Input preprocessed_image must have pixel_values to pad images.")

    vision_block = mm_processor._get_vision_block(self.config)
    model_name = getattr(self.config, "model_name", None)
    if (
        vision_block is None
        or not getattr(self.config, "use_multimodal", False)
        or vision_block in ["qwen3_omni", "qwen3_vl", "qwen3_5"]
    ):
      return preprocessed_image
    elif model_name and model_name.startswith("qwen"):
      raise ValueError(
          f"Qwen multimodal model '{model_name}' with vision_block '{vision_block}' was not "
          f"registered in `PadOrTrimToMaxLength`. Qwen models use native dynamic grids and must be "
          f"registered to bypass image padding."
      )

    image_offsets = mm_processor.get_image_offsets(self.config, preprocessed_image)
    single_image_offset = image_offsets // preprocessed_image.pixel_values.shape[0]
    max_num_items = (self.max_length - 1) // single_image_offset
    if self.max_num_images_per_example > 0:
      max_num_items = min(self.max_num_images_per_example, max_num_items)

    image_tensor = preprocessed_image.pixel_values
    mask_tensor = preprocessed_image.pixel_mask

    def _pad(tensor: np.ndarray) -> np.ndarray:
      if tensor.ndim in (4, 5):
        tensor_type = "images"
      elif tensor.ndim == 2:
        tensor_type = "masks"
      else:
        raise ValueError(
            "Input tensor must be 2D (mask), 4D (image), or 5D (tiled image), " f"but got {tensor.ndim} dimensions."
        )
      if tensor.shape[0] > max_num_items:
        raise ValueError(f"Number of {tensor_type} ({tensor.shape[0]}) exceeds the maximum allowed ({max_num_items}).")
      if tensor.shape[0] < max_num_items:
        pad_size = max_num_items - tensor.shape[0]
        pad_shape_suffix = tensor.shape[1:]
        pad_shape = (pad_size,) + pad_shape_suffix
        pad_tensor = np.zeros(pad_shape, dtype=tensor.dtype)
        if tensor.size > 0:
          tensor = np.concatenate([tensor, pad_tensor], axis=0)
        else:
          tensor = pad_tensor
      return tensor

    preprocessed_image.pixel_values = _pad(image_tensor)
    if mask_tensor is not None:
      preprocessed_image.pixel_mask = _pad(mask_tensor)
    return preprocessed_image

  def map(
      self, element: dict[str, np.ndarray | mm_utils.PreprocessorOutput]
  ) -> dict[str, np.ndarray | mm_utils.PreprocessorOutput]:
    """map to each element"""
    data_columns = list(element.keys())
    preserve_pad_valued_tokens = (
        self.config is not None and getattr(self.config, "training_objective", "causal_lm") == "block_diffusion"
    )
    for data_column in data_columns:
      if data_column not in ("images", "completion_mask"):
        if isinstance(element[data_column], mm_utils.PreprocessorOutput):
          raise TypeError("Only 'images' column can be of type PreprocessorOutput.")
        if preserve_pad_valued_tokens:
          element[f"{data_column}_segmentation"] = np.ones(element[data_column].shape[0], dtype=np.int32)
        else:
          element[f"{data_column}_segmentation"] = element[data_column] != self.pad_id
        element[f"{data_column}_segmentation"] = element[f"{data_column}_segmentation"].astype(np.int32)
        element[f"{data_column}_position"] = np.arange(element[data_column].shape[0], dtype=np.int32)
        if self.add_true_length:
          element[f"{data_column}_true_length"] = np.array([element[data_column].shape[0]], dtype=np.int32)

    for key, _ in element.items():
      if key == "images":
        if self.config.model_name is None:
          raise ValueError("model_name must be provided when padding images")
        element["images"] = self._pad_image_and_mask(element["images"])
      elif key == "completion_mask" or (preserve_pad_valued_tokens and key.endswith(("_segmentation", "_position"))):
        element[key] = self._pad_text(element[key], self.max_length, 0)
      elif "true_length" not in key:
        element[key] = self._pad_text(element[key], self.max_length, self.pad_id)
    return element
