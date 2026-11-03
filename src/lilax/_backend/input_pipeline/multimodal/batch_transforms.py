"""Extraction and batch folding of preprocessed multimodal image and video tensors."""

import dataclasses

import grain.python as grain
import numpy as np

from lilax._backend.multimodal import processor as mm_processor
from lilax._backend.multimodal import utils as mm_utils


@dataclasses.dataclass
class ExtractImagesAndMasks(grain.MapTransform):
  """Extracts images and masks from a PreprocessorOutput object.

  This transform is used in multi-modal data pipelines to extract the image
  tensors and their corresponding masks from a PreprocessorOutput object.
  The extracted images and masks are then added to the data element under
  the keys 'images' and 'image_masks', respectively.

  If the 'images' key is not present in the input element, the transform
  returns the element unchanged.
  """

  def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Applies the extraction transformation to the 'images' field if present."""
    preprocessed_image = element.get("images")
    if preprocessed_image is None:
      return element
    if not isinstance(preprocessed_image, mm_utils.PreprocessorOutput):
      raise TypeError(f"'images' must be of type PreprocessorOutput, but got {type(preprocessed_image)}")
    output = element.copy()
    if getattr(preprocessed_image, "video_values", None) is not None:
      output["images"] = preprocessed_image.video_values
      if getattr(preprocessed_image, "video_mask", None) is not None:
        output["image_masks"] = preprocessed_image.video_mask
      if getattr(preprocessed_image, "video_grid_thw", None) is not None:
        output["video_grid_thw"] = preprocessed_image.video_grid_thw
      if getattr(preprocessed_image, "video_second_per_grid", None) is not None:
        output["second_per_grids"] = preprocessed_image.video_second_per_grid
      return output
    output["images"] = preprocessed_image.pixel_values
    if preprocessed_image.pixel_mask is not None:
      output["image_masks"] = preprocessed_image.pixel_mask
    pixel_grid_thw = getattr(preprocessed_image, "pixel_grid_thw", None)
    if pixel_grid_thw is not None:
      output["image_grid_thw"] = pixel_grid_thw
    return output


@dataclasses.dataclass
class FoldImagesIntoBatch(grain.MapTransform):
  """Folds the 'image' dimension into the batch dimension.

  This transform is used in multi-modal data pipelines where each data example
  might have multiple associated images. For model processing, it's often
  efficient to treat each image as a separate item in a larger batch.

  This operation reshapes the 'images' tensor from a shape like
  (B, N, T, H, W, C) to (B * N, T, H, W, C), where B is the batch size, N is
  the number of images per example, and T is the number of image tiles.

  The transformation is triggered only if the input 'images' tensor has more
  dimensions than the expected batched image tensor.
  """

  model_name: str | None = None

  def __post_init__(self):
    """Initializes the target shape after the dataclass is created."""
    self.target_shape = mm_processor.get_dummy_image_shape_for_init(self.model_name)

  def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Applies the folding transformation to the 'images' field if present."""
    images = element.get("images")
    if images is None:
      return element
    video_grid_thw = element.get("video_grid_thw")
    if video_grid_thw is not None and images.ndim > len(self.target_shape):
      element["images"] = images.reshape(-1, *images.shape[2:])
      image_masks = element.get("image_masks")
      if image_masks is not None:
        element["image_masks"] = image_masks.reshape(-1, *image_masks.shape[2:])
      element["video_grid_thw"] = video_grid_thw.reshape(-1, video_grid_thw.shape[-1])
      second_per_grids = element.get("second_per_grids")
      if second_per_grids is not None:
        element["second_per_grids"] = second_per_grids.reshape(-1)
      return element
    if images.ndim > len(self.target_shape):
      trailing_dims = self.target_shape[1:]
      element["images"] = images.reshape(-1, *trailing_dims)
    return element
