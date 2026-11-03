"""Native TensorBoard writer operations."""

from lilax._backend.common.vertex_tensorboard import VertexTensorboardManager
from lilax._backend.utils.max_utils import (
    add_text_to_summary_writer,
    close_summary_writer,
    initialize_summary_writer,
)
