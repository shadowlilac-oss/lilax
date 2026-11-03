"""Utilities for the learn-to-init phase of distillation."""

import re

from flax import nnx

from lilax._backend.utils import max_logging


def _nav_to_attr(root, parts):
  obj = root
  for part in parts:
    if hasattr(obj, part):
      obj = getattr(obj, part)
      continue
    try:
      obj = obj[part]
      continue
    except (KeyError, TypeError, IndexError):
      pass
    try:
      obj = obj[int(part)]
      continue
    except (ValueError, TypeError, KeyError, IndexError):
      pass
    raise AttributeError(
        f"prepare_student_weights: cannot resolve path component {part!r} on "
        f"{type(obj).__name__}. Make sure copy/share map paths match "
        f"nnx.graph.iter_graph output for this model."
    )
  return obj


def prepare_student_weights(
    student_model: nnx.Module,
    teacher_model: nnx.Module,
    teacher_weights_copy_map: dict[str, str],
    student_weights_share_map: dict[str, str],
):
  """Copy teacher weights and share student weights in place."""
  max_logging.log("Starting teacher weight injection...")
  teacher_graph = {"/".join(map(str, path)): node for path, node in nnx.graph.iter_graph(teacher_model)}
  student_graph = {"/".join(map(str, path)): node for path, node in nnx.graph.iter_graph(student_model)}

  for source_pattern, dest_pattern in student_weights_share_map.items():
    matched_any = False
    for source_path, source_node in student_graph.items():
      match = re.fullmatch(source_pattern, source_path)
      if match:
        matched_any = True
        dest_path = source_path if source_pattern == dest_pattern else match.expand(dest_pattern)
        dest_node = student_graph.get(dest_path)
        assert dest_node is not None, f"Student parameter sharing: Could not find dest_node model parameter at path: {dest_path}"
        assert source_node.get_value().shape == dest_node.get_value().shape, (
            f"Shape mismatch for sharing parameter between {source_path} and {dest_path}: "
            f"{source_node.get_value().shape} vs {dest_node.get_value().shape}"
        )
        max_logging.info(f"Sharing parameter {source_path} with {dest_path}")
        dest_parts = dest_path.split("/")
        dest_parent = _nav_to_attr(student_model, dest_parts[:-1])
        dest_attr = dest_parts[-1]
        if hasattr(dest_parent, dest_attr):
          setattr(dest_parent, dest_attr, source_node)
        else:
          dest_parent[dest_attr] = source_node
    if not matched_any:
      raise ValueError(f"Student parameter sharing: No paths matched the source pattern: {source_pattern}")

  for teacher_pattern, student_pattern in teacher_weights_copy_map.items():
    matched_any = False
    for teacher_path, teacher_node in teacher_graph.items():
      match = re.fullmatch(teacher_pattern, teacher_path)
      if match:
        matched_any = True
        student_path = teacher_path if teacher_pattern == student_pattern else match.expand(student_pattern)
        student_node = _nav_to_attr(student_model, student_path.split("/"))
        assert student_node is not None, (
            f"Could not find student model parameter at path: {student_path} for teacher parameter {teacher_path}"
        )
        assert student_node.get_value().shape == teacher_node.get_value().shape, (
            f"Shape mismatch for {teacher_path}. Teacher: {teacher_node.get_value().shape},"
            f" Student: {student_node.get_value().shape}"
        )
        student_node.set_value(teacher_node.get_value())
        max_logging.info(f"Inserted teacher weight parameter {teacher_path} to the student at {student_path}")
    if not matched_any:
      raise ValueError(f"Teacher weight injection: No paths matched the source pattern: {teacher_pattern}")

  max_logging.info("Teacher weight injection complete.")
