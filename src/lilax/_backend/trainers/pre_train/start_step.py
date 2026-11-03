"""Resume step lookup for native training states."""


def get_first_step(model, state):
  del model
  if hasattr(state, "inner_state"):
    step_val = state.step.get_value() if hasattr(state.step, "get_value") else state.step
    return int(step_val)
  return int(state.optimizer.step.get_value())
