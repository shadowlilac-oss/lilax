"""UniPC multistep ODE solver diffusion noise scheduler."""

from functools import partial
from typing import List, Optional, Tuple, Union

import flax
import jax
import jax.numpy as jnp

from .scheduling_utils import CommonSchedulerState, FlaxSchedulerMixin, FlaxSchedulerOutput, add_noise_common, register_to_config


@flax.struct.dataclass
class UniPCMultistepSchedulerState:
  common: CommonSchedulerState
  sigmas: jnp.ndarray
  alpha_t: jnp.ndarray
  sigma_t: jnp.ndarray
  lambda_t: jnp.ndarray
  init_noise_sigma: float
  timesteps: jnp.ndarray = None
  model_outputs: jnp.ndarray = None
  timestep_list: jnp.ndarray = None
  lower_order_nums: int = 0
  last_sample: Optional[jnp.ndarray] = None
  step_index: Optional[int] = None
  begin_index: Optional[int] = None
  this_order: int = 0

  @classmethod
  def create(cls, common_state: CommonSchedulerState, alpha_t: jnp.ndarray, sigma_t: jnp.ndarray,
             lambda_t: jnp.ndarray, sigmas: jnp.ndarray, init_noise_sigma: jnp.ndarray):
    return cls(common=common_state, alpha_t=alpha_t, sigma_t=sigma_t, lambda_t=lambda_t, sigmas=sigmas,
               init_noise_sigma=init_noise_sigma, lower_order_nums=0, last_sample=None,
               step_index=None, begin_index=None, this_order=0)


@flax.struct.dataclass(frozen=False)
class FlaxUniPCMultistepSchedulerOutput(FlaxSchedulerOutput):
  state: UniPCMultistepSchedulerState


class FlaxUniPCMultistepScheduler(FlaxSchedulerMixin):
  dtype: jnp.dtype

  @property
  def has_state(self) -> bool:
    return True

  @register_to_config
  def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02,
               beta_schedule: str = "linear", trained_betas: Optional[Union[jnp.ndarray, List[float]]] = None,
               solver_order: int = 2, prediction_type: str = "epsilon", thresholding: bool = False,
               dynamic_thresholding_ratio: float = 0.995, sample_max_value: float = 1.0, predict_x0: bool = True,
               solver_type: str = "bh2", lower_order_final: bool = True, disable_corrector: List[int] = [],
               solver_p: Optional[FlaxSchedulerMixin] = None, use_flow_sigmas: Optional[bool] = False,
               flow_shift: Optional[float] = 1.0, timestep_spacing: str = "linspace", steps_offset: int = 0,
               final_sigmas_type: Optional[str] = "zero", rescale_zero_terminal_snr: bool = False,
               dtype: jnp.dtype = jnp.float32):
    self.dtype = dtype
    if self.config.solver_type not in ["bh1", "bh2"]:
      raise NotImplementedError(f"{self.config.solver_type} is not implemented for {self.__class__}")

  def create_state(self, common: Optional[CommonSchedulerState] = None) -> UniPCMultistepSchedulerState:
    if common is None:
      common = CommonSchedulerState.create(self)
    if self.config.get("rescale_zero_terminal_snr", False):
      alphas_cumprod = common.alphas_cumprod.at[-1].set(2**-24)
      common = common.replace(alphas_cumprod=alphas_cumprod)
    alpha_t = jnp.sqrt(common.alphas_cumprod)
    sigma_t = jnp.sqrt(1 - common.alphas_cumprod)
    lambda_t = jnp.log(alpha_t) - jnp.log(sigma_t)
    sigmas = ((1 - common.alphas_cumprod) / common.alphas_cumprod) ** 0.5
    init_noise_sigma = jnp.array(1.0, dtype=self.dtype)
    if self.config.solver_type not in ["bh1", "bh2"]:
      if self.config.solver_type in ["midpoint", "heun", "logrho"]:
        self.config.solver_type = "bh2"
      else:
        raise NotImplementedError(f"{self.config.solver_type} is not implemented for {self.__class__}")
    return UniPCMultistepSchedulerState.create(common_state=common, alpha_t=alpha_t, sigma_t=sigma_t,
                                               lambda_t=lambda_t, sigmas=sigmas, init_noise_sigma=init_noise_sigma)

  def set_begin_index(self, state: UniPCMultistepSchedulerState, begin_index: int = 0) -> UniPCMultistepSchedulerState:
    return state.replace(begin_index=begin_index)

  def set_timesteps(self, state: UniPCMultistepSchedulerState, num_inference_steps: int,
                    shape: Tuple) -> UniPCMultistepSchedulerState:
    last_timestep = self.config.num_train_timesteps
    if self.config.timestep_spacing == "linspace":
      timesteps = jnp.linspace(0, last_timestep - 1, num_inference_steps + 1).round()[::-1][:-1].astype(jnp.int32)
    elif self.config.timestep_spacing == "leading":
      step_ratio = last_timestep // (num_inference_steps + 1)
      timesteps = (jnp.arange(0, num_inference_steps + 1) * step_ratio).round()[::-1][:-1].copy().astype(jnp.int32)
      timesteps += self.config.steps_offset
    elif self.config.timestep_spacing == "trailing":
      step_ratio = self.config.num_train_timesteps / num_inference_steps
      timesteps = jnp.arange(last_timestep, 0, -step_ratio).round().copy().astype(jnp.int32)
      timesteps -= 1
    else:
      raise ValueError(f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of"
                       " 'linspace', 'leading' or 'trailing'.")
    sigmas = state.sigmas
    if self.config.use_flow_sigmas:
      sigmas = jnp.linspace(1.0, 1.0 / self.config.num_train_timesteps, num_inference_steps + 1)[:-1]
      sigmas = self.config.flow_shift * sigmas / (1 + (self.config.flow_shift - 1) * sigmas)
      eps = 1e-6
      sigmas = sigmas.at[0].set(jnp.where(jnp.abs(sigmas[0] - 1.0) < eps, sigmas[0] - eps, sigmas[0]))
      timesteps = (sigmas * self.config.num_train_timesteps).copy().astype(jnp.int32)
      if self.config.final_sigmas_type == "sigma_min":
        sigma_last = sigmas[-1]
      elif self.config.final_sigmas_type == "zero":
        sigma_last = 0
      else:
        raise ValueError(f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.config.final_sigmas_type}")
      sigmas = jnp.concatenate([sigmas, jnp.array([sigma_last])]).astype(jnp.float32)
    else:
      sigmas = jnp.interp(timesteps, jnp.arange(0, len(sigmas)), sigmas)
      if self.config.final_sigmas_type == "sigma_min":
        sigma_last = ((1 - state.common.alphas_cumprod[0]) / state.common.alphas_cumprod[0]) ** 0.5
      elif self.config.final_sigmas_type == "zero":
        sigma_last = 0
      else:
        raise ValueError(f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.config.final_sigmas_type}")
      sigmas = jnp.concatenate([sigmas, jnp.array([sigma_last])]).astype(jnp.float32)
    model_outputs = jnp.zeros((self.config.solver_order,) + shape, dtype=self.dtype)
    timestep_list = jnp.zeros((self.config.solver_order,), dtype=jnp.int32)
    return state.replace(timesteps=timesteps, sigmas=sigmas, model_outputs=model_outputs, timestep_list=timestep_list,
                         lower_order_nums=0, step_index=None, begin_index=None, last_sample=None, this_order=0)

  def convert_model_output(self, state: UniPCMultistepSchedulerState, model_output: jnp.ndarray,
                           sample: jnp.ndarray) -> jnp.ndarray:
    sigma = state.sigmas[state.step_index]
    alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
    if self.config.predict_x0:
      if self.config.prediction_type == "epsilon":
        x0_pred = (sample - sigma_t * model_output) / alpha_t
      elif self.config.prediction_type == "sample":
        x0_pred = model_output
      elif self.config.prediction_type == "v_prediction":
        x0_pred = alpha_t * sample - sigma_t * model_output
      elif self.config.prediction_type == "flow_prediction":
        x0_pred = sample - sigma * model_output
      else:
        raise ValueError(f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, "
                         "`v_prediction`, or `flow_prediction` for the UniPCMultistepScheduler.")
      if self.config.thresholding:
        raise NotImplementedError("Dynamic thresholding isn't implemented.")
      return x0_pred
    if self.config.prediction_type == "epsilon":
      return model_output
    elif self.config.prediction_type == "sample":
      return (sample - alpha_t * model_output) / sigma_t
    elif self.config.prediction_type == "v_prediction":
      return alpha_t * model_output + sigma_t * sample
    raise ValueError(f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                     " `v_prediction` for the UniPCMultistepScheduler.")

  def multistep_uni_p_bh_update(self, state: UniPCMultistepSchedulerState, model_output: jnp.ndarray,
                                sample: jnp.ndarray, order: int) -> jnp.ndarray:
    if self.config.solver_p:
      raise NotImplementedError("Nested `solver_p` is not implemented in JAX version yet.")
    m0 = state.model_outputs[self.config.solver_order - 1]
    x = sample
    sigma_t_val, sigma_s0_val = state.sigmas[state.step_index + 1], state.sigmas[state.step_index]
    alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t_val)
    alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0_val)
    lambda_t = jnp.log(alpha_t + 1e-10) - jnp.log(sigma_t + 1e-10)
    lambda_s0 = jnp.log(alpha_s0 + 1e-10) - jnp.log(sigma_s0 + 1e-10)
    h = lambda_t - lambda_s0

    def rk_d1_loop_body(i, carry):
      rks, D1s = carry
      history_idx = self.config.solver_order - 2 - i
      mi = state.model_outputs[history_idx]
      si_val = state.timestep_list[history_idx]
      alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(state.sigmas[self.index_for_timestep(state, si_val)])
      lambda_si = jnp.log(alpha_si + 1e-10) - jnp.log(sigma_si + 1e-10)
      rk = (lambda_si - lambda_s0) / h
      Di = (mi - m0) / rk
      return rks.at[i].set(rk), D1s.at[i].set(Di)

    rks_init = jnp.zeros(self.config.solver_order, dtype=h.dtype)
    D1s_init = jnp.zeros((self.config.solver_order - 1, *m0.shape), dtype=m0.dtype)
    if self.config.solver_order == 1:
      D1s_init = jnp.zeros((1, *m0.shape), dtype=m0.dtype)
    rks, D1s = jax.lax.fori_loop(0, order - 1, rk_d1_loop_body, (rks_init, D1s_init))
    rks = rks.at[order - 1].set(1.0)
    hh = -h if self.config.predict_x0 else h
    h_phi_1 = jnp.expm1(hh)
    if self.config.solver_type == "bh1":
      B_h = hh
    elif self.config.solver_type == "bh2":
      B_h = jnp.expm1(hh)
    else:
      raise NotImplementedError()

    def rb_loop_body(i, carry):
      R, b, current_h_phi_k, factorial_val = carry
      R = R.at[i].set(jnp.power(rks, i))
      b = b.at[i].set(current_h_phi_k * factorial_val / B_h)

      def update_fn(vals):
        _h_phi_k, _fac = vals
        next_fac = _fac * (i + 2)
        next_h_phi_k = _h_phi_k / hh - 1.0 / next_fac
        return next_h_phi_k, next_fac

      current_h_phi_k, factorial_val = jax.lax.cond(i < order - 1, update_fn, lambda vals: vals,
                                                     (current_h_phi_k, factorial_val))
      return R, b, current_h_phi_k, factorial_val

    R_init = jnp.zeros((self.config.solver_order, self.config.solver_order), dtype=h.dtype)
    b_init = jnp.zeros(self.config.solver_order, dtype=h.dtype)
    R, b, _, _ = jax.lax.fori_loop(0, order, rb_loop_body, (R_init, b_init, h_phi_1 / hh - 1.0, 1.0))
    if len(D1s) > 0:
      D1s = jnp.stack(D1s, axis=1)

    def solve_for_rhos_p(R_mat, b_vec, current_order):
      mask_size = self.config.solver_order - 1
      mask = jnp.arange(mask_size) < (current_order - 1)
      mask_2d = mask[:, None] & mask[None, :]
      R_safe = jnp.where(mask_2d, R_mat[:mask_size, :mask_size], jnp.eye(mask_size, dtype=R_mat.dtype))
      b_safe = jnp.where(mask, b_vec[:mask_size], 0.0)
      solved_rhos = jnp.linalg.solve(R_safe, b_safe)
      return jnp.where(mask, solved_rhos, 0.0)

    if self.config.solver_order == 1:
      rhos_p_order2 = jnp.zeros(1, dtype=x.dtype)
    else:
      rhos_p_order2 = jnp.zeros(self.config.solver_order - 1, dtype=x.dtype).at[0].set(0.5)
    rhos_p_general = solve_for_rhos_p(R, b, order)
    rhos_p = jnp.where(order == 2, rhos_p_order2, rhos_p_general)
    pred_res = jax.lax.cond(order > 1, lambda _: jnp.einsum("k,bkc...->bc...", rhos_p, D1s).astype(x.dtype),
                            lambda _: jnp.zeros_like(x), operand=None)
    if self.config.predict_x0:
      x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
      x_t = x_t_ - alpha_t * B_h * pred_res
    else:
      x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
      x_t = x_t_ - sigma_t * B_h * pred_res
    return x_t.astype(x.dtype)

  def index_for_timestep(self, state: UniPCMultistepSchedulerState, timestep: Union[int, jnp.ndarray],
                         schedule_timesteps: Optional[jnp.ndarray] = None) -> int:
    if schedule_timesteps is None:
      schedule_timesteps = state.timesteps
    index_candidates = jnp.where(schedule_timesteps == timestep, size=1, fill_value=-1)[0]
    return jnp.where(index_candidates[0] == -1, len(schedule_timesteps) - 1, index_candidates[0])

  def _init_step_index(self, state: UniPCMultistepSchedulerState,
                       timestep: Union[int, jnp.ndarray]) -> UniPCMultistepSchedulerState:
    if state.begin_index is None:
      step_index_val = self.index_for_timestep(state, timestep)
      return state.replace(step_index=step_index_val)
    return state.replace(step_index=state.begin_index)

  @partial(jax.jit, static_argnums=(0, 5))
  def step(self, state: UniPCMultistepSchedulerState, model_output: jnp.ndarray, timestep: Union[int, jnp.ndarray],
           sample: jnp.ndarray, return_dict: bool = True, generator: Optional[jax.random.PRNGKey] = None
           ) -> Union[FlaxUniPCMultistepSchedulerOutput, Tuple[jnp.ndarray]]:
    sample = sample.astype(self.dtype)
    if state.timesteps is None:
      raise ValueError("Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler")
    timestep_scalar = jnp.array(timestep)
    if state.step_index is None:
      state = self._init_step_index(state, timestep_scalar)
    use_corrector = ((state.step_index > 0)
                     & (~jnp.isin(state.step_index - 1, jnp.array(self.config.disable_corrector)))
                     & (state.last_sample is not None))
    model_output_for_history = self.convert_model_output(state, model_output, sample)
    model_output_for_history = model_output_for_history.astype(self.dtype)
    sample = jax.lax.cond(
        use_corrector,
        lambda: self.multistep_uni_c_bh_update(state=state, this_model_output=model_output_for_history,
                                                last_sample=state.last_sample, this_sample=sample, order=state.this_order),
        lambda: sample,
    )

    def step_idx0_branch():
      updated_model_outputs_history = state.model_outputs.at[-1].set(model_output_for_history)
      updated_timestep_list_history = state.timestep_list.at[-1].set(timestep_scalar)
      return updated_model_outputs_history, updated_timestep_list_history

    def non_step_idx0_branch():
      updated_model_outputs_history = jnp.roll(state.model_outputs, shift=-1, axis=0)
      updated_model_outputs_history = updated_model_outputs_history.at[-1].set(model_output_for_history)
      updated_timestep_list_history = jnp.roll(state.timestep_list, shift=-1)
      updated_timestep_list_history = updated_timestep_list_history.at[-1].set(timestep_scalar)
      return updated_model_outputs_history, updated_timestep_list_history

    updated_model_outputs_history, updated_timestep_list_history = jax.lax.cond(
        state.step_index == 0, step_idx0_branch, non_step_idx0_branch
    )
    state = state.replace(model_outputs=updated_model_outputs_history, timestep_list=updated_timestep_list_history)
    this_order = jnp.where(self.config.lower_order_final,
                           jnp.minimum(self.config.solver_order, len(state.timesteps) - state.step_index),
                           self.config.solver_order)
    new_this_order = jnp.minimum(this_order, state.lower_order_nums + 1)
    state = state.replace(this_order=new_this_order)
    state = state.replace(last_sample=sample)
    prev_sample = self.multistep_uni_p_bh_update(state=state, model_output=model_output, sample=sample,
                                                 order=state.this_order)
    new_lower_order_nums = jnp.where(state.lower_order_nums < self.config.solver_order,
                                     state.lower_order_nums + 1, state.lower_order_nums)
    state = state.replace(lower_order_nums=new_lower_order_nums)
    state = state.replace(step_index=state.step_index + 1)
    if not return_dict:
      return (prev_sample, state)
    return FlaxUniPCMultistepSchedulerOutput(prev_sample=prev_sample, state=state)

  def scale_model_input(self, state: UniPCMultistepSchedulerState, sample: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
    return sample

  def add_noise(self, state: UniPCMultistepSchedulerState, original_samples: jnp.ndarray, noise: jnp.ndarray,
                timesteps: jnp.ndarray) -> jnp.ndarray:
    return add_noise_common(state.common, original_samples, noise, timesteps)

  def _sigma_to_alpha_sigma_t(self, sigma):
    if self.config.use_flow_sigmas:
      alpha_t = 1 - sigma
      sigma_t = sigma
    else:
      alpha_t = 1 / ((sigma**2 + 1) ** 0.5)
      sigma_t = sigma * alpha_t
    return alpha_t, sigma_t

  def __len__(self) -> int:
    return self.config.num_train_timesteps

  def multistep_uni_c_bh_update(self, state: UniPCMultistepSchedulerState, this_model_output: jnp.ndarray,
                                last_sample: jnp.ndarray, this_sample: jnp.ndarray, order: int) -> jnp.ndarray:
    model_output_list = state.model_outputs
    m0 = model_output_list[self.config.solver_order - 1]
    x = last_sample if last_sample is not None else jnp.zeros_like(this_sample)
    x_t = this_sample
    model_t = this_model_output
    sigma_t_val = state.sigmas[state.step_index]
    sigma_s0_val = state.sigmas[state.step_index - 1]
    alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t_val)
    alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0_val)
    lambda_t = jnp.log(alpha_t + 1e-10) - jnp.log(sigma_t + 1e-10)
    lambda_s0 = jnp.log(alpha_s0 + 1e-10) - jnp.log(sigma_s0 + 1e-10)
    h = lambda_t - lambda_s0

    def rk_d1_loop_body(i, carry):
      rks, D1s = carry
      history_idx = self.config.solver_order - (i + 2)
      mi = state.model_outputs[history_idx]
      si_val = state.timestep_list[history_idx]
      alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(state.sigmas[self.index_for_timestep(state, si_val)])
      lambda_si = jnp.log(alpha_si + 1e-10) - jnp.log(sigma_si + 1e-10)
      rk = (lambda_si - lambda_s0) / h
      Di = (mi - m0) / rk
      return rks.at[i].set(rk), D1s.at[i].set(Di)

    rks_init = jnp.zeros(self.config.solver_order, dtype=h.dtype)
    D1s_init = jnp.zeros((self.config.solver_order - 1, *m0.shape), dtype=m0.dtype)
    if self.config.solver_order == 1:
      D1s_init = jnp.zeros((1, *m0.shape), dtype=m0.dtype)
    rks, D1s = jax.lax.fori_loop(0, order - 1, rk_d1_loop_body, (rks_init, D1s_init))
    rks = rks.at[order - 1].set(1.0)
    hh = -h if self.config.predict_x0 else h
    h_phi_1 = jnp.expm1(hh)
    if self.config.solver_type == "bh1":
      B_h = hh
    elif self.config.solver_type == "bh2":
      B_h = jnp.expm1(hh)
    else:
      raise NotImplementedError()

    def rb_loop_body(i, carry):
      R, b, current_h_phi_k, factorial_val = carry
      R = R.at[i].set(jnp.power(rks, i))
      b = b.at[i].set(current_h_phi_k * factorial_val / B_h)

      def update_fn(vals):
        _h_phi_k, _fac = vals
        next_fac = _fac * (i + 2)
        next_h_phi_k = _h_phi_k / hh - 1.0 / next_fac
        return next_h_phi_k, next_fac

      current_h_phi_k, factorial_val = jax.lax.cond(i < order - 1, update_fn, lambda vals: vals,
                                                     (current_h_phi_k, factorial_val))
      return R, b, current_h_phi_k, factorial_val

    R_init = jnp.zeros((self.config.solver_order, self.config.solver_order), dtype=h.dtype)
    b_init = jnp.zeros(self.config.solver_order, dtype=h.dtype)
    R, b, _, _ = jax.lax.fori_loop(0, order, rb_loop_body, (R_init, b_init, h_phi_1 / hh - 1.0, 1.0))
    if len(D1s) > 0:
      D1s = jnp.stack(D1s, axis=1)

    def solve_for_rhos(R_mat, b_vec, current_order):
      mask = jnp.arange(self.config.solver_order) < current_order
      mask_2d = mask[:, None] & mask[None, :]
      R_safe = jnp.where(mask_2d, R_mat, jnp.eye(self.config.solver_order, dtype=R_mat.dtype))
      b_safe = jnp.where(mask, b_vec, 0.0)
      solved_rhos = jnp.linalg.solve(R_safe, b_safe)
      return jnp.where(mask, solved_rhos, 0.0)

    rhos_c_order1 = jnp.zeros(self.config.solver_order, dtype=x_t.dtype).at[0].set(0.5)
    rhos_c_general = solve_for_rhos(R, b, order)
    rhos_c = jnp.where(order == 1, rhos_c_order1, rhos_c_general)
    D1_t = model_t - m0
    corr_res = jax.lax.cond(order > 1, lambda _: jnp.einsum("k,bkc...->bc...", rhos_c[:-1], D1s),
                            lambda _: jnp.zeros_like(D1_t), operand=None)
    final_rho = jnp.dot(rhos_c, jax.nn.one_hot(order - 1, self.config.solver_order, dtype=rhos_c.dtype))
    if self.config.predict_x0:
      x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
      x_t = x_t_ - alpha_t * B_h * (corr_res + final_rho * D1_t)
    else:
      x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
      x_t = x_t_ - sigma_t * B_h * (corr_res + final_rho * D1_t)
    return x_t.astype(x.dtype)
