import types

import torch
from torch.optim.optimizer import Optimizer


def create_preconditioner(x, dim):
  return torch.zeros((x.shape[dim], x.shape[dim]), dtype=x.dtype, device=x.device)


class _LoraRiteHelper:
  def __init__(self, maybe_inf_to_nan: bool = True):
    self._maybe_inf_to_nan = maybe_inf_to_nan

  def inf_to_nan(self, array):
    """Converting Infinity values to the more sticky NaN."""
    if not self._maybe_inf_to_nan:
      return array
    return torch.nan_to_num(array, nan=torch.nan, posinf=torch.nan, neginf=torch.nan)

  def bias_corrected_decay(self, step, decay: float):
    """Incorporates bias correction into decay.

    Please see section 7.1 in https://arxiv.org/pdf/1804.04235.pdf for the
    derivation of the formulas below. With bias-corrected decay, we can simply
    do

    m_{t} = decay1 * m_{t-1} + (1 - decay1) * g
    v_{t} = decay2 * v_{t-1} + (1 - decay2) * g ^ 2

    without further bias correction.

    Args:
      step: current step, 0-based.
      decay: the raw decay. As t -> infinity, bias corrected decay converges to
        this value.

    Returns:
      Bias corrected decay.
    """
    t = step + 1.
    return decay * (1. - decay**(t - 1.)) / (1. - decay**t)

  def move_lora_dim_to_last(self, x, dim):
    x = torch.moveaxis(x, dim, -1)
    return x.reshape(-1, x.shape[-1]), x.shape

  def restore_original_shape_and_dim(self, x, dim, shape):
    x = x.reshape(shape)
    return torch.moveaxis(x, -1, dim)

  def restore_param_shape(self, x, p, dim):
    _, shape = self.move_lora_dim_to_last(p, dim)
    return self.restore_original_shape_and_dim(x, dim, shape)

  def change_major_dim(self, x, dim):
    x = torch.moveaxis(x, dim, 0)
    return x.reshape(x.shape[0], -1), x.shape

  def reverse_change_major_dim(self, x, dim, shape):
    x = x.reshape(shape)
    return torch.moveaxis(x, 0, dim)

  def inverse_sqrt(self, x, esc, epsilon, epsilon_root, relative_epsilon=True, force_positive=False):
    if relative_epsilon:
      eps = torch.max(torch.linalg.eigvalsh(p_new))*epsilon_root
    else:
      eps = epsilon_root

    w, v = torch.linalg.eigh(x)
    if force_positive:
      w = torch.maximum(w, torch.zeros_like(w))

    w = 1 / (torch.sqrt(w + esc) + epsilon)
    w = torch.unsqueeze(w, 0)
    return self.make_symmetric(v * w @ v.T).to(x.dtype)

  def make_symmetric(self, x):
    return (x + x.T) / 2

  def transform_first_moment_to_new_basis(
      self, m, dual_basis_old, dual_basis, dim, dim_other
  ):
    m, shape = self.move_lora_dim_to_last(m, dim)
    dual_basis_old, _ = self.move_lora_dim_to_last(dual_basis_old, dim_other)
    dual_basis, _ = self.move_lora_dim_to_last(dual_basis, dim_other)

    p = dual_basis.T @ dual_basis_old
    m = m @ p.T
    return self.restore_original_shape_and_dim(m, dim, shape)

  def transform_second_moment_to_new_basis(
      self, v, dual_basis_old, dual_basis, dim, dim_other
  ):
    dual_basis_old, _ = self.move_lora_dim_to_last(dual_basis_old, dim_other)
    dual_basis, _ = self.move_lora_dim_to_last(dual_basis, dim_other)

    p = dual_basis.T @ dual_basis_old

    v_new = p @ v @ p.T
    v_new = self.make_symmetric(v_new)
    trace_old = torch.trace(v)
    trace_new = torch.trace(v_new)
    v_new = torch.nan_to_num(v_new)
    return v_new

  def get_unmagnified_rotate_second_escape(
      self, v_new, v_old
  ):
    eig_old = torch.linalg.eigvalsh(v_old)
    eig_new = torch.linalg.eigvalsh(v_new)
    eigen_diff = torch.maximum(torch.max(eig_old-eig_new), torch.tensor(0).to(eig_old))
    trace_diff = torch.maximum(torch.trace(v_old)-torch.trace(v_new), torch.tensor(0).to(eig_old))
    escape_mass = torch.minimum(eigen_diff, trace_diff)
    return escape_mass

  def get_unmagnified_grad(
      self, g, ri, dim
  ):
    u, shape = self.move_lora_dim_to_last(g, dim)
    u = u@ri
    return self.restore_original_shape_and_dim(u, dim, shape)

  def rotate_update(self, g, ri, dim):
    u, shape = self.move_lora_dim_to_last(g, dim)
    u = u@ri.T
    return self.restore_original_shape_and_dim(u, dim, shape)

  def get_preconditioned_update(
      self,
      g,
      p,
      e,
      epsilon,
      epsilon_root,
      relative_epsilon,
      apply_escape,
      dim,
  ):
    u, shape = self.move_lora_dim_to_last(g, dim)
    q = self.make_symmetric(p)

    if not apply_escape:
      e = 0

    use_combined_epsilon = True

    qir = self.make_symmetric(
      self.inverse_sqrt(q, e, epsilon, epsilon_root, relative_epsilon)
    )
    u = u@qir
    u = torch.nan_to_num(u)
    return self.restore_original_shape_and_dim(u, dim, shape)

  def update_moment(self, step, update,
                     m, beta: float):
    beta_decay = self.bias_corrected_decay(step, beta)
    m = (1.0 - beta_decay) * update + beta_decay * m
    return m

  def update_moments(self, step, update,
                     m, v, beta1: float,
                     beta2: float):
    """Updates momentum values."""
    beta1_decay = self.bias_corrected_decay(step, beta1)
    beta2_decay = self.bias_corrected_decay(step, beta2)
    m = (1.0 - beta1_decay) * update + beta1_decay * m
    v = (1.0 - beta2_decay) * (update**2) + beta2_decay * v
    return m, v

  def update_first_moments(
      self, step, update, moments, beta1: float
  ):
    """Updates momentum values."""
    beta1_decay = self.bias_corrected_decay(step, beta1)
    s = (1.0 - beta1_decay) * update + beta1_decay * moments
    return s

  def compute_second_moments(
      self, update, dim: int
  ):
    update, _ = self.move_lora_dim_to_last(update, dim)
    s = (update.T @ update) / update.shape[0]
    s = self.make_symmetric(s)
    return s

  def update_second_moments(
      self, step, update, moments, beta2: float
  ):
    """Updates momentum values."""
    beta2_decay = self.bias_corrected_decay(step, beta2)
    s = (1.0 - beta2_decay) * update + beta2_decay * moments
    s = self.make_symmetric(s)
    return s

  def update_second_escape(
      self, step, update, moments, beta2: float,
  ):
    beta2_decay = self.bias_corrected_decay(step, beta2)
    return (1.0 - beta2_decay) * update + beta2_decay * moments

  def polar(self, m):   # express polar decomposition in terms of singular-value decomposition
    U, S, Vh = torch.linalg.svd(m, full_matrices=False)
    u = U @ Vh
    p = Vh.T @ S.diag() @ Vh
    p = self.make_symmetric(p)
    return  u.to(m.dtype), p.to(m.dtype)

  def get_rotation_and_basis(self, w, dim):
    w, shape = self.move_lora_dim_to_last(w, dim)

    decomposition = torch.linalg.qr(w)

    r = decomposition[1]
    u = self.restore_original_shape_and_dim(decomposition[0], dim, shape)
    return u, r

  def reduce_rms(self, x):
    return torch.sqrt(torch.mean(torch.pow(x, 2)))

  def clip_update(self, update, clip_threshold: float):
    mean_update = self.inf_to_nan(self.reduce_rms(update))
    denom = torch.maximum(torch.ones_like(mean_update), mean_update / clip_threshold)
    return update / denom

  def skip_update(self, update, skip_threshold: float):
    mean_update = self.inf_to_nan(self.reduce_rms(update))

    if mean_update > skip_threshold:
      update = torch.zeros_like(update)
    return update


class LoRARite(Optimizer):
  def __init__(
      self,
      params,
      betas,
      eps=1e-6,
      lr=1e-3,
      relative_epsilon: bool = False,
      clip_unmagnified_grad: float = 1.0,
      update_capping: float = 0.0,
      update_skipping: float = 1.0,
      weight_decay: float = 0.0,
      apply_escape: float = False,
      lora_l_dim: int = 0,
      lora_r_dim: int = -1,
      maybe_inf_to_nan: bool = True,
      balance_param: bool = False,
  ):
    print("LoRARite init")
    beta1, beta2 = betas
    epsilon = eps
    epsilon_root = eps**2
    if lr < 0.0:
      raise ValueError(
          "Invalid learning rate: {} - should be >= 0.0".format(lr)
      )
    if not 0.0 <= beta1 < 1.0:
      raise ValueError(
          "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(
              beta1
          )
      )
    if not 0.0 <= beta2 < 1.0:
      raise ValueError(
          "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(
              beta2
          )
      )
    if not 0.0 <= epsilon:
      raise ValueError(
          "Invalid epsilon value: {} - should be >= 0.0".format(epsilon)
      )
    defaults = dict(
        lr=lr,
        weight_decay=weight_decay,
    )
    super().__init__(params, defaults)

    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.epsilon_root = epsilon_root
    self.relative_epsilon = relative_epsilon
    self.apply_escape = apply_escape
    self.clip_unmagnified_grad = clip_unmagnified_grad
    self.update_capping = update_capping
    self.update_skipping = update_skipping
    self.weight_decay = weight_decay
    self.lora_l_dim = lora_l_dim
    self.lora_r_dim = lora_r_dim
    self.maybe_inf_to_nan = maybe_inf_to_nan
    self.balance_param = balance_param
    self.helper = _LoraRiteHelper()
    self.state_initialized = False
    self.count = 0

  def step(self, closure=None):
    helper = self.helper
    update_capping = self.update_capping
    update_skipping = self.update_skipping
    clip_unmagnified_grad = self.clip_unmagnified_grad
    weight_decay = self.weight_decay
    lora_l_dim = self.lora_l_dim
    lora_r_dim = self.lora_r_dim
    maybe_inf_to_nan = self.maybe_inf_to_nan
    balance_param = self.balance_param

    beta1 = self.beta1
    beta2 = self.beta2
    epsilon = self.epsilon
    epsilon_root = self.epsilon_root
    relative_epsilon = self.relative_epsilon
    apply_escape = self.apply_escape

    loss = None
    if closure is not None:
      loss = closure()

    self.count += 1

    if not self.state_initialized:
      self.state_initialized = True
      for group in self.param_groups:
        # TODO: make this more generic
        for p1, p2 in list(zip(group["params"], group["params"][1:]))[::2]:
          param_l, param_r = p1.data, p2.data

          self.state[p1]["attr"] = types.SimpleNamespace()
          state = self.state[p1]["attr"]
          state.step = 0
          state.v_l = create_preconditioner(param_l, lora_l_dim)
          state.v_r = create_preconditioner(param_r, lora_r_dim)
          state.m_l = torch.zeros_like(param_l)
          state.m_r = torch.zeros_like(param_r)
          state.basis_l_old = torch.zeros_like(param_l)
          state.basis_r_old = torch.zeros_like(param_r)
          state.escape_l = 0.0
          state.escape_r = 0.0

    g_norm = 0
    g_norm_sq = 0
    for group in self.param_groups:
      # TODO: make this more generic
      for p1, p2 in list(zip(group["params"], group["params"][1:]))[::2]:
        param_l, param_r = p1.data, p2.data

        state = self.state[p1]["attr"]

        update_l, update_r = helper.inf_to_nan(p1.grad.data), helper.inf_to_nan(p2.grad.data)

        decompose_l = helper.get_rotation_and_basis(param_l, lora_l_dim)
        decompose_r = helper.get_rotation_and_basis(param_r, lora_r_dim)

        basis_l = decompose_l[0]
        basis_r = decompose_r[0]

        rotate_l = decompose_l[1]
        rotate_r = decompose_r[1]

        rotate_inv_l = torch.linalg.pinv(rotate_l)
        rotate_inv_r = torch.linalg.pinv(rotate_r)

        update_l = helper.get_unmagnified_grad(update_l, rotate_inv_r, lora_l_dim)
        update_r = helper.get_unmagnified_grad(update_r, rotate_inv_l, lora_r_dim)

        if update_skipping > 0:
          update_l = helper.skip_update(update_l, update_skipping)
          update_r = helper.skip_update(update_r, update_skipping)

        # TODO: find some other way to do this?
        state.basis_l = basis_l
        state.basis_r = basis_r
        state.rotate_inv_l = rotate_inv_l
        state.rotate_inv_r = rotate_inv_r
        state.update_l = update_l
        state.update_r = update_r

        g_norm_sq += torch.linalg.norm(update_l)**2
        g_norm_sq += torch.linalg.norm(update_r)**2
    g_norm = torch.sqrt(g_norm_sq)

    for group in self.param_groups:
      # TODO: make this more generic
      for p1, p2 in list(zip(group["params"], group["params"][1:]))[::2]:
        param_l, param_r = p1.data, p2.data

        state = self.state[p1]["attr"]
        count = state.step

        basis_l = state.basis_l
        basis_r = state.basis_r
        rotate_inv_l = state.rotate_inv_l
        rotate_inv_r = state.rotate_inv_r
        update_l = state.update_l
        update_r = state.update_r

        if clip_unmagnified_grad > 0:
          if g_norm > clip_unmagnified_grad:
            update_l = update_l/g_norm*clip_unmagnified_grad
            update_r = update_r/g_norm*clip_unmagnified_grad

        s_l = helper.compute_second_moments(update_l, lora_l_dim)
        s_r = helper.compute_second_moments(update_r, lora_r_dim)

        transformed_v_l = helper.transform_second_moment_to_new_basis(
            state.v_l,
            state.basis_r_old,
            basis_r,
            lora_l_dim,
            lora_r_dim,
        )
        transformed_v_r = helper.transform_second_moment_to_new_basis(
            state.v_r,
            state.basis_l_old,
            basis_l,
            lora_r_dim,
            lora_l_dim,
        )

        if apply_escape:
          escape_l = helper.get_unmagnified_rotate_second_escape(
              transformed_v_l,
              state.v_l,
          )

          escape_r = helper.get_unmagnified_rotate_second_escape(
              transformed_v_r,
              state.v_r,
          )

          escape_l = helper.update_second_escape(
              count, 0, escape_l + state.escape_l, beta2
          )
          escape_r = helper.update_second_escape(
              count, 0, escape_r + state.escape_r, beta2
          )
        else:
          escape_l = escape_r = 0

        v_l = helper.update_second_moments(count, s_l, transformed_v_l, beta2)
        v_r = helper.update_second_moments(count, s_r, transformed_v_r, beta2)

        update_l = helper.get_preconditioned_update(
            update_l,
            v_l,
            escape_l,
            epsilon,
            epsilon_root,
            relative_epsilon,
            apply_escape,
            lora_l_dim,
        )
        update_r = helper.get_preconditioned_update(
            update_r,
            v_r,
            escape_r,
            epsilon,
            epsilon_root,
            relative_epsilon,
            apply_escape,
            lora_r_dim,
        )

        m_l = helper.transform_first_moment_to_new_basis(
            state.m_l,
            state.basis_r_old,
            basis_r,
            lora_l_dim,
            lora_r_dim,
        )
        m_r = helper.transform_first_moment_to_new_basis(
            state.m_r,
            state.basis_l_old,
            basis_l,
            lora_r_dim,
            lora_l_dim,
        )

        m_l = helper.update_first_moments(count, update_l, m_l, beta1)
        m_r = helper.update_first_moments(count, update_r, m_r, beta1)

        update_l = m_l
        update_r = m_r

        if update_capping > 0:
          update_l = helper.clip_update(update_l, update_capping)
          update_r = helper.clip_update(update_r, update_capping)

        update_l = helper.rotate_update(update_l, rotate_inv_r, lora_l_dim)
        update_r = helper.rotate_update(update_r, rotate_inv_l, lora_r_dim)

        if weight_decay > 0:
          update_l = update_l + weight_decay * param_l
          update_r = update_r + weight_decay * param_r

        step_size = -1.0 * group["lr"]

        # Finally, fold in step size.
        update_l = step_size * update_l
        update_r = step_size * update_r

        if balance_param:
          l_norm = torch.linalg.norm(param_l + update_l) + 1e-6
          r_norm = torch.linalg.norm(param_r + update_r) + 1e-6

          balanced_norm = torch.sqrt(l_norm * r_norm)

          update_l = update_l * (balanced_norm / l_norm) + param_l * (
              balanced_norm / l_norm - 1
          )
          update_r = update_r * (balanced_norm / r_norm) + param_r * (
              balanced_norm / r_norm - 1
          )

        p1.data.add_(update_l.to(p1.data.dtype))
        p2.data.add_(update_r.to(p2.data.dtype))

        basis_l_old = basis_l
        basis_r_old = basis_r
        state.step += 1
        state.v_l = v_l
        state.v_r = v_r
        state.m_l = m_l
        state.m_r = m_r
        state.basis_l_old = basis_l_old
        state.basis_r_old = basis_r_old
        state.escape_l = escape_l
        state.escape_r = escape_r

    return loss
