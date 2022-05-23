'''
This part of code is mainly comes form Neural ODE
https://arxiv.org/abs/1806.07366
and
https://github.com/rtqichen/torchdiffeq
with some modifications
'''

import abc
import warnings
from enum import Enum
import bisect
import collections
import numpy as np
import torch

class FixedGridODESolver(metaclass=abc.ABCMeta):
    order: int

    def __init__(self, func, h_tilde, h0, mode, step_size=None, grid_constructor=None, interp="linear", perturb=False, **unused_kwargs):
        self.atol = unused_kwargs.pop('atol')
        unused_kwargs.pop('rtol', None)
        unused_kwargs.pop('norm', None)
        del unused_kwargs

        self.func = func
        self.h0 = h0
        self.h_tilde = h_tilde
        self.dtype = h0.dtype
        self.device = h0.device
        self.step_size = step_size
        self.perturb = perturb
        self.interp = interp
        self.mode = mode

        if step_size is None:
            if grid_constructor is None:
                self.grid_constructor = lambda f, h0, t: t
            else:
                self.grid_constructor = grid_constructor
        else:
            if grid_constructor is None:
                self.grid_constructor = self._grid_constructor_from_step_size(step_size)
            else:
                raise ValueError("step_size and grid_constructor are mutually exclusive arguments.")

    @staticmethod
    def _grid_constructor_from_step_size(step_size):
        def _grid_constructor(func, h0, t):
            start_time = t[0]
            end_time = t[-1]

            niters = torch.ceil((end_time - start_time) / step_size + 1).item()
            t_infer = torch.arange(0, niters, dtype=t.dtype, device=t.device) * step_size + start_time
            t_infer[-1] = t[-1]

            return t_infer
        return _grid_constructor

    @abc.abstractmethod
    def _step_func(self, func, t0, dt, t1, h0, h_tilde, mode):
        pass

    def integrate(self, t):
        time_grid = self.grid_constructor(self.func, self.h0, t)
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]

        solution = torch.empty(len(t), *self.h0.shape, dtype=self.h0.dtype, device=self.h0.device)
        solution[0] = self.h0

        j = 1
        h0 = self.h0
        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            dt = t1 - t0
            dh, f0, h_tilde = self._step_func(self.func, t0, dt, t1, h0, self.h_tilde, self.mode)
            h1 = h0 + dh

            while j < len(t) and t1 >= t[j]:
                if self.interp == "linear":
                    solution[j] = self._linear_interp(t0, t1, h0, h1, t[j])
                elif self.interp == "cubic":
                    f1 = self.func(t1, h1)
                    solution[j] = self._cubic_hermite_interp(t0, h0, f0, t1, h1, f1, t[j])
                else:
                    raise ValueError(f"Unknown interpolation method {self.interp}")
                j += 1
            h0 = h1

        return solution, h_tilde

    def _cubic_hermite_interp(self, t0, h0, f0, t1, h1, f1, t):
        h = (t - t0) / (t1 - t0)
        h00 = (1 + 2 * h) * (1 - h) * (1 - h)
        h10 = h * (1 - h) * (1 - h)
        h01 = h * h * (3 - 2 * h)
        h11 = h * h * (h - 1)
        dt = (t1 - t0)
        return h00 * h0 + h10 * dt * f0 + h01 * h1 + h11 * dt * f1

    def _linear_interp(self, t0, t1, h0, h1, t):
        if t == t0:
            return h0
        if t == t1:
            return h1
        slope = (t - t0) / (t1 - t0)
        return h0 + slope * (h1 - h0)

class Euler(FixedGridODESolver):
    order = 1

    def _step_func(self, func, t0, dt, t1, h0, h_tilde, mode):
        if mode == 'mgn':
            f0, h_tilde = func(t0, h0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        elif mode == 'joint':
            f0 = func(t0, h_tilde, h0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
            h_tilde = None
        return dt * f0, f0, h_tilde

class Midpoint(FixedGridODESolver):
    order = 2

    def _step_func(self, func, t0, dt, t1, h0, h_tilde, mode):
        half_dt = 0.5 * dt
        if mode == 'mgn':
            f0, h_tilde = func(t0, h0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
            h_mid = h0 + f0 * half_dt
            f0_mid, h_tilde = func(t0 + half_dt, h_mid)
        elif mode == 'joint':
            f0 = func(t0, h_tilde, h0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
            h_mid = h0 + f0 * half_dt
            f0_mid = func(t0 + half_dt, h_tilde, h_mid)
            h_tilde = None
        return dt * f0_mid, f0, h_tilde

class RK4(FixedGridODESolver):
    order = 4

    def _step_func(self, func, t0, dt, t1, h0, h_tilde, mode):
        _one_third = 1 / 3
        _two_thirds = 2 / 3
        _one_sixth = 1 / 6
        if mode == 'mgn':
            k1, h_tilde = func(t0, h0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
            k2, h_tilde = func(t0 + dt * _one_third, h0 + dt * k1 * _one_third)
            k3, h_tilde = func(t0 + dt * _two_thirds, h0 + dt * (k2 - k1 * _one_third))
            k4, h_tilde = func(t1, h0 + dt * (k1 - k2 + k3), perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
            f0 = k1
        elif mode == 'joint':
            k1 = func(t0, h_tilde, h0)
            k2 = func(t0 + dt * _one_third, h_tilde, h0 + dt * k1 * _one_third)
            k3 = func(t0 + dt * _two_thirds, h_tilde, h0 + dt * (k2 - k1 * _one_third))
            k4 = func(t1, h_tilde, h0 + dt * (k1 - k2 + k3), perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
            f0 = k1
            h_tilde = None
        return (k1 + 3 * (k2 + k3) + k4) * dt * 0.125, f0, h_tilde

def _rms_norm(tensor):
    return tensor.pow(2).mean().sqrt()

def _mixed_norm(tensor_tuple):
    if len(tensor_tuple) == 0:
        return 0.
    return max([_rms_norm(tensor) for tensor in tensor_tuple])

class AdaptiveStepsizeODESolver(metaclass=abc.ABCMeta):
    def __init__(self, dtype, h0, **unused_kwargs):
        del unused_kwargs

        self.h0 = h0
        self.dtype = dtype
        self.norm = _mixed_norm

    def _before_integrate(self, t):
        pass

    @abc.abstractmethod
    def _advance(self, next_t):
        raise NotImplementedError

    def integrate(self, t):
        solution = torch.empty(len(t), *self.h0.shape, dtype=self.h0.dtype, device=self.h0.device)
        solution[0] = self.h0
        t = t.to(self.dtype)
        self._before_integrate(t)
        for i in range(1, len(t)):
            solution[i], h_tilde = self._advance(t[i])
        return solution, h_tilde

def _select_initial_step(func, t0, h_tilde, h0, order, rtol, atol, norm, mode, f0=None):
    dtype = h0.dtype
    device = h0.device
    t_dtype = t0.dtype
    t0 = t0.to(dtype)

    scale = atol + torch.abs(h0) * rtol
    d0 = norm(h0 / scale)
    d1 = norm(f0 / scale)

    if d0 < 1e-5 or d1 < 1e-5:
        a0 = torch.tensor(1e-6, dtype=dtype, device=device)
    else:
        a0 = 0.01 * d0 / d1

    h1 = h0 + a0 * f0

    if mode == 'mgn':
        f1, _ = func(t0 + a0, h1)

    elif mode == 'joint':
        f1 = func(t0 + a0, h_tilde, h1)

    d2 = norm((f1 - f0) / scale) / a0

    if d1 <= 1e-15 and d2 <= 1e-15:
        a1 = torch.max(torch.tensor(1e-6, dtype=dtype, device=device), a0 * 1e-3)
    else:
        a1 = (0.01 / max(d1, d2)) ** (1. / float(order + 1))

    return torch.min(100 * a0, a1).to(t_dtype)

def _sort_tvals(tvals, t0):
    # TODO: add warning if tvals come before t0?
    tvals = tvals[tvals >= t0]
    return torch.sort(tvals).values

def _interp_fit(y0, y1, y_mid, f0, f1, dt):
    """Fit coefficients for 4th order polynomial interpolation.

    Args:
        y0: function value at the start of the interval.
        y1: function value at the end of the interval.
        y_mid: function value at the mid-point of the interval.
        f0: derivative value at the start of the interval.
        f1: derivative value at the end of the interval.
        dt: width of the interval.

    Returns:
        List of coefficients `[a, b, c, d, e]` for interpolating with the polynomial
        `p = a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e` for values of `x`
        between 0 (start of interval) and 1 (end of interval).
    """
    a = 2 * dt * (f1 - f0) - 8 * (y1 + y0) + 16 * y_mid
    b = dt * (5 * f0 - 3 * f1) + 18 * y0 + 14 * y1 - 32 * y_mid
    c = dt * (f1 - 4 * f0) - 11 * y0 - 5 * y1 + 16 * y_mid
    d = dt * f0
    e = y0
    return [e, d, c, b, a]

def _interp_evaluate(coefficients, t0, t1, t):
    """Evaluate polynomial interpolation at the given time point.

    Args:
        coefficients: list of Tensor coefficients as created by `interp_fit`.
        t0: scalar float64 Tensor giving the start of the interval.
        t1: scalar float64 Tensor giving the end of the interval.
        t: scalar float64 Tensor giving the desired interpolation point.

    Returns:
        Polynomial interpolation of the coefficients at time `t`.
    """

    assert (t0 <= t) & (t <= t1), 'invalid interpolation, fails `t0 <= t <= t1`: {}, {}, {}'.format(t0, t, t1)
    x = (t - t0) / (t1 - t0)
    x = x.to(coefficients[0].dtype)

    total = coefficients[0] + x * coefficients[1]
    x_power = x
    for coefficient in coefficients[2:]:
        x_power = x_power * x
        total = total + x_power * coefficient

    return total

class Perturb(Enum):
    NONE = 0
    PREV = 1
    NEXT = 2

class _UncheckedAssign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scratch, value, index):
        ctx.index = index
        scratch.data[index] = value  # sneak past the version checker
        return scratch

    @staticmethod
    def backward(ctx, grad_scratch):
        return grad_scratch, grad_scratch[ctx.index], None

def _runge_kutta_step(func, h_tilde, h0, f0, t0, dt, t1, mode, tableau):
    """Take an arbitrary Runge-Kutta step and estimate error.
    Args:
        func: Function to evaluate like `func(t, y)` to compute the time derivative of `y`.
        y0: Tensor initial value for the state.
        f0: Tensor initial value for the derivative, computed from `func(t0, y0)`.
        t0: float64 scalar Tensor giving the initial time.
        dt: float64 scalar Tensor giving the size of the desired time step.
        t1: float64 scalar Tensor giving the end time; equal to t0 + dt. This is used (rather than t0 + dt) to ensure
            floating point accuracy when needed.
        tableau: _ButcherTableau describing how to take the Runge-Kutta step.
    Returns:
        Tuple `(y1, f1, y1_error, k)` giving the estimated function value after
        the Runge-Kutta step at `t1 = t0 + dt`, the derivative of the state at `t1`,
        estimated error at `t1`, and a list of Runge-Kutta coefficients `k` used for
        calculating these terms.
    """

    t0 = t0.to(h0.dtype)
    dt = dt.to(h0.dtype)
    t1 = t1.to(h0.dtype)

    # We use an unchecked assign to put data into k without incrementing its _version counter, so that the backward
    # doesn't throw an (overzealous) error about in-place correctness. We know that it's actually correct.
    k = torch.empty(*f0.shape, len(tableau.alpha) + 1, dtype=h0.dtype, device=h0.device)
    k = _UncheckedAssign.apply(k, f0, (..., 0))
    for i, (alpha_i, beta_i) in enumerate(zip(tableau.alpha, tableau.beta)):
        if alpha_i == 1.:
            # Always step to perturbing just before the end time, in case of discontinuities.
            ti = t1
            perturb = Perturb.PREV
        else:
            ti = t0 + alpha_i * dt
            perturb = Perturb.NONE
        hi = h0 + k[..., :i + 1].matmul(beta_i * dt).view_as(f0)
        if mode == 'mgn':
            f, h_tilde = func(ti, hi, perturb=perturb)
        elif mode == 'joint':
            f = func(ti, h_tilde, hi, perturb=perturb)
        k = _UncheckedAssign.apply(k, f, (..., i + 1))

    if not (tableau.c_sol[-1] == 0 and (tableau.c_sol[:-1] == tableau.beta[-1]).all()):
        # This property (true for Dormand-Prince) lets us save a few FLOPs.
        hi = h0 + k.matmul(dt * tableau.c_sol).view_as(f0)

    h1 = hi
    f1 = k[..., -1]
    y1_error = k.matmul(dt * tableau.c_error)
    return h_tilde, h1, f1, y1_error, k

def _compute_error_ratio(error_estimate, rtol, atol, y0, y1, norm):
    error_tol = atol + rtol * torch.max(y0.abs(), y1.abs())
    return norm(error_estimate / error_tol)

@torch.no_grad()
def _optimal_step_size(last_step, error_ratio, safety, ifactor, dfactor, order):
    """Calculate the optimal size for the next step."""
    if error_ratio == 0:
        return last_step * ifactor
    if error_ratio < 1:
        dfactor = torch.ones((), dtype=last_step.dtype, device=last_step.device)
    error_ratio = error_ratio.type_as(last_step)
    exponent = torch.tensor(order, dtype=last_step.dtype, device=last_step.device).reciprocal()
    factor = torch.min(ifactor, torch.max(safety / error_ratio ** exponent, dfactor))
    return last_step * factor

_ButcherTableau = collections.namedtuple('_ButcherTableau', 'alpha, beta, c_sol, c_error')
_RungeKuttaState = collections.namedtuple('_RungeKuttaState', 'h_tilde, h1, f1, t0, t1, dt, interp_coeff, mode')

class RKAdaptiveStepsizeODESolver(AdaptiveStepsizeODESolver):
    order: int
    tableau: _ButcherTableau
    mid: torch.Tensor

    def __init__(self, func, h_tilde, h0, mode, rtol, atol,
                 first_step=None,
                 step_t=None,
                 jump_t=None,
                 safety=0.9,
                 ifactor=10.0,
                 dfactor=0.2,
                 max_num_steps=2 ** 31 - 1,
                 dtype=torch.float64,
                 **kwargs):
        super(RKAdaptiveStepsizeODESolver, self).__init__(dtype=dtype, h0=h0, **kwargs)


        dtype = torch.promote_types(dtype, h0.dtype)
        device = h0.device

        self.func = func
        self.h_tilde = h_tilde
        self.mode = mode
        self.rtol = torch.as_tensor(rtol, dtype=dtype, device=device)
        self.atol = torch.as_tensor(atol, dtype=dtype, device=device)
        self.first_step = None if first_step is None else torch.as_tensor(first_step, dtype=dtype, device=device)
        self.safety = torch.as_tensor(safety, dtype=dtype, device=device)
        self.ifactor = torch.as_tensor(ifactor, dtype=dtype, device=device)
        self.dfactor = torch.as_tensor(dfactor, dtype=dtype, device=device)
        self.max_num_steps = torch.as_tensor(max_num_steps, dtype=torch.int32, device=device)
        self.dtype = dtype

        self.step_t = None if step_t is None else torch.as_tensor(step_t, dtype=dtype, device=device)
        self.jump_t = None if jump_t is None else torch.as_tensor(jump_t, dtype=dtype, device=device)

        # Copy from class to instance to set device
        self.tableau = _ButcherTableau(alpha=self.tableau.alpha.to(device=device, dtype=h0.dtype),
                                       beta=[b.to(device=device, dtype=h0.dtype) for b in self.tableau.beta],
                                       c_sol=self.tableau.c_sol.to(device=device, dtype=h0.dtype),
                                       c_error=self.tableau.c_error.to(device=device, dtype=h0.dtype))
        self.mid = self.mid.to(device=device, dtype=h0.dtype)

    def _before_integrate(self, t):
        if self.mode == 'mgn':
            t0 = t[0]
            f0, h_tilde = self.func(t[0], self.h0)
            if self.first_step is None:
                first_step = _select_initial_step(self.func, t[0], None, self.h0, self.order - 1, self.rtol, self.atol,
                                                  self.norm, self.mode, f0=f0)
            else:
                first_step = self.first_step

        elif self.mode == 'joint':
            t0 = t[0]
            f0 = self.func(t[0], self.h_tilde, self.h0)
            if self.first_step is None:
                first_step = _select_initial_step(self.func, t[0], self.h_tilde, self.h0, self.order - 1, self.rtol, self.atol,
                                                  self.norm, self.mode, f0=f0)
            else:
                first_step = self.first_step

        self.rk_state = _RungeKuttaState(self.h_tilde, self.h0, f0, t[0], t[0], first_step, [self.h0] * 5, self.mode)

        # Handle step_t and jump_t arguments.
        if self.step_t is None:
            step_t = torch.tensor([], dtype=self.dtype, device=self.h0.device)
        else:
            step_t = _sort_tvals(self.step_t, t0)
            step_t = step_t.to(self.dtype)
        if self.jump_t is None:
            jump_t = torch.tensor([], dtype=self.dtype, device=self.h0.device)
        else:
            jump_t = _sort_tvals(self.jump_t, t0)
            jump_t = jump_t.to(self.dtype)
        counts = torch.cat([step_t, jump_t]).unique(return_counts=True)[1]
        if (counts > 1).any():
            raise ValueError("`step_t` and `jump_t` must not have any repeated elements between them.")

        self.step_t = step_t
        self.jump_t = jump_t
        self.next_step_index = min(bisect.bisect(self.step_t.tolist(), t[0]), len(self.step_t) - 1)
        self.next_jump_index = min(bisect.bisect(self.jump_t.tolist(), t[0]), len(self.jump_t) - 1)

    def _advance(self, next_t):
        """Interpolate through the next time point, integrating as necessary."""
        n_steps = 0
        while next_t > self.rk_state.t1:
            assert n_steps < self.max_num_steps, 'max_num_steps exceeded ({}>={})'.format(n_steps, self.max_num_steps)
            self.rk_state = self._adaptive_step(self.rk_state)
            n_steps += 1
        return _interp_evaluate(self.rk_state.interp_coeff, self.rk_state.t0, self.rk_state.t1, next_t), self.rk_state.h_tilde

    def _adaptive_step(self, rk_state):
        """Take an adaptive Runge-Kutta step to integrate the ODE."""
        # _RungeKuttaState = collections.namedtuple('_RungeKuttaState', 'y1, f1, t0, t1, dt, interp_coeff')
        # self.rk_state = _RungeKuttaState(self.h0, f0, t[0], t[0], first_step, [self.h0] * 5)
        h_tilde, h0, f0, _, t0, dt, interp_coeff, mode = rk_state
        t1 = t0 + dt

        ########################################################
        #                      Assertions                      #
        ########################################################
        assert t0 + dt > t0, 'underflow in dt {}'.format(dt.item())
        assert torch.isfinite(h0).all(), 'non-finite values in state `y`: {}'.format(h0)

        ########################################################
        #     Make step, respecting prescribed grid points     #
        ########################################################

        on_step_t = False
        if len(self.step_t):
            next_step_t = self.step_t[self.next_step_index]
            on_step_t = t0 < next_step_t < t0 + dt
            if on_step_t:
                t1 = next_step_t
                dt = t1 - t0

        on_jump_t = False
        if len(self.jump_t):
            next_jump_t = self.jump_t[self.next_jump_index]
            on_jump_t = t0 < next_jump_t < t0 + dt
            if on_jump_t:
                on_step_t = False
                t1 = next_jump_t
                dt = t1 - t0

        # Must be arranged as doing all the step_t handling, then all the jump_t handling, in case we
        # trigger both. (i.e. interleaving them would be wrong.)

        h_tilde, h1, f1, h1_error, k = _runge_kutta_step(self.func, h_tilde, h0, f0, t0, dt, t1, mode, tableau=self.tableau)

        ########################################################
        #                     Error Ratio                      #
        ########################################################
        error_ratio = _compute_error_ratio(h1_error, self.rtol, self.atol, h0, h1, self.norm)
        accept_step = error_ratio <= 1
        # dtypes:
        # error_ratio.dtype == self.dtype

        ########################################################
        #                   Update RK State                    #
        ########################################################
        if accept_step:
            t_next = t1
            h_next = h1
            interp_coeff = self._interp_fit(h0, h_next, k, dt)
            if on_step_t:
                if self.next_step_index != len(self.step_t) - 1:
                    self.next_step_index += 1
            if on_jump_t:
                if self.next_jump_index != len(self.jump_t) - 1:
                    self.next_jump_index += 1
                # We've just passed a discontinuity in f; we should update f to match the side of the discontinuity
                # we're now on.
                if mode == 'mgn':
                    f1, h_tilde = self.func(t_next, h_next, perturb=Perturb.NEXT)
                elif mode == 'joint':
                    f1 = self.func(t_next, h_tilde, h_next, perturb=Perturb.NEXT)
            f_next = f1
        else:
            t_next = t0
            h_next = h0
            f_next = f0
        dt_next = _optimal_step_size(dt, error_ratio, self.safety, self.ifactor, self.dfactor, self.order)
        rk_state = _RungeKuttaState(h_tilde, h_next, f_next, t0, t_next, dt_next, interp_coeff, mode)
        return rk_state

    def _interp_fit(self, h0, h1, k, dt):
        """Fit an interpolating polynomial to the results of a Runge-Kutta step."""
        dt = dt.type_as(h0)
        y_mid = h0 + k.matmul(dt * self.mid).view_as(h0)
        f0 = k[..., 0]
        f1 = k[..., -1]
        return _interp_fit(h0, h1, y_mid, f0, f1, dt)

_DORMAND_PRINCE_SHAMPINE_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([1 / 5, 3 / 10, 4 / 5, 8 / 9, 1., 1.], dtype=torch.float64),
    beta=[
        torch.tensor([1 / 5], dtype=torch.float64),
        torch.tensor([3 / 40, 9 / 40], dtype=torch.float64),
        torch.tensor([44 / 45, -56 / 15, 32 / 9], dtype=torch.float64),
        torch.tensor([19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729], dtype=torch.float64),
        torch.tensor([9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656], dtype=torch.float64),
        torch.tensor([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84], dtype=torch.float64),
    ],
    c_sol=torch.tensor([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0], dtype=torch.float64),
    c_error=torch.tensor([
        35 / 384 - 1951 / 21600,
        0,
        500 / 1113 - 22642 / 50085,
        125 / 192 - 451 / 720,
        -2187 / 6784 - -12231 / 42400,
        11 / 84 - 649 / 6300,
        -1. / 60.,
    ], dtype=torch.float64),
)

DPS_C_MID = torch.tensor([
    6025192743 / 30085553152 / 2, 0, 51252292925 / 65400821598 / 2, -2691868925 / 45128329728 / 2,
    187940372067 / 1594534317056 / 2, -1776094331 / 19743644256 / 2, 11237099 / 235043384 / 2
], dtype=torch.float64)


class Dopri5Solver(RKAdaptiveStepsizeODESolver):
    order = 5
    tableau = _DORMAND_PRINCE_SHAMPINE_TABLEAU
    mid = DPS_C_MID



def _tuple_tol(name, tol, shapes):
    try:
        iter(tol)
    except TypeError:
        return tol
    tol = tuple(tol)
    assert len(tol) == len(shapes), "If using tupled {} it must have the same length as the tuple y0".format(name)
    tol = [torch.as_tensor(tol_).expand(shape.numel()) for tol_, shape in zip(tol, shapes)]
    return torch.cat(tol)

def _flat_to_shape(tensor, length, shapes):
    tensor_list = []
    total = 0
    for shape in shapes:
        next_total = total + shape.numel()
        # It's important that this be view((...)), not view(...). Else when length=(), shape=() it fails.
        tensor_list.append(tensor[..., total:next_total].view((*length, *shape)))
        total = next_total
    return tuple(tensor_list)

class _TupleFunc(torch.nn.Module):
    def __init__(self, base_func, shapes):
        super(_TupleFunc, self).__init__()
        self.base_func = base_func
        self.shapes = shapes

    def forward(self, t, y):
        f = self.base_func(t, _flat_to_shape(y, (), self.shapes))
        return torch.cat([f_.reshape(-1) for f_ in f])

def _assert_floating(name, t):
    if not torch.is_floating_point(t):
        raise TypeError('`{}` must be a floating point Tensor but is a {}'.format(name, t.type()))

def _check_timelike(name, timelike, can_grad):
    assert isinstance(timelike, torch.Tensor), '{} must be a torch.Tensor'.format(name)
    _assert_floating(name, timelike)
    assert timelike.ndimension() == 1, "{} must be one dimensional".format(name)
    if not can_grad:
        assert not timelike.requires_grad, "{} cannot require gradient".format(name)
    diff = timelike[1:] > timelike[:-1]
    assert diff.all() or (~diff).all(), '{} must be strictly increasing or decreasing'.format(name)

class _ReverseFunc(torch.nn.Module):
    def __init__(self, base_func, mul=1.0):
        super(_ReverseFunc, self).__init__()
        self.base_func = base_func
        self.mul = mul

    def forward(self, t, y):
        return self.mul * self.base_func(-t, y)

def _flip_option(options, option_name):
    try:
        option_value = options[option_name]
    except KeyError:
        pass
    else:
        if isinstance(option_value, torch.Tensor):
            options[option_name] = -option_value
        # else: an error will be raised when the option is attempted to be used in Solver.__init__, but we defer raising
        # the error until then to keep things tidy.

def _assert_increasing(name, t):
    assert (t[1:] > t[:-1]).all(), '{} must be strictly increasing or decreasing'.format(name)

def _nextafter(x1, x2):
    with torch.no_grad():
        if hasattr(torch, "nextafter"):
            out = torch.nextafter(x1, x2)
        else:
            out = np_nextafter(x1, x2)
    return _StitchGradient.apply(x1, out)

def np_nextafter(x1, x2):
    warnings.warn("torch.nextafter is only available in PyTorch 1.7 or newer."
                  "Falling back to numpy.nextafter. Upgrade PyTorch to remove this warning.")
    x1_np = x1.detach().cpu().numpy()
    x2_np = x2.detach().cpu().numpy()
    out = torch.tensor(np.nextafter(x1_np, x2_np)).to(x1)
    return out

class _StitchGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, out):
        return out

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None

class _PerturbFunc_mgn(torch.nn.Module):

    def __init__(self, base_func):
        super(_PerturbFunc_mgn, self).__init__()
        self.base_func = base_func

    def forward(self, t, h, *, perturb=Perturb.NONE):
        assert isinstance(perturb, Perturb), "perturb argument must be of type Perturb enum"
        # This dtype change here might be buggy.
        # The exact time value should be determined inside the solver,
        # but this can slightly change it due to numerical differences during casting.
        t = t.to(h.dtype)
        if perturb is Perturb.NEXT:
            # Replace with next smallest representable value.
            t = _nextafter(t, t + 1)
        elif perturb is Perturb.PREV:
            # Replace with prev largest representable value.
            t = _nextafter(t, t - 1)
        else:
            # Do nothing.
            pass
        return self.base_func(t, h)

class _PerturbFunc_joint(torch.nn.Module):

    def __init__(self, base_func):
        super(_PerturbFunc_joint, self).__init__()
        self.base_func = base_func

    def forward(self, t, h_tilde, h, *, perturb=Perturb.NONE):
        assert isinstance(perturb, Perturb), "perturb argument must be of type Perturb enum"
        # This dtype change here might be buggy.
        # The exact time value should be determined inside the solver,
        # but this can slightly change it due to numerical differences during casting.
        t = t.to(h.dtype)
        if perturb is Perturb.NEXT:
            # Replace with next smallest representable value.
            t = _nextafter(t, t + 1)
        elif perturb is Perturb.PREV:
            # Replace with prev largest representable value.
            t = _nextafter(t, t - 1)
        else:
            # Do nothing.
            pass
        return self.base_func(t, h_tilde, h)

def _check_inputs(func, y0, t, rtol, atol, method, mode, SOLVERS, options=None):
    # Normalise to tensor (non-tupled) input
    shapes = None
    is_tuple = not isinstance(y0, torch.Tensor)
    if is_tuple:
        assert isinstance(y0, tuple), 'y0 must be either a torch.Tensor or a tuple'
        shapes = [y0_.shape for y0_ in y0]
        rtol = _tuple_tol('rtol', rtol, shapes)
        atol = _tuple_tol('atol', atol, shapes)
        y0 = torch.cat([y0_.reshape(-1) for y0_ in y0])
        func = _TupleFunc(func, shapes)

    _assert_floating('y0', y0)

    # Normalise method and options
    if options is None:
        options = {}
    else:
        options = options.copy()
    if method is None:
        method = 'dopri5'
    if method not in SOLVERS:
        raise ValueError('Invalid method "{}". Must be one of {}'.format(method, '{"' + '", "'.join(SOLVERS.keys()) + '"}.'))

    if is_tuple:
        # We accept tupled input. This is an abstraction that is hidden from the rest of odeint (exception when
        # returning values), so here we need to maintain the abstraction by wrapping norm functions.

        if 'norm' in options:
            # If the user passed a norm then get that...
            norm = options['norm']
        else:
            # ...otherwise we default to a mixed Linf/L2 norm over tupled input.
            norm = _mixed_norm

        # In either case, norm(...) is assumed to take a tuple of tensors as input. (As that's what the state looks
        # like from the point of view of the user.)
        # So here we take the tensor that the machinery of odeint has given us, and turn it in the tuple that the
        # norm function is expecting.
        def _norm(tensor):
            y = _flat_to_shape(tensor, (), shapes)
            return norm(y)
        options['norm'] = _norm

    else:
        if 'norm' in options:
            # No need to change the norm function.
            pass
        else:
            # Else just use the default norm.
            # Technically we don't need to set that here (RKAdaptiveStepsizeODESolver has it as a default), but it
            # makes it easier to reason about, in the adjoint norm logic, if we know that options['norm'] is
            # definitely set to something.
            options['norm'] = _rms_norm

    # Normalise time
    _check_timelike('t', t, True)
    t_is_reversed = False
    if len(t) > 1 and t[0] > t[1]:
        t_is_reversed = True

    if t_is_reversed:
        # Change the integration times to ascending order.
        # We do this by negating the time values and all associated arguments.
        t = -t

        # Ensure time values are un-negated when calling functions.
        func = _ReverseFunc(func, mul=-1.0)


        # For fixed step solvers.
        try:
            _grid_constructor = options['grid_constructor']
        except KeyError:
            pass
        else:
            options['grid_constructor'] = lambda func, y0, t: -_grid_constructor(func, y0, -t)

        # For RK solvers.
        _flip_option(options, 'step_t')
        _flip_option(options, 'jump_t')

    # Can only do after having normalised time
    _assert_increasing('t', t)

    # Tol checking
    if torch.is_tensor(rtol):
        assert not rtol.requires_grad, "rtol cannot require gradient"
    if torch.is_tensor(atol):
        assert not atol.requires_grad, "atol cannot require gradient"

    # Backward compatibility: Allow t and y0 to be on different devices
    if t.device != y0.device:
        warnings.warn("t is not on the same device as y0. Coercing to y0.device.")
        t = t.to(y0.device)
    # ~Backward compatibility

    # Add perturb argument to func.
    if mode == 'mgn':
        func = _PerturbFunc_mgn(func)
    elif mode == 'joint':
        func = _PerturbFunc_joint(func)

    return shapes, func, y0, t, rtol, atol, method, options, t_is_reversed

SOLVERS = {
    'dopri5': Dopri5Solver,
    'euler': Euler,
    'midpoint': Midpoint,
    'rk4': RK4,
}

def odeint(func, h_tilde, h0, t, *, rtol=1e-7, atol=1e-9, method=None, mode='mgn'):
    shapes, func, y0, t, rtol, atol, method, options, t_is_reversed \
        = _check_inputs(func, h0, t, rtol, atol, method, mode, SOLVERS)

    solver = SOLVERS[method](func=func, h_tilde=h_tilde, h0=h0, mode=mode, rtol=rtol, atol=atol, **options)
    solution, h_tilde = solver.integrate(t)
    return solution[-1], h_tilde
