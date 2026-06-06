import math
import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def Boys(n, t):
    if t < 1e-8:
        # F_n(t) small-t series
        s = 0.0
        term_t = 1.0
        factorial = 1.0
        sign = 1.0
        for k in range(8):
            if k > 0:
                term_t *= t
                factorial *= k
                sign = -sign
            s += sign * term_t / (factorial * (2 * n + 2 * k + 1))
        return s

    root_t = np.sqrt(t)
    exp_t = np.exp(-t)
    inv2t = 0.5 / t
    f = 0.5 * np.sqrt(np.pi) * math.erf(root_t) / root_t
    if n == 0:
        return f

    for m in range(1, n + 1):
        f = ((2 * m - 1) * f - exp_t) * inv2t
    return f


@njit(cache=True, fastmath=True)
def Boys012(t):
    """Return F_0(t), F_1(t), and F_2(t) in one pass."""
    if t < 1e-8:
        f0 = 0.0
        f1 = 0.0
        f2 = 0.0
        term_t = 1.0
        factorial = 1.0
        sign = 1.0
        for k in range(8):
            if k > 0:
                term_t *= t
                factorial *= k
                sign = -sign
            f0 += sign * term_t / (factorial * (2 * 0 + 2 * k + 1))
            f1 += sign * term_t / (factorial * (2 * 1 + 2 * k + 1))
            f2 += sign * term_t / (factorial * (2 * 2 + 2 * k + 1))
        return f0, f1, f2

    root_t = np.sqrt(t)
    exp_t = np.exp(-t)
    inv2t = 0.5 / t
    f0 = 0.5 * np.sqrt(np.pi) * math.erf(root_t) / root_t
    f1 = (f0 - exp_t) * inv2t
    f2 = (3.0 * f1 - exp_t) * inv2t
    return f0, f1, f2


@njit(cache=True, fastmath=True)
def Boys01234(t):
    """Return F_0(t) through F_4(t) in one pass."""
    if t < 1e-8:
        f0 = 0.0
        f1 = 0.0
        f2 = 0.0
        f3 = 0.0
        f4 = 0.0
        term_t = 1.0
        factorial = 1.0
        sign = 1.0
        for k in range(8):
            if k > 0:
                term_t *= t
                factorial *= k
                sign = -sign
            f0 += sign * term_t / (factorial * (2 * 0 + 2 * k + 1))
            f1 += sign * term_t / (factorial * (2 * 1 + 2 * k + 1))
            f2 += sign * term_t / (factorial * (2 * 2 + 2 * k + 1))
            f3 += sign * term_t / (factorial * (2 * 3 + 2 * k + 1))
            f4 += sign * term_t / (factorial * (2 * 4 + 2 * k + 1))
        return f0, f1, f2, f3, f4

    root_t = np.sqrt(t)
    exp_t = np.exp(-t)
    inv2t = 0.5 / t
    f0 = 0.5 * np.sqrt(np.pi) * math.erf(root_t) / root_t
    f1 = (f0 - exp_t) * inv2t
    f2 = (3.0 * f1 - exp_t) * inv2t
    f3 = (5.0 * f2 - exp_t) * inv2t
    f4 = (7.0 * f3 - exp_t) * inv2t
    return f0, f1, f2, f3, f4
