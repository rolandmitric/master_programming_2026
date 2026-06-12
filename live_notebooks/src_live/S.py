import numpy as np
from numba import njit
@njit(cache = True, fastmath = True)
def S(i, j, k, l, m, n, Dx, Dy, Dz, KAB, P, Q, alpha, beta):
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 0, 0):
        return np.pi**(3/2)*KAB/P**(3/2)
    if (i, j, k, l, m, n) == (0, 0, 0, 1, 0, 0):
        return np.pi**(3/2)*Dx*KAB*alpha/P**(5/2)
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 1, 0):
        return np.pi**(3/2)*Dy*KAB*alpha/P**(5/2)
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 0, 1):
        return np.pi**(3/2)*Dz*KAB*alpha/P**(5/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 0, 0):
        return -np.pi**(3/2)*Dx*KAB*beta/P**(5/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 1, 0, 0):
        return (1/2)*np.pi**(3/2)*KAB*(-2*Dx**2*Q + P)/P**(7/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 1, 0):
        return -np.pi**(3/2)*Dx*Dy*KAB*Q/P**(7/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 0, 1):
        return -np.pi**(3/2)*Dx*Dz*KAB*Q/P**(7/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 0, 0):
        return -np.pi**(3/2)*Dy*KAB*beta/P**(5/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 1, 0, 0):
        return -np.pi**(3/2)*Dx*Dy*KAB*Q/P**(7/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 1, 0):
        return (1/2)*np.pi**(3/2)*KAB*(-2*Dy**2*Q + P)/P**(7/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 0, 1):
        return -np.pi**(3/2)*Dy*Dz*KAB*Q/P**(7/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 0, 0):
        return -np.pi**(3/2)*Dz*KAB*beta/P**(5/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 1, 0, 0):
        return -np.pi**(3/2)*Dx*Dz*KAB*Q/P**(7/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 1, 0):
        return -np.pi**(3/2)*Dy*Dz*KAB*Q/P**(7/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 0, 1):
        return (1/2)*np.pi**(3/2)*KAB*(-2*Dz**2*Q + P)/P**(7/2)