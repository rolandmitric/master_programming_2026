import numpy as np
from numba import njit
@njit(cache = True, fastmath = True)
def T(i,j,k,l,m,n,Dx, Dy, Dz, P, Q, RAB2, alpha, beta):
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 0, 0):
        return np.pi**(3/2)*Q*(-2*Q*RAB2 + 3*alpha + 3*beta)*np.exp(-Q*RAB2/P)/P**(7/2)
    if (i, j, k, l, m, n) == (0, 0, 0, 1, 0, 0):
        return np.pi**(3/2)*Dx*Q*alpha*(-2*Q*RAB2 + 5*alpha + 5*beta)*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 1, 0):
        return np.pi**(3/2)*Dy*Q*alpha*(-2*Q*RAB2 + 5*alpha + 5*beta)*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 0, 1):
        return np.pi**(3/2)*Dz*Q*alpha*(-2*Q*RAB2 + 5*alpha + 5*beta)*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 0, 0):
        return np.pi**(3/2)*Dx*Q*beta*(-2*P + 2*Q*RAB2 - 3*alpha - 3*beta)*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 1, 0, 0):
        return -1/2*np.pi**(3/2)*Q*(2*Dx**2*Q*(-2*Q*RAB2 + 3*alpha + 3*beta) - 2*P**2 + P*(8*Dx**2*Q + 2*Q*RAB2 - 3*alpha - 3*beta))*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 1, 0):
        return np.pi**(3/2)*Dx*Dy*Q**2*(2*Q*RAB2 - 7*alpha - 7*beta)*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 0, 1):
        return np.pi**(3/2)*Dx*Dz*Q**2*(2*Q*RAB2 - 7*alpha - 7*beta)*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 0, 0):
        return np.pi**(3/2)*Dy*Q*beta*(-2*P + 2*Q*RAB2 - 3*alpha - 3*beta)*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 1, 0, 0):
        return np.pi**(3/2)*Dx*Dy*Q**2*(2*Q*RAB2 - 7*alpha - 7*beta)*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 1, 0):
        return -1/2*np.pi**(3/2)*Q*(2*Dy**2*Q*(-2*Q*RAB2 + 3*alpha + 3*beta) - 2*P**2 + P*(8*Dy**2*Q + 2*Q*RAB2 - 3*alpha - 3*beta))*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 0, 1):
        return np.pi**(3/2)*Dy*Dz*Q**2*(2*Q*RAB2 - 7*alpha - 7*beta)*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 0, 0):
        return np.pi**(3/2)*Dz*Q*beta*(-2*P + 2*Q*RAB2 - 3*alpha - 3*beta)*np.exp(-Q*RAB2/P)/P**(9/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 1, 0, 0):
        return np.pi**(3/2)*Dx*Dz*Q**2*(2*Q*RAB2 - 7*alpha - 7*beta)*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 1, 0):
        return np.pi**(3/2)*Dy*Dz*Q**2*(2*Q*RAB2 - 7*alpha - 7*beta)*np.exp(-Q*RAB2/P)/P**(11/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 0, 1):
        return -1/2*np.pi**(3/2)*Q*(2*Dz**2*Q*(-2*Q*RAB2 + 3*alpha + 3*beta) - 2*P**2 + P*(8*Dz**2*Q + 2*Q*RAB2 - 3*alpha - 3*beta))*np.exp(-Q*RAB2/P)/P**(11/2)