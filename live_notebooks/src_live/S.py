import numpy as np
from numba import njit
@njit(cache=True, fastmath=True)
def S(i, j, k, l, m, n, Dx, Dy, Dz, P, Q, RAB2):
    if rep == 0:
        return np.pi**(3/2)*np.exp(-Q*(Dx**2 + Dy**2 + Dz**2)/P)/P**(3/2)
    if rep == 1:
        return np.pi**(3/2)*Dz*alpha*np.exp(-Q*(Dx**2 + Dy**2 + Dz**2)/P)/P**(5/2)
    if rep == 2:
        return (1/2)*np.pi**(3/2)*(-2*Dz**2*Q + P)*np.exp(-Q*(Dx**2 + Dy**2 + Dz**2)/P)/P**(7/2)
    if rep == 3:
        return -np.pi**(3/2)*Dy*Dz*Q*np.exp(-Q*(Dx**2 + Dy**2 + Dz**2)/P)/P**(7/2)
    raise KeyError((i,j,k,l,m,n))