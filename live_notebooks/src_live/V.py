import numpy as np
from .boys import Boys
from numba import njit
@njit(cache = True, fastmath = True)
def V(i, j, k, l, m, n, Dx, Dy, Dz, KAB, P, Q, Qx, Qy, Qz, alpha, beta, u):
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 0, 0):
        return 2*np.pi*KAB*Boys(0, u)/P
    if (i, j, k, l, m, n) == (0, 0, 0, 1, 0, 0):
        return 2*np.pi*KAB*(Dx*alpha*Boys(0, u) - Qx*Boys(1, u))/P**2
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 1, 0):
        return 2*np.pi*KAB*(Dy*alpha*Boys(0, u) - Qy*Boys(1, u))/P**2
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 0, 1):
        return 2*np.pi*KAB*(Dz*alpha*Boys(0, u) - Qz*Boys(1, u))/P**2
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 0, 0):
        return -2*np.pi*KAB*(Dx*beta*Boys(0, u) + Qx*Boys(1, u))/P**2
    if (i, j, k, l, m, n) == (1, 0, 0, 1, 0, 0):
        return np.pi*KAB*(-2*Dx**2*Q*Boys(0, u) - 2*Dx*Qx*alpha*Boys(1, u) + 2*Dx*Qx*beta*Boys(1, u) + P*(Boys(0, u) - Boys(1, u)) + 2*Qx**2*Boys(2, u))/P**3
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 1, 0):
        return 2*np.pi*KAB*(-Dx*Dy*Q*Boys(0, u) + Dx*Qy*beta*Boys(1, u) - Dy*Qx*alpha*Boys(1, u) + Qx*Qy*Boys(2, u))/P**3
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 0, 1):
        return 2*np.pi*KAB*(-Dx*Dz*Q*Boys(0, u) + Dx*Qz*beta*Boys(1, u) - Dz*Qx*alpha*Boys(1, u) + Qx*Qz*Boys(2, u))/P**3
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 0, 0):
        return -2*np.pi*KAB*(Dy*beta*Boys(0, u) + Qy*Boys(1, u))/P**2
    if (i, j, k, l, m, n) == (0, 1, 0, 1, 0, 0):
        return 2*np.pi*KAB*(-Dx*Dy*Q*Boys(0, u) - Dx*Qy*alpha*Boys(1, u) + Dy*Qx*beta*Boys(1, u) + Qx*Qy*Boys(2, u))/P**3
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 1, 0):
        return np.pi*KAB*(-2*Dy**2*Q*Boys(0, u) - 2*Dy*Qy*alpha*Boys(1, u) + 2*Dy*Qy*beta*Boys(1, u) + P*(Boys(0, u) - Boys(1, u)) + 2*Qy**2*Boys(2, u))/P**3
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 0, 1):
        return 2*np.pi*KAB*(-Dy*Dz*Q*Boys(0, u) + Dy*Qz*beta*Boys(1, u) - Dz*Qy*alpha*Boys(1, u) + Qy*Qz*Boys(2, u))/P**3
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 0, 0):
        return -2*np.pi*KAB*(Dz*beta*Boys(0, u) + Qz*Boys(1, u))/P**2
    if (i, j, k, l, m, n) == (0, 0, 1, 1, 0, 0):
        return 2*np.pi*KAB*(-Dx*Dz*Q*Boys(0, u) - Dx*Qz*alpha*Boys(1, u) + Dz*Qx*beta*Boys(1, u) + Qx*Qz*Boys(2, u))/P**3
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 1, 0):
        return 2*np.pi*KAB*(-Dy*Dz*Q*Boys(0, u) - Dy*Qz*alpha*Boys(1, u) + Dz*Qy*beta*Boys(1, u) + Qy*Qz*Boys(2, u))/P**3
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 0, 1):
        return np.pi*KAB*(-2*Dz**2*Q*Boys(0, u) - 2*Dz*Qz*alpha*Boys(1, u) + 2*Dz*Qz*beta*Boys(1, u) + P*(Boys(0, u) - Boys(1, u)) + 2*Qz**2*Boys(2, u))/P**3