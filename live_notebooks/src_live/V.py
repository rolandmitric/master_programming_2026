import numpy as np
from numba import njit
from .boys import Boys
@njit(cache = True, fastmath = True)
def V(i, j, k, l, m, n, Dx, Dy, Dz, KAB, P, Q, Qx, Qy, Qz, alpha, beta):
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 0, 0):
        return 2*np.pi*KAB*Boys(0, (Qx**2 + Qy**2 + Qz**2)/P)/P
    if (i, j, k, l, m, n) == (0, 0, 0, 1, 0, 0):
        return 2*np.pi*KAB*(Dx*alpha*Boys(0, (Qx**2 + Qy**2 + Qz**2)/P) - Qx*Boys(1, (Qx**2 + Qy**2 + Qz**2)/P))/P**2
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 1, 0):
        return 2*np.pi*KAB*(Dy*alpha*Boys(0, (Qx**2 + Qy**2 + Qz**2)/P) - Qy*Boys(1, (Qx**2 + Qy**2 + Qz**2)/P))/P**2
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 0, 1):
        return 2*np.pi*KAB*(Dz*alpha*Boys(0, (Qx**2 + Qy**2 + Qz**2)/P) - Qz*Boys(1, (Qx**2 + Qy**2 + Qz**2)/P))/P**2
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 0, 0):
        return -2*np.pi*KAB*(Dx*beta*Boys(0, (Qx**2 + Qy**2 + Qz**2)/P) + Qx*Boys(1, (Qx**2 + Qy**2 + Qz**2)/P))/P**2
    if (i, j, k, l, m, n) == (1, 0, 0, 1, 0, 0):
        return np.pi*KAB*(-2*Dx**2*Q*Boys(0, (Qx**2 + Qy**2 + Qz**2)/P) - 2*Dx*Qx*alpha*Boys(1, (Qx**2 + Qy**2 + Qz**2)/P) + 2*Dx*Qx*beta*Boys(1, (Qx**2 + Qy**2 + Qz**2)/P) + P*(Boys(0, (Qx**2 + Qy**2 + Qz**2)/P) - Boys(1, (Qx**2 + Qy**2 + Qz**2)/P)) + 2*Qx**2*Boys(2, (Qx**2 + Qy**2 + Qz**2)/P))/P**3
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 1, 0):
        return 2*np.pi*KAB*(-Dx*Dy*Q*Boys(0, (Qx**2 + Qy**2 + Qz**2)/P) + Dx*Qy*beta*Boys(1, (Qx**2 + Qy**2 + Qz**2)/P) - Dy*Qx*alpha*Boys(1, (Qx**2 + Qy**2 + Qz**2)/P) + Qx*Qy*Boys(2, (Qx**2 + Qy**2 + Qz**2)/P))/P**3
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 0, 1):
        return 2*np.pi*KAB*(-Dx*Dz*Q*Boys(0, (Qx**2 + Qy**2 + Qz**2)/P) + Dx*Qz*beta*Boys(1, (Qx**2 + Qy**2 + Qz**2)/P) - Dz*Qx*alpha*Boys(1, (Qx**2 + Qy**2 + Qz**2)/P) + Qx*Qz*Boys(2, (Qx**2 + Qy**2 + Qz**2)/P))/P**3
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 0, 0):
        return -2*np.pi*KAB*(Dy*beta*Boys(0, (Qx**2 + Qy**2 + Qz**2)/P) + Qy*Boys(1, (Qx**2 + Qy**2 + Qz**2)/P))/P**2
    if (i, j, k, l, m, n) == (0, 1, 0, 1, 0, 0):
        return 2*np.pi*KAB*(-Dx*Dy*Q*Boys(0, (Qx**2 + Qy**2 + Qz**2)/P) - Dx*Qy*alpha*Boys(1, (Qx**2 + Qy**2 + Qz**2)/P) + Dy*Qx*beta*Boys(1, (Qx**2 + Qy**2 + Qz**2)/P) + Qx*Qy*Boys(2, (Qx**2 + Qy**2 + Qz**2)/P))/P**3
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 1, 0):
        return np.pi*KAB*(-2*Dy**2*Q*Boys(0, (Qx**2 + Qy**2 + Qz**2)/P) - 2*Dy*Qy*alpha*Boys(1, (Qx**2 + Qy**2 + Qz**2)/P) + 2*Dy*Qy*beta*Boys(1, (Qx**2 + Qy**2 + Qz**2)/P) + P*(Boys(0, (Qx**2 + Qy**2 + Qz**2)/P) - Boys(1, (Qx**2 + Qy**2 + Qz**2)/P)) + 2*Qy**2*Boys(2, (Qx**2 + Qy**2 + Qz**2)/P))/P**3
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 0, 1):
        return 2*np.pi*KAB*(-Dy*Dz*Q*Boys(0, (Qx**2 + Qy**2 + Qz**2)/P) + Dy*Qz*beta*Boys(1, (Qx**2 + Qy**2 + Qz**2)/P) - Dz*Qy*alpha*Boys(1, (Qx**2 + Qy**2 + Qz**2)/P) + Qy*Qz*Boys(2, (Qx**2 + Qy**2 + Qz**2)/P))/P**3
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 0, 0):
        return -2*np.pi*KAB*(Dz*beta*Boys(0, (Qx**2 + Qy**2 + Qz**2)/P) + Qz*Boys(1, (Qx**2 + Qy**2 + Qz**2)/P))/P**2
    if (i, j, k, l, m, n) == (0, 0, 1, 1, 0, 0):
        return 2*np.pi*KAB*(-Dx*Dz*Q*Boys(0, (Qx**2 + Qy**2 + Qz**2)/P) - Dx*Qz*alpha*Boys(1, (Qx**2 + Qy**2 + Qz**2)/P) + Dz*Qx*beta*Boys(1, (Qx**2 + Qy**2 + Qz**2)/P) + Qx*Qz*Boys(2, (Qx**2 + Qy**2 + Qz**2)/P))/P**3
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 1, 0):
        return 2*np.pi*KAB*(-Dy*Dz*Q*Boys(0, (Qx**2 + Qy**2 + Qz**2)/P) - Dy*Qz*alpha*Boys(1, (Qx**2 + Qy**2 + Qz**2)/P) + Dz*Qy*beta*Boys(1, (Qx**2 + Qy**2 + Qz**2)/P) + Qy*Qz*Boys(2, (Qx**2 + Qy**2 + Qz**2)/P))/P**3
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 0, 1):
        return np.pi*KAB*(-2*Dz**2*Q*Boys(0, (Qx**2 + Qy**2 + Qz**2)/P) - 2*Dz*Qz*alpha*Boys(1, (Qx**2 + Qy**2 + Qz**2)/P) + 2*Dz*Qz*beta*Boys(1, (Qx**2 + Qy**2 + Qz**2)/P) + P*(Boys(0, (Qx**2 + Qy**2 + Qz**2)/P) - Boys(1, (Qx**2 + Qy**2 + Qz**2)/P)) + 2*Qz**2*Boys(2, (Qx**2 + Qy**2 + Qz**2)/P))/P**3