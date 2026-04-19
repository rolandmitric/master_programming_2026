import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def V(i, j, k, l, m, n, ABx, ABy, ABz,p,alpha, beta,Qx, Qy, Qz,F0,F1,F2,KAB):    
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 0, 0):
        return 2*np.pi*KAB*F0/p
    if (i, j, k, l, m, n) == (0, 0, 0, 1, 0, 0):
        return 2*np.pi*KAB*(ABx*alpha*F0 - Qx*F1)/p**2
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 1, 0):
        return 2*np.pi*KAB*(ABy*alpha*F0 - Qy*F1)/p**2
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 0, 1):
        return 2*np.pi*KAB*(ABz*alpha*F0 - Qz*F1)/p**2
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 0, 0):
        return -2*np.pi*KAB*(ABx*beta*F0 + Qx*F1)/p**2
    if (i, j, k, l, m, n) == (1, 0, 0, 1, 0, 0):
        t0 = 2*alpha
        t1 = F1
        t2 = ABx*Qx*t1
        t3 = F0
        return np.pi*KAB*(-ABx**2*beta*t0*t3 + 2*Qx**2*F2 + 2*beta*t2 + p*(-t1 + t3) - t0*t2)/p**3
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 1, 0):
        t0 = F1
        t1 = ABy*alpha
        return 2*np.pi*KAB*(ABx*Qy*beta*t0 - ABx*beta*t1*F0 + Qx*Qy*F2 - Qx*t0*t1)/p**3
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 0, 1):
        t0 = F1
        t1 = ABz*alpha
        return 2*np.pi*KAB*(ABx*Qz*beta*t0 - ABx*beta*t1*F0 + Qx*Qz*F2 - Qx*t0*t1)/p**3
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 0, 0):
        return -2*np.pi*KAB*(ABy*beta*F0 + Qy*F1)/p**2
    if (i, j, k, l, m, n) == (0, 1, 0, 1, 0, 0):
        t0 = F1
        t1 = ABx*alpha
        return 2*np.pi*KAB*(ABy*Qx*beta*t0 - ABy*beta*t1*F0 + Qx*Qy*F2 - Qy*t0*t1)/p**3
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 1, 0):
        t0 = 2*alpha
        t1 = F1
        t2 = ABy*Qy*t1
        t3 = F0
        return np.pi*KAB*(-ABy**2*beta*t0*t3 + 2*Qy**2*F2 + 2*beta*t2 + p*(-t1 + t3) - t0*t2)/p**3
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 0, 1):
        t0 = F1
        t1 = ABz*alpha
        return 2*np.pi*KAB*(ABy*Qz*beta*t0 - ABy*beta*t1*F0 + Qy*Qz*F2 - Qy*t0*t1)/p**3
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 0, 0):
        return -2*np.pi*KAB*(ABz*beta*F0 + Qz*F1)/p**2
    if (i, j, k, l, m, n) == (0, 0, 1, 1, 0, 0):
        t0 = F1
        t1 = ABx*alpha
        return 2*np.pi*KAB*(ABz*Qx*beta*t0 - ABz*beta*t1*F0 + Qx*Qz*F2 - Qz*t0*t1)/p**3
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 1, 0):
        t0 = F1
        t1 = ABy*alpha
        return 2*np.pi*KAB*(ABz*Qy*beta*t0 - ABz*beta*t1*F0 + Qy*Qz*F2 - Qz*t0*t1)/p**3
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 0, 1):
        t0 = 2*alpha
        t1 = F1
        t2 = ABz*Qz*t1
        t3 = F0
        return np.pi*KAB*(-ABz**2*beta*t0*t3 + 2*Qz**2*F2 + 2*beta*t2 + p*(-t1 + t3) - t0*t2)/p**3
    raise KeyError((i, j, k, l, m, n))
