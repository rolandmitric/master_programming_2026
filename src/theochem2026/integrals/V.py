import numpy as np
from numba import njit
@njit(cache=True, fastmath=True)
def V(i, j, k, l, m, n, ABx, ABy, ABz,p,alpha, beta,Qx, Qy, Qz,u,KAB):    
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 0, 0):
        return 2*np.pi*KAB*Boys(0, u)/p
    if (i, j, k, l, m, n) == (0, 0, 0, 1, 0, 0):
        return 2*np.pi*KAB*(ABx*alpha*Boys(0, u) - Qx*Boys(1, u))/p**2
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 1, 0):
        return 2*np.pi*KAB*(ABy*alpha*Boys(0, u) - Qy*Boys(1, u))/p**2
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 0, 1):
        return 2*np.pi*KAB*(ABz*alpha*Boys(0, u) - Qz*Boys(1, u))/p**2
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 0, 0):
        return -2*np.pi*KAB*(ABx*beta*Boys(0, u) + Qx*Boys(1, u))/p**2
    if (i, j, k, l, m, n) == (1, 0, 0, 1, 0, 0):
        t0 = 2*alpha
        t1 = Boys(1, u)
        t2 = ABx*Qx*t1
        t3 = Boys(0, u)
        return np.pi*KAB*(-ABx**2*beta*t0*t3 + 2*Qx**2*Boys(2, u) + 2*beta*t2 + p*(-t1 + t3) - t0*t2)/p**3
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 1, 0):
        t0 = Boys(1, u)
        t1 = ABy*alpha
        return 2*np.pi*KAB*(ABx*Qy*beta*t0 - ABx*beta*t1*Boys(0, u) + Qx*Qy*Boys(2, u) - Qx*t0*t1)/p**3
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 0, 1):
        t0 = Boys(1, u)
        t1 = ABz*alpha
        return 2*np.pi*KAB*(ABx*Qz*beta*t0 - ABx*beta*t1*Boys(0, u) + Qx*Qz*Boys(2, u) - Qx*t0*t1)/p**3
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 0, 0):
        return -2*np.pi*KAB*(ABy*beta*Boys(0, u) + Qy*Boys(1, u))/p**2
    if (i, j, k, l, m, n) == (0, 1, 0, 1, 0, 0):
        t0 = Boys(1, u)
        t1 = ABx*alpha
        return 2*np.pi*KAB*(ABy*Qx*beta*t0 - ABy*beta*t1*Boys(0, u) + Qx*Qy*Boys(2, u) - Qy*t0*t1)/p**3
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 1, 0):
        t0 = 2*alpha
        t1 = Boys(1, u)
        t2 = ABy*Qy*t1
        t3 = Boys(0, u)
        return np.pi*KAB*(-ABy**2*beta*t0*t3 + 2*Qy**2*Boys(2, u) + 2*beta*t2 + p*(-t1 + t3) - t0*t2)/p**3
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 0, 1):
        t0 = Boys(1, u)
        t1 = ABz*alpha
        return 2*np.pi*KAB*(ABy*Qz*beta*t0 - ABy*beta*t1*Boys(0, u) + Qy*Qz*Boys(2, u) - Qy*t0*t1)/p**3
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 0, 0):
        return -2*np.pi*KAB*(ABz*beta*Boys(0, u) + Qz*Boys(1, u))/p**2
    if (i, j, k, l, m, n) == (0, 0, 1, 1, 0, 0):
        t0 = Boys(1, u)
        t1 = ABx*alpha
        return 2*np.pi*KAB*(ABz*Qx*beta*t0 - ABz*beta*t1*Boys(0, u) + Qx*Qz*Boys(2, u) - Qz*t0*t1)/p**3
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 1, 0):
        t0 = Boys(1, u)
        t1 = ABy*alpha
        return 2*np.pi*KAB*(ABz*Qy*beta*t0 - ABz*beta*t1*Boys(0, u) + Qy*Qz*Boys(2, u) - Qz*t0*t1)/p**3
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 0, 1):
        t0 = 2*alpha
        t1 = Boys(1, u)
        t2 = ABz*Qz*t1
        t3 = Boys(0, u)
        return np.pi*KAB*(-ABz**2*beta*t0*t3 + 2*Qz**2*Boys(2, u) + 2*beta*t2 + p*(-t1 + t3) - t0*t2)/p**3
    raise KeyError((i, j, k, l, m, n))
