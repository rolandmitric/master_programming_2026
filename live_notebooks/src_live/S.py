import numpy as np
from numba import njit
@njit(cache=True, fastmath=True)
def S(rep, Dx, Dy, Dz, P, Q, RAB2, alpha, beta):
    if rep == 0:
        return np.pi**(3/2)*np.exp(-Q*(Dx**2 + Dy**2 + Dz**2)/P)/P**(3/2)
    if rep == 1:
        return np.pi**(3/2)*Dz*alpha*np.exp(-Q*(Dx**2 + Dy**2 + Dz**2)/P)/P**(5/2)
    if rep == 2:
        return (1/2)*np.pi**(3/2)*(P**2 - alpha*(-2*Dz**2*Q + P))*np.exp(-Q*(Dx**2 + Dy**2 + Dz**2)/P)/(P**(7/2)*beta)
    if rep == 3:
        return np.pi**(3/2)*Dy*Dz*alpha**2*np.exp(-Q*(Dx**2 + Dy**2 + Dz**2)/P)/P**(7/2)
    if rep == 4:
        return (1/2)*np.pi**(3/2)*(-2*Dz**2*Q + P)*np.exp(-Q*(Dx**2 + Dy**2 + Dz**2)/P)/P**(7/2)
    if rep == 5:
        return (1/2)*np.pi**(3/2)*(-Dz*P**2 + Dz*alpha*(-2*Dz**2*Q + 3*alpha + 3*beta))*np.exp(-Q*(Dx**2 + Dy**2 + Dz**2)/P)/P**(9/2)
    if rep == 6:
        return -np.pi**(3/2)*Dy*Dz*Q*np.exp(-Q*(Dx**2 + Dy**2 + Dz**2)/P)/P**(7/2)
    if rep == 7:
        return (1/2)*np.pi**(3/2)*Dy*alpha*(-2*Dz**2*Q + P)*np.exp(-Q*(Dx**2 + Dy**2 + Dz**2)/P)/P**(9/2)
    if rep == 8:
        return (1/2)*np.pi**(3/2)*(-Dz*P**2 - Dz*alpha*(2*Dy**2*Q - P))*np.exp(-Q*(Dx**2 + Dy**2 + Dz**2)/P)/P**(9/2)
    if rep == 9:
        return -np.pi**(3/2)*Dx*Dy*Dz*Q*alpha*np.exp(-Q*(Dx**2 + Dy**2 + Dz**2)/P)/P**(9/2)
    if rep == 10:
        return (1/4)*np.pi**(3/2)*(P**4 - P**2*alpha*(-2*Dz**2*Q + P) - P**2*beta*(-2*Dz**2*Q + P) + Q*(4*Dz**4*Q**2 - 12*Dz**2*P*Q + 3*P**2))*np.exp(-Q*(Dx**2 + Dy**2 + Dz**2)/P)/(P**(11/2)*Q)
    if rep == 11:
        return (1/2)*np.pi**(3/2)*Dy*Dz*alpha*(P**2 - beta*(-2*Dz**2*Q + 3*alpha + 3*beta))*np.exp(-Q*(Dx**2 + Dy**2 + Dz**2)/P)/P**(11/2)
    if rep == 12:
        return (1/4)*np.pi**(3/2)*(P**4 - P**2*alpha*(-2*Dy**2*Q + P) - P**2*beta*(-2*Dz**2*Q + P) + Q*(4*Dy**2*Dz**2*Q**2 + P**2 - 2*P*Q*(Dy**2 + Dz**2)))*np.exp(-Q*(Dx**2 + Dy**2 + Dz**2)/P)/(P**(11/2)*Q)
    if rep == 13:
        return (1/2)*np.pi**(3/2)*Dx*Dy*alpha*(P**2 - beta*(-2*Dz**2*Q + P))*np.exp(-Q*(Dx**2 + Dy**2 + Dz**2)/P)/P**(11/2)
    if rep == 14:
        return np.pi**(3/2)*(Dy**2*Dz**2*Q**2 + (1/4)*P**2 - 1/2*P*Q*(Dy**2 + Dz**2))*np.exp(-Q*(Dx**2 + Dy**2 + Dz**2)/P)/P**(11/2)
    if rep == 15:
        return -1/2*np.pi**(3/2)*Dx*Dy*Q*(-2*Dz**2*Q + P)*np.exp(-Q*(Dx**2 + Dy**2 + Dz**2)/P)/P**(11/2)
    raise ValueError("Invalid overlap representative")
