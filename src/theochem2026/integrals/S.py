import numpy as np
from numba import njit
@njit(cache=True, fastmath=True)
def S(i, j, k, l, m, n, Dx, Dy, Dz, D2, AB_sum, AB_product,alpha, beta):
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 0, 0):
        return np.pi**(3/2)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(3/2)
    if (i, j, k, l, m, n) == (0, 0, 0, 1, 0, 0):
        return np.pi**(3/2)*Dx*alpha*np.exp(-AB_product*D2/AB_sum)/AB_sum**(5/2)
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 1, 0):
        return np.pi**(3/2)*Dy*alpha*np.exp(-AB_product*D2/AB_sum)/AB_sum**(5/2)
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 0, 1):
        return np.pi**(3/2)*Dz*alpha*np.exp(-AB_product*D2/AB_sum)/AB_sum**(5/2)
    if (i, j, k, l, m, n) == (0, 0, 0, 2, 0, 0):
        return (1/2)*np.pi**(3/2)*(AB_sum**2 + alpha*(2*AB_product*Dx**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(7/2)*beta)
    if (i, j, k, l, m, n) == (0, 0, 0, 1, 1, 0):
        return np.pi**(3/2)*Dx*Dy*alpha**2*np.exp(-AB_product*D2/AB_sum)/AB_sum**(7/2)
    if (i, j, k, l, m, n) == (0, 0, 0, 1, 0, 1):
        return np.pi**(3/2)*Dx*Dz*alpha**2*np.exp(-AB_product*D2/AB_sum)/AB_sum**(7/2)
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 2, 0):
        return (1/2)*np.pi**(3/2)*(AB_sum**2 + alpha*(2*AB_product*Dy**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(7/2)*beta)
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 1, 1):
        return np.pi**(3/2)*Dy*Dz*alpha**2*np.exp(-AB_product*D2/AB_sum)/AB_sum**(7/2)
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 0, 2):
        return (1/2)*np.pi**(3/2)*(AB_sum**2 + alpha*(2*AB_product*Dz**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(7/2)*beta)
    if (i, j, k, l, m, n) == (0, 0, 0, 3, 0, 0):
        return (1/2)*np.pi**(3/2)*Dx*alpha*(3*AB_sum**2 - alpha*(-2*AB_product*Dx**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(9/2)*beta)
    if (i, j, k, l, m, n) == (0, 0, 0, 2, 1, 0):
        return (1/2)*np.pi**(3/2)*Dy*alpha*(AB_sum**2 + alpha*(2*AB_product*Dx**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(9/2)*beta)
    if (i, j, k, l, m, n) == (0, 0, 0, 2, 0, 1):
        return (1/2)*np.pi**(3/2)*Dz*alpha*(AB_sum**2 + alpha*(2*AB_product*Dx**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(9/2)*beta)
    if (i, j, k, l, m, n) == (0, 0, 0, 1, 2, 0):
        return (1/2)*np.pi**(3/2)*Dx*alpha*(AB_sum**2 + alpha*(2*AB_product*Dy**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(9/2)*beta)
    if (i, j, k, l, m, n) == (0, 0, 0, 1, 1, 1):
        return np.pi**(3/2)*Dx*Dy*Dz*alpha**3*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (0, 0, 0, 1, 0, 2):
        return (1/2)*np.pi**(3/2)*Dx*alpha*(AB_sum**2 + alpha*(2*AB_product*Dz**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(9/2)*beta)
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 3, 0):
        return (1/2)*np.pi**(3/2)*Dy*alpha*(3*AB_sum**2 - alpha*(-2*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(9/2)*beta)
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 2, 1):
        return (1/2)*np.pi**(3/2)*Dz*alpha*(AB_sum**2 + alpha*(2*AB_product*Dy**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(9/2)*beta)
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 1, 2):
        return (1/2)*np.pi**(3/2)*Dy*alpha*(AB_sum**2 + alpha*(2*AB_product*Dz**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(9/2)*beta)
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 0, 3):
        return (1/2)*np.pi**(3/2)*Dz*alpha*(3*AB_sum**2 - alpha*(-2*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(9/2)*beta)
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 0, 0):
        return -np.pi**(3/2)*Dx*beta*np.exp(-AB_product*D2/AB_sum)/AB_sum**(5/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 1, 0, 0):
        return np.pi**(3/2)*(-AB_product*Dx**2 + (1/2)*AB_sum)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(7/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 1, 0):
        return -np.pi**(3/2)*AB_product*Dx*Dy*np.exp(-AB_product*D2/AB_sum)/AB_sum**(7/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 0, 1):
        return -np.pi**(3/2)*AB_product*Dx*Dz*np.exp(-AB_product*D2/AB_sum)/AB_sum**(7/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 2, 0, 0):
        return (1/2)*np.pi**(3/2)*Dx*(-AB_sum**2 + alpha*(-2*AB_product*Dx**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 1, 1, 0):
        return (1/2)*np.pi**(3/2)*Dy*alpha*(-2*AB_product*Dx**2 + AB_sum)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 1, 0, 1):
        return (1/2)*np.pi**(3/2)*Dz*alpha*(-2*AB_product*Dx**2 + AB_sum)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 2, 0):
        return (1/2)*np.pi**(3/2)*Dx*(-AB_sum**2 - alpha*(2*AB_product*Dy**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 1, 1):
        return -np.pi**(3/2)*AB_product*Dx*Dy*Dz*alpha*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 0, 2):
        return (1/2)*np.pi**(3/2)*Dx*(-AB_sum**2 - alpha*(2*AB_product*Dz**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 3, 0, 0):
        return (1/4)*np.pi**(3/2)*(-3*AB_sum**2*(2*AB_product*Dx**2 - AB_sum) - alpha*(4*AB_product**2*Dx**4 - 12*AB_product*AB_sum*Dx**2 + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(11/2)*beta)
    if (i, j, k, l, m, n) == (1, 0, 0, 2, 1, 0):
        return -1/2*np.pi**(3/2)*Dx*Dy*alpha*(AB_sum**2 - alpha*(-2*AB_product*Dx**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 2, 0, 1):
        return -1/2*np.pi**(3/2)*Dx*Dz*alpha*(AB_sum**2 - alpha*(-2*AB_product*Dx**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 1, 2, 0):
        return (1/4)*np.pi**(3/2)*(-AB_sum**2*(2*AB_product*Dx**2 - AB_sum) - alpha*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + Dy**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(11/2)*beta)
    if (i, j, k, l, m, n) == (1, 0, 0, 1, 1, 1):
        return (1/2)*np.pi**(3/2)*Dy*Dz*alpha**2*(-2*AB_product*Dx**2 + AB_sum)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 1, 0, 2):
        return (1/4)*np.pi**(3/2)*(-AB_sum**2*(2*AB_product*Dx**2 - AB_sum) - alpha*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + Dz**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(11/2)*beta)
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 3, 0):
        return -1/2*np.pi**(3/2)*Dx*Dy*alpha*(3*AB_sum**2 - alpha*(-2*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 2, 1):
        return -1/2*np.pi**(3/2)*Dx*Dz*alpha*(AB_sum**2 + alpha*(2*AB_product*Dy**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 1, 2):
        return -1/2*np.pi**(3/2)*Dx*Dy*alpha*(AB_sum**2 + alpha*(2*AB_product*Dz**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 0, 3):
        return -1/2*np.pi**(3/2)*Dx*Dz*alpha*(3*AB_sum**2 - alpha*(-2*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 0, 0):
        return -np.pi**(3/2)*Dy*beta*np.exp(-AB_product*D2/AB_sum)/AB_sum**(5/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 1, 0, 0):
        return -np.pi**(3/2)*AB_product*Dx*Dy*np.exp(-AB_product*D2/AB_sum)/AB_sum**(7/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 1, 0):
        return np.pi**(3/2)*(-AB_product*Dy**2 + (1/2)*AB_sum)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(7/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 0, 1):
        return -np.pi**(3/2)*AB_product*Dy*Dz*np.exp(-AB_product*D2/AB_sum)/AB_sum**(7/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 2, 0, 0):
        return (1/2)*np.pi**(3/2)*Dy*(-AB_sum**2 - alpha*(2*AB_product*Dx**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 1, 1, 0):
        return (1/2)*np.pi**(3/2)*Dx*alpha*(-2*AB_product*Dy**2 + AB_sum)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 1, 0, 1):
        return -np.pi**(3/2)*AB_product*Dx*Dy*Dz*alpha*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 2, 0):
        return (1/2)*np.pi**(3/2)*Dy*(-AB_sum**2 + alpha*(-2*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 1, 1):
        return (1/2)*np.pi**(3/2)*Dz*alpha*(-2*AB_product*Dy**2 + AB_sum)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 0, 2):
        return (1/2)*np.pi**(3/2)*Dy*(-AB_sum**2 - alpha*(2*AB_product*Dz**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 3, 0, 0):
        return -1/2*np.pi**(3/2)*Dx*Dy*alpha*(3*AB_sum**2 - alpha*(-2*AB_product*Dx**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 2, 1, 0):
        return (1/4)*np.pi**(3/2)*(-AB_sum**2*(2*AB_product*Dy**2 - AB_sum) - alpha*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + Dy**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(11/2)*beta)
    if (i, j, k, l, m, n) == (0, 1, 0, 2, 0, 1):
        return -1/2*np.pi**(3/2)*Dy*Dz*alpha*(AB_sum**2 + alpha*(2*AB_product*Dx**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 1, 2, 0):
        return -1/2*np.pi**(3/2)*Dx*Dy*alpha*(AB_sum**2 - alpha*(-2*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 1, 1, 1):
        return (1/2)*np.pi**(3/2)*Dx*Dz*alpha**2*(-2*AB_product*Dy**2 + AB_sum)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 1, 0, 2):
        return -1/2*np.pi**(3/2)*Dx*Dy*alpha*(AB_sum**2 + alpha*(2*AB_product*Dz**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 3, 0):
        return (1/4)*np.pi**(3/2)*(-3*AB_sum**2*(2*AB_product*Dy**2 - AB_sum) - alpha*(4*AB_product**2*Dy**4 - 12*AB_product*AB_sum*Dy**2 + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(11/2)*beta)
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 2, 1):
        return -1/2*np.pi**(3/2)*Dy*Dz*alpha*(AB_sum**2 - alpha*(-2*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 1, 2):
        return (1/4)*np.pi**(3/2)*(-AB_sum**2*(2*AB_product*Dy**2 - AB_sum) - alpha*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + Dz**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(11/2)*beta)
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 0, 3):
        return -1/2*np.pi**(3/2)*Dy*Dz*alpha*(3*AB_sum**2 - alpha*(-2*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 0, 0):
        return -np.pi**(3/2)*Dz*beta*np.exp(-AB_product*D2/AB_sum)/AB_sum**(5/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 1, 0, 0):
        return -np.pi**(3/2)*AB_product*Dx*Dz*np.exp(-AB_product*D2/AB_sum)/AB_sum**(7/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 1, 0):
        return -np.pi**(3/2)*AB_product*Dy*Dz*np.exp(-AB_product*D2/AB_sum)/AB_sum**(7/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 0, 1):
        return np.pi**(3/2)*(-AB_product*Dz**2 + (1/2)*AB_sum)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(7/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 2, 0, 0):
        return (1/2)*np.pi**(3/2)*Dz*(-AB_sum**2 - alpha*(2*AB_product*Dx**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 1, 1, 0):
        return -np.pi**(3/2)*AB_product*Dx*Dy*Dz*alpha*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 1, 0, 1):
        return (1/2)*np.pi**(3/2)*Dx*alpha*(-2*AB_product*Dz**2 + AB_sum)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 2, 0):
        return (1/2)*np.pi**(3/2)*Dz*(-AB_sum**2 - alpha*(2*AB_product*Dy**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 1, 1):
        return (1/2)*np.pi**(3/2)*Dy*alpha*(-2*AB_product*Dz**2 + AB_sum)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 0, 2):
        return (1/2)*np.pi**(3/2)*Dz*(-AB_sum**2 + alpha*(-2*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 3, 0, 0):
        return -1/2*np.pi**(3/2)*Dx*Dz*alpha*(3*AB_sum**2 - alpha*(-2*AB_product*Dx**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 2, 1, 0):
        return -1/2*np.pi**(3/2)*Dy*Dz*alpha*(AB_sum**2 + alpha*(2*AB_product*Dx**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 2, 0, 1):
        return (1/4)*np.pi**(3/2)*(-AB_sum**2*(2*AB_product*Dz**2 - AB_sum) - alpha*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + Dz**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(11/2)*beta)
    if (i, j, k, l, m, n) == (0, 0, 1, 1, 2, 0):
        return -1/2*np.pi**(3/2)*Dx*Dz*alpha*(AB_sum**2 + alpha*(2*AB_product*Dy**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 1, 1, 1):
        return (1/2)*np.pi**(3/2)*Dx*Dy*alpha**2*(-2*AB_product*Dz**2 + AB_sum)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 1, 0, 2):
        return -1/2*np.pi**(3/2)*Dx*Dz*alpha*(AB_sum**2 - alpha*(-2*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 3, 0):
        return -1/2*np.pi**(3/2)*Dy*Dz*alpha*(3*AB_sum**2 - alpha*(-2*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 2, 1):
        return (1/4)*np.pi**(3/2)*(-AB_sum**2*(2*AB_product*Dz**2 - AB_sum) - alpha*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + Dz**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(11/2)*beta)
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 1, 2):
        return -1/2*np.pi**(3/2)*Dy*Dz*alpha*(AB_sum**2 - alpha*(-2*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 0, 3):
        return (1/4)*np.pi**(3/2)*(-3*AB_sum**2*(2*AB_product*Dz**2 - AB_sum) - alpha*(4*AB_product**2*Dz**4 - 12*AB_product*AB_sum*Dz**2 + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(11/2)*beta)
    if (i, j, k, l, m, n) == (2, 0, 0, 0, 0, 0):
        return (1/2)*np.pi**(3/2)*(AB_sum**2 + beta*(2*AB_product*Dx**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(7/2)*alpha)
    if (i, j, k, l, m, n) == (2, 0, 0, 1, 0, 0):
        return (1/2)*np.pi**(3/2)*Dx*(AB_sum**2 - beta*(-2*AB_product*Dx**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (2, 0, 0, 0, 1, 0):
        return (1/2)*np.pi**(3/2)*Dy*(AB_sum**2 + beta*(2*AB_product*Dx**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (2, 0, 0, 0, 0, 1):
        return (1/2)*np.pi**(3/2)*Dz*(AB_sum**2 + beta*(2*AB_product*Dx**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (2, 0, 0, 2, 0, 0):
        return (1/4)*np.pi**(3/2)*(AB_product*(4*AB_product**2*Dx**4 - 12*AB_product*AB_sum*Dx**2 + 3*AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dx**2 - AB_sum) + AB_sum**2*beta*(2*AB_product*Dx**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(11/2))
    if (i, j, k, l, m, n) == (2, 0, 0, 1, 1, 0):
        return (1/2)*np.pi**(3/2)*Dx*Dy*alpha*(AB_sum**2 - beta*(-2*AB_product*Dx**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (2, 0, 0, 1, 0, 1):
        return (1/2)*np.pi**(3/2)*Dx*Dz*alpha*(AB_sum**2 - beta*(-2*AB_product*Dx**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (2, 0, 0, 0, 2, 0):
        return (1/4)*np.pi**(3/2)*(AB_product*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + Dy**2) + AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dy**2 - AB_sum) + AB_sum**2*beta*(2*AB_product*Dx**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(11/2))
    if (i, j, k, l, m, n) == (2, 0, 0, 0, 1, 1):
        return (1/2)*np.pi**(3/2)*Dy*Dz*alpha*(AB_sum**2 + beta*(2*AB_product*Dx**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (2, 0, 0, 0, 0, 2):
        return (1/4)*np.pi**(3/2)*(AB_product*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + Dz**2) + AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dz**2 - AB_sum) + AB_sum**2*beta*(2*AB_product*Dx**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(11/2))
    if (i, j, k, l, m, n) == (2, 0, 0, 3, 0, 0):
        return (1/4)*np.pi**(3/2)*Dx*(AB_product*(4*AB_product**2*Dx**4 - 20*AB_product*AB_sum*Dx**2 + 15*AB_sum**2) + 3*AB_sum**4 - AB_sum**2*alpha*(-2*AB_product*Dx**2 + 3*alpha + 3*beta) - 3*AB_sum**2*beta*(-2*AB_product*Dx**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*beta)
    if (i, j, k, l, m, n) == (2, 0, 0, 2, 1, 0):
        return (1/4)*np.pi**(3/2)*Dy*(AB_product*(4*AB_product**2*Dx**4 - 12*AB_product*AB_sum*Dx**2 + 3*AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dx**2 - AB_sum) + AB_sum**2*beta*(2*AB_product*Dx**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*beta)
    if (i, j, k, l, m, n) == (2, 0, 0, 2, 0, 1):
        return (1/4)*np.pi**(3/2)*Dz*(AB_product*(4*AB_product**2*Dx**4 - 12*AB_product*AB_sum*Dx**2 + 3*AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dx**2 - AB_sum) + AB_sum**2*beta*(2*AB_product*Dx**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*beta)
    if (i, j, k, l, m, n) == (2, 0, 0, 1, 2, 0):
        return (1/4)*np.pi**(3/2)*Dx*(AB_product*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + 3*Dy**2) + 3*AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dy**2 - AB_sum) - AB_sum**2*beta*(-2*AB_product*Dx**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*beta)
    if (i, j, k, l, m, n) == (2, 0, 0, 1, 1, 1):
        return (1/2)*np.pi**(3/2)*Dx*Dy*Dz*alpha**2*(AB_sum**2 - beta*(-2*AB_product*Dx**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (2, 0, 0, 1, 0, 2):
        return (1/4)*np.pi**(3/2)*Dx*(AB_product*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + 3*Dz**2) + 3*AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dz**2 - AB_sum) - AB_sum**2*beta*(-2*AB_product*Dx**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*beta)
    if (i, j, k, l, m, n) == (2, 0, 0, 0, 3, 0):
        return (1/4)*np.pi**(3/2)*Dy*(AB_product*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(3*Dx**2 + Dy**2) + 3*AB_sum**2) + 3*AB_sum**4 - AB_sum**2*alpha*(-2*AB_product*Dy**2 + 3*alpha + 3*beta) + 3*AB_sum**2*beta*(2*AB_product*Dx**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*beta)
    if (i, j, k, l, m, n) == (2, 0, 0, 0, 2, 1):
        return (1/4)*np.pi**(3/2)*Dz*(AB_product*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + Dy**2) + AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dy**2 - AB_sum) + AB_sum**2*beta*(2*AB_product*Dx**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*beta)
    if (i, j, k, l, m, n) == (2, 0, 0, 0, 1, 2):
        return (1/4)*np.pi**(3/2)*Dy*(AB_product*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + Dz**2) + AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dz**2 - AB_sum) + AB_sum**2*beta*(2*AB_product*Dx**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*beta)
    if (i, j, k, l, m, n) == (2, 0, 0, 0, 0, 3):
        return (1/4)*np.pi**(3/2)*Dz*(AB_product*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(3*Dx**2 + Dz**2) + 3*AB_sum**2) + 3*AB_sum**4 - AB_sum**2*alpha*(-2*AB_product*Dz**2 + 3*alpha + 3*beta) + 3*AB_sum**2*beta*(2*AB_product*Dx**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*beta)
    if (i, j, k, l, m, n) == (1, 1, 0, 0, 0, 0):
        return np.pi**(3/2)*Dx*Dy*beta**2*np.exp(-AB_product*D2/AB_sum)/AB_sum**(7/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 1, 0, 0):
        return (1/2)*np.pi**(3/2)*Dy*beta*(2*AB_product*Dx**2 - AB_sum)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 0, 1, 0):
        return (1/2)*np.pi**(3/2)*Dx*beta*(2*AB_product*Dy**2 - AB_sum)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 0, 0, 1):
        return np.pi**(3/2)*AB_product*Dx*Dy*Dz*beta*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 2, 0, 0):
        return (1/2)*np.pi**(3/2)*Dx*Dy*beta*(AB_sum**2 - alpha*(-2*AB_product*Dx**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 1, 1, 0):
        return np.pi**(3/2)*(AB_product**2*Dx**2*Dy**2 - 1/2*AB_product*AB_sum*(Dx**2 + Dy**2) + (1/4)*AB_sum**2)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 1, 0, 1):
        return (1/2)*np.pi**(3/2)*AB_product*Dy*Dz*(2*AB_product*Dx**2 - AB_sum)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 0, 2, 0):
        return (1/2)*np.pi**(3/2)*Dx*Dy*beta*(AB_sum**2 - alpha*(-2*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 0, 1, 1):
        return (1/2)*np.pi**(3/2)*AB_product*Dx*Dz*(2*AB_product*Dy**2 - AB_sum)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 0, 0, 2):
        return (1/2)*np.pi**(3/2)*Dx*Dy*beta*(AB_sum**2 + alpha*(2*AB_product*Dz**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 3, 0, 0):
        return (1/4)*np.pi**(3/2)*Dy*(3*AB_sum**2*(2*AB_product*Dx**2 - AB_sum) + alpha*(4*AB_product**2*Dx**4 - 12*AB_product*AB_sum*Dx**2 + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 2, 1, 0):
        return (1/4)*np.pi**(3/2)*Dx*(AB_sum**2*(2*AB_product*Dy**2 - AB_sum) + alpha*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + 3*Dy**2) + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 2, 0, 1):
        return (1/2)*np.pi**(3/2)*AB_product*Dx*Dy*Dz*(AB_sum**2 - alpha*(-2*AB_product*Dx**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 1, 2, 0):
        return (1/4)*np.pi**(3/2)*Dy*(AB_sum**2*(2*AB_product*Dx**2 - AB_sum) + alpha*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(3*Dx**2 + Dy**2) + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 1, 1, 1):
        return (1/4)*np.pi**(3/2)*Dz*alpha*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + Dy**2) + AB_sum**2)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 1, 0, 2):
        return (1/4)*np.pi**(3/2)*Dy*(AB_sum**2*(2*AB_product*Dx**2 - AB_sum) + alpha*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + Dz**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 0, 3, 0):
        return (1/4)*np.pi**(3/2)*Dx*(3*AB_sum**2*(2*AB_product*Dy**2 - AB_sum) + alpha*(4*AB_product**2*Dy**4 - 12*AB_product*AB_sum*Dy**2 + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 0, 2, 1):
        return (1/2)*np.pi**(3/2)*AB_product*Dx*Dy*Dz*(AB_sum**2 - alpha*(-2*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 0, 1, 2):
        return (1/4)*np.pi**(3/2)*Dx*(AB_sum**2*(2*AB_product*Dy**2 - AB_sum) + alpha*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + Dz**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 0, 0, 3):
        return (1/2)*np.pi**(3/2)*AB_product*Dx*Dy*Dz*(3*AB_sum**2 - alpha*(-2*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 0, 0, 0):
        return np.pi**(3/2)*Dx*Dz*beta**2*np.exp(-AB_product*D2/AB_sum)/AB_sum**(7/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 1, 0, 0):
        return (1/2)*np.pi**(3/2)*Dz*beta*(2*AB_product*Dx**2 - AB_sum)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 0, 1, 0):
        return np.pi**(3/2)*AB_product*Dx*Dy*Dz*beta*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 0, 0, 1):
        return (1/2)*np.pi**(3/2)*Dx*beta*(2*AB_product*Dz**2 - AB_sum)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 2, 0, 0):
        return (1/2)*np.pi**(3/2)*Dx*Dz*beta*(AB_sum**2 - alpha*(-2*AB_product*Dx**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 1, 1, 0):
        return (1/2)*np.pi**(3/2)*AB_product*Dy*Dz*(2*AB_product*Dx**2 - AB_sum)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 1, 0, 1):
        return np.pi**(3/2)*(AB_product**2*Dx**2*Dz**2 - 1/2*AB_product*AB_sum*(Dx**2 + Dz**2) + (1/4)*AB_sum**2)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 0, 2, 0):
        return (1/2)*np.pi**(3/2)*Dx*Dz*beta*(AB_sum**2 + alpha*(2*AB_product*Dy**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 0, 1, 1):
        return (1/2)*np.pi**(3/2)*AB_product*Dx*Dy*(2*AB_product*Dz**2 - AB_sum)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 0, 0, 2):
        return (1/2)*np.pi**(3/2)*Dx*Dz*beta*(AB_sum**2 - alpha*(-2*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 3, 0, 0):
        return (1/4)*np.pi**(3/2)*Dz*(3*AB_sum**2*(2*AB_product*Dx**2 - AB_sum) + alpha*(4*AB_product**2*Dx**4 - 12*AB_product*AB_sum*Dx**2 + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 2, 1, 0):
        return (1/2)*np.pi**(3/2)*AB_product*Dx*Dy*Dz*(AB_sum**2 - alpha*(-2*AB_product*Dx**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 2, 0, 1):
        return (1/4)*np.pi**(3/2)*Dx*(AB_sum**2*(2*AB_product*Dz**2 - AB_sum) + alpha*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + 3*Dz**2) + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 1, 2, 0):
        return (1/4)*np.pi**(3/2)*Dz*(AB_sum**2*(2*AB_product*Dx**2 - AB_sum) + alpha*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + Dy**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 1, 1, 1):
        return (1/4)*np.pi**(3/2)*Dy*alpha*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + Dz**2) + AB_sum**2)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 1, 0, 2):
        return (1/4)*np.pi**(3/2)*Dz*(AB_sum**2*(2*AB_product*Dx**2 - AB_sum) + alpha*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(3*Dx**2 + Dz**2) + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 0, 3, 0):
        return (1/2)*np.pi**(3/2)*AB_product*Dx*Dy*Dz*(3*AB_sum**2 - alpha*(-2*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 0, 2, 1):
        return (1/4)*np.pi**(3/2)*Dx*(AB_sum**2*(2*AB_product*Dz**2 - AB_sum) + alpha*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + Dz**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 0, 1, 2):
        return (1/2)*np.pi**(3/2)*AB_product*Dx*Dy*Dz*(AB_sum**2 - alpha*(-2*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 0, 0, 3):
        return (1/4)*np.pi**(3/2)*Dx*(3*AB_sum**2*(2*AB_product*Dz**2 - AB_sum) + alpha*(4*AB_product**2*Dz**4 - 12*AB_product*AB_sum*Dz**2 + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 2, 0, 0, 0, 0):
        return (1/2)*np.pi**(3/2)*(AB_sum**2 + beta*(2*AB_product*Dy**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(7/2)*alpha)
    if (i, j, k, l, m, n) == (0, 2, 0, 1, 0, 0):
        return (1/2)*np.pi**(3/2)*Dx*(AB_sum**2 + beta*(2*AB_product*Dy**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (0, 2, 0, 0, 1, 0):
        return (1/2)*np.pi**(3/2)*Dy*(AB_sum**2 - beta*(-2*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (0, 2, 0, 0, 0, 1):
        return (1/2)*np.pi**(3/2)*Dz*(AB_sum**2 + beta*(2*AB_product*Dy**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (0, 2, 0, 2, 0, 0):
        return (1/4)*np.pi**(3/2)*(AB_product*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + Dy**2) + AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dx**2 - AB_sum) + AB_sum**2*beta*(2*AB_product*Dy**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(11/2))
    if (i, j, k, l, m, n) == (0, 2, 0, 1, 1, 0):
        return (1/2)*np.pi**(3/2)*Dx*Dy*alpha*(AB_sum**2 - beta*(-2*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 2, 0, 1, 0, 1):
        return (1/2)*np.pi**(3/2)*Dx*Dz*alpha*(AB_sum**2 + beta*(2*AB_product*Dy**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 2, 0, 0, 2, 0):
        return (1/4)*np.pi**(3/2)*(AB_product*(4*AB_product**2*Dy**4 - 12*AB_product*AB_sum*Dy**2 + 3*AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dy**2 - AB_sum) + AB_sum**2*beta*(2*AB_product*Dy**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(11/2))
    if (i, j, k, l, m, n) == (0, 2, 0, 0, 1, 1):
        return (1/2)*np.pi**(3/2)*Dy*Dz*alpha*(AB_sum**2 - beta*(-2*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 2, 0, 0, 0, 2):
        return (1/4)*np.pi**(3/2)*(AB_product*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + Dz**2) + AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dz**2 - AB_sum) + AB_sum**2*beta*(2*AB_product*Dy**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(11/2))
    if (i, j, k, l, m, n) == (0, 2, 0, 3, 0, 0):
        return (1/4)*np.pi**(3/2)*Dx*(AB_product*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + 3*Dy**2) + 3*AB_sum**2) + 3*AB_sum**4 - AB_sum**2*alpha*(-2*AB_product*Dx**2 + 3*alpha + 3*beta) + 3*AB_sum**2*beta*(2*AB_product*Dy**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*beta)
    if (i, j, k, l, m, n) == (0, 2, 0, 2, 1, 0):
        return (1/4)*np.pi**(3/2)*Dy*(AB_product*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(3*Dx**2 + Dy**2) + 3*AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dx**2 - AB_sum) - AB_sum**2*beta*(-2*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*beta)
    if (i, j, k, l, m, n) == (0, 2, 0, 2, 0, 1):
        return (1/4)*np.pi**(3/2)*Dz*(AB_product*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + Dy**2) + AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dx**2 - AB_sum) + AB_sum**2*beta*(2*AB_product*Dy**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*beta)
    if (i, j, k, l, m, n) == (0, 2, 0, 1, 2, 0):
        return (1/4)*np.pi**(3/2)*Dx*(AB_product*(4*AB_product**2*Dy**4 - 12*AB_product*AB_sum*Dy**2 + 3*AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dy**2 - AB_sum) + AB_sum**2*beta*(2*AB_product*Dy**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*beta)
    if (i, j, k, l, m, n) == (0, 2, 0, 1, 1, 1):
        return (1/2)*np.pi**(3/2)*Dx*Dy*Dz*alpha**2*(AB_sum**2 - beta*(-2*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 2, 0, 1, 0, 2):
        return (1/4)*np.pi**(3/2)*Dx*(AB_product*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + Dz**2) + AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dz**2 - AB_sum) + AB_sum**2*beta*(2*AB_product*Dy**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*beta)
    if (i, j, k, l, m, n) == (0, 2, 0, 0, 3, 0):
        return (1/4)*np.pi**(3/2)*Dy*(AB_product*(4*AB_product**2*Dy**4 - 20*AB_product*AB_sum*Dy**2 + 15*AB_sum**2) + 3*AB_sum**4 - AB_sum**2*alpha*(-2*AB_product*Dy**2 + 3*alpha + 3*beta) - 3*AB_sum**2*beta*(-2*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*beta)
    if (i, j, k, l, m, n) == (0, 2, 0, 0, 2, 1):
        return (1/4)*np.pi**(3/2)*Dz*(AB_product*(4*AB_product**2*Dy**4 - 12*AB_product*AB_sum*Dy**2 + 3*AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dy**2 - AB_sum) + AB_sum**2*beta*(2*AB_product*Dy**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*beta)
    if (i, j, k, l, m, n) == (0, 2, 0, 0, 1, 2):
        return (1/4)*np.pi**(3/2)*Dy*(AB_product*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + 3*Dz**2) + 3*AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dz**2 - AB_sum) - AB_sum**2*beta*(-2*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*beta)
    if (i, j, k, l, m, n) == (0, 2, 0, 0, 0, 3):
        return (1/4)*np.pi**(3/2)*Dz*(AB_product*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(3*Dy**2 + Dz**2) + 3*AB_sum**2) + 3*AB_sum**4 - AB_sum**2*alpha*(-2*AB_product*Dz**2 + 3*alpha + 3*beta) + 3*AB_sum**2*beta*(2*AB_product*Dy**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*beta)
    if (i, j, k, l, m, n) == (0, 1, 1, 0, 0, 0):
        return np.pi**(3/2)*Dy*Dz*beta**2*np.exp(-AB_product*D2/AB_sum)/AB_sum**(7/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 1, 0, 0):
        return np.pi**(3/2)*AB_product*Dx*Dy*Dz*beta*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 0, 1, 0):
        return (1/2)*np.pi**(3/2)*Dz*beta*(2*AB_product*Dy**2 - AB_sum)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 0, 0, 1):
        return (1/2)*np.pi**(3/2)*Dy*beta*(2*AB_product*Dz**2 - AB_sum)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 2, 0, 0):
        return (1/2)*np.pi**(3/2)*Dy*Dz*beta*(AB_sum**2 + alpha*(2*AB_product*Dx**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 1, 1, 0):
        return (1/2)*np.pi**(3/2)*AB_product*Dx*Dz*(2*AB_product*Dy**2 - AB_sum)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 1, 0, 1):
        return (1/2)*np.pi**(3/2)*AB_product*Dx*Dy*(2*AB_product*Dz**2 - AB_sum)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 0, 2, 0):
        return (1/2)*np.pi**(3/2)*Dy*Dz*beta*(AB_sum**2 - alpha*(-2*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 0, 1, 1):
        return np.pi**(3/2)*(AB_product**2*Dy**2*Dz**2 - 1/2*AB_product*AB_sum*(Dy**2 + Dz**2) + (1/4)*AB_sum**2)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 0, 0, 2):
        return (1/2)*np.pi**(3/2)*Dy*Dz*beta*(AB_sum**2 - alpha*(-2*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 3, 0, 0):
        return (1/2)*np.pi**(3/2)*AB_product*Dx*Dy*Dz*(3*AB_sum**2 - alpha*(-2*AB_product*Dx**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 2, 1, 0):
        return (1/4)*np.pi**(3/2)*Dz*(AB_sum**2*(2*AB_product*Dy**2 - AB_sum) + alpha*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + Dy**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 2, 0, 1):
        return (1/4)*np.pi**(3/2)*Dy*(AB_sum**2*(2*AB_product*Dz**2 - AB_sum) + alpha*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + Dz**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 1, 2, 0):
        return (1/2)*np.pi**(3/2)*AB_product*Dx*Dy*Dz*(AB_sum**2 - alpha*(-2*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 1, 1, 1):
        return (1/4)*np.pi**(3/2)*Dx*alpha*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + Dz**2) + AB_sum**2)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 1, 0, 2):
        return (1/2)*np.pi**(3/2)*AB_product*Dx*Dy*Dz*(AB_sum**2 - alpha*(-2*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 0, 3, 0):
        return (1/4)*np.pi**(3/2)*Dz*(3*AB_sum**2*(2*AB_product*Dy**2 - AB_sum) + alpha*(4*AB_product**2*Dy**4 - 12*AB_product*AB_sum*Dy**2 + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 0, 2, 1):
        return (1/4)*np.pi**(3/2)*Dy*(AB_sum**2*(2*AB_product*Dz**2 - AB_sum) + alpha*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + 3*Dz**2) + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 0, 1, 2):
        return (1/4)*np.pi**(3/2)*Dz*(AB_sum**2*(2*AB_product*Dy**2 - AB_sum) + alpha*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(3*Dy**2 + Dz**2) + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 0, 0, 3):
        return (1/4)*np.pi**(3/2)*Dy*(3*AB_sum**2*(2*AB_product*Dz**2 - AB_sum) + alpha*(4*AB_product**2*Dz**4 - 12*AB_product*AB_sum*Dz**2 + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 0, 2, 0, 0, 0):
        return (1/2)*np.pi**(3/2)*(AB_sum**2 + beta*(2*AB_product*Dz**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(7/2)*alpha)
    if (i, j, k, l, m, n) == (0, 0, 2, 1, 0, 0):
        return (1/2)*np.pi**(3/2)*Dx*(AB_sum**2 + beta*(2*AB_product*Dz**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (0, 0, 2, 0, 1, 0):
        return (1/2)*np.pi**(3/2)*Dy*(AB_sum**2 + beta*(2*AB_product*Dz**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (0, 0, 2, 0, 0, 1):
        return (1/2)*np.pi**(3/2)*Dz*(AB_sum**2 - beta*(-2*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (0, 0, 2, 2, 0, 0):
        return (1/4)*np.pi**(3/2)*(AB_product*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + Dz**2) + AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dx**2 - AB_sum) + AB_sum**2*beta*(2*AB_product*Dz**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(11/2))
    if (i, j, k, l, m, n) == (0, 0, 2, 1, 1, 0):
        return (1/2)*np.pi**(3/2)*Dx*Dy*alpha*(AB_sum**2 + beta*(2*AB_product*Dz**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 0, 2, 1, 0, 1):
        return (1/2)*np.pi**(3/2)*Dx*Dz*alpha*(AB_sum**2 - beta*(-2*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 0, 2, 0, 2, 0):
        return (1/4)*np.pi**(3/2)*(AB_product*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + Dz**2) + AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dy**2 - AB_sum) + AB_sum**2*beta*(2*AB_product*Dz**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(11/2))
    if (i, j, k, l, m, n) == (0, 0, 2, 0, 1, 1):
        return (1/2)*np.pi**(3/2)*Dy*Dz*alpha*(AB_sum**2 - beta*(-2*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 0, 2, 0, 0, 2):
        return (1/4)*np.pi**(3/2)*(AB_product*(4*AB_product**2*Dz**4 - 12*AB_product*AB_sum*Dz**2 + 3*AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dz**2 - AB_sum) + AB_sum**2*beta*(2*AB_product*Dz**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(11/2))
    if (i, j, k, l, m, n) == (0, 0, 2, 3, 0, 0):
        return (1/4)*np.pi**(3/2)*Dx*(AB_product*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + 3*Dz**2) + 3*AB_sum**2) + 3*AB_sum**4 - AB_sum**2*alpha*(-2*AB_product*Dx**2 + 3*alpha + 3*beta) + 3*AB_sum**2*beta*(2*AB_product*Dz**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*beta)
    if (i, j, k, l, m, n) == (0, 0, 2, 2, 1, 0):
        return (1/4)*np.pi**(3/2)*Dy*(AB_product*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + Dz**2) + AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dx**2 - AB_sum) + AB_sum**2*beta*(2*AB_product*Dz**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*beta)
    if (i, j, k, l, m, n) == (0, 0, 2, 2, 0, 1):
        return (1/4)*np.pi**(3/2)*Dz*(AB_product*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(3*Dx**2 + Dz**2) + 3*AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dx**2 - AB_sum) - AB_sum**2*beta*(-2*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*beta)
    if (i, j, k, l, m, n) == (0, 0, 2, 1, 2, 0):
        return (1/4)*np.pi**(3/2)*Dx*(AB_product*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + Dz**2) + AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dy**2 - AB_sum) + AB_sum**2*beta*(2*AB_product*Dz**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*beta)
    if (i, j, k, l, m, n) == (0, 0, 2, 1, 1, 1):
        return (1/2)*np.pi**(3/2)*Dx*Dy*Dz*alpha**2*(AB_sum**2 - beta*(-2*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 0, 2, 1, 0, 2):
        return (1/4)*np.pi**(3/2)*Dx*(AB_product*(4*AB_product**2*Dz**4 - 12*AB_product*AB_sum*Dz**2 + 3*AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dz**2 - AB_sum) + AB_sum**2*beta*(2*AB_product*Dz**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*beta)
    if (i, j, k, l, m, n) == (0, 0, 2, 0, 3, 0):
        return (1/4)*np.pi**(3/2)*Dy*(AB_product*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + 3*Dz**2) + 3*AB_sum**2) + 3*AB_sum**4 - AB_sum**2*alpha*(-2*AB_product*Dy**2 + 3*alpha + 3*beta) + 3*AB_sum**2*beta*(2*AB_product*Dz**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*beta)
    if (i, j, k, l, m, n) == (0, 0, 2, 0, 2, 1):
        return (1/4)*np.pi**(3/2)*Dz*(AB_product*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(3*Dy**2 + Dz**2) + 3*AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dy**2 - AB_sum) - AB_sum**2*beta*(-2*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*beta)
    if (i, j, k, l, m, n) == (0, 0, 2, 0, 1, 2):
        return (1/4)*np.pi**(3/2)*Dy*(AB_product*(4*AB_product**2*Dz**4 - 12*AB_product*AB_sum*Dz**2 + 3*AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dz**2 - AB_sum) + AB_sum**2*beta*(2*AB_product*Dz**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*beta)
    if (i, j, k, l, m, n) == (0, 0, 2, 0, 0, 3):
        return (1/4)*np.pi**(3/2)*Dz*(AB_product*(4*AB_product**2*Dz**4 - 20*AB_product*AB_sum*Dz**2 + 15*AB_sum**2) + 3*AB_sum**4 - AB_sum**2*alpha*(-2*AB_product*Dz**2 + 3*alpha + 3*beta) - 3*AB_sum**2*beta*(-2*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*beta)
    if (i, j, k, l, m, n) == (3, 0, 0, 0, 0, 0):
        return -1/2*np.pi**(3/2)*Dx*beta*(3*AB_sum**2 - beta*(-2*AB_product*Dx**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(9/2)*alpha)
    if (i, j, k, l, m, n) == (3, 0, 0, 1, 0, 0):
        return (1/4)*np.pi**(3/2)*(-3*AB_sum**2*(2*AB_product*Dx**2 - AB_sum) - beta*(4*AB_product**2*Dx**4 - 12*AB_product*AB_sum*Dx**2 + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(11/2)*alpha)
    if (i, j, k, l, m, n) == (3, 0, 0, 0, 1, 0):
        return -1/2*np.pi**(3/2)*Dx*Dy*beta*(3*AB_sum**2 - beta*(-2*AB_product*Dx**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (3, 0, 0, 0, 0, 1):
        return -1/2*np.pi**(3/2)*Dx*Dz*beta*(3*AB_sum**2 - beta*(-2*AB_product*Dx**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (3, 0, 0, 2, 0, 0):
        return -1/4*np.pi**(3/2)*Dx*(AB_product*(4*AB_product**2*Dx**4 - 20*AB_product*AB_sum*Dx**2 + 15*AB_sum**2) + 3*AB_sum**4 - 3*AB_sum**2*alpha*(-2*AB_product*Dx**2 + 3*alpha + 3*beta) - AB_sum**2*beta*(-2*AB_product*Dx**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*alpha)
    if (i, j, k, l, m, n) == (3, 0, 0, 1, 1, 0):
        return -1/4*np.pi**(3/2)*Dy*(3*AB_sum**2*(2*AB_product*Dx**2 - AB_sum) + beta*(4*AB_product**2*Dx**4 - 12*AB_product*AB_sum*Dx**2 + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (3, 0, 0, 1, 0, 1):
        return -1/4*np.pi**(3/2)*Dz*(3*AB_sum**2*(2*AB_product*Dx**2 - AB_sum) + beta*(4*AB_product**2*Dx**4 - 12*AB_product*AB_sum*Dx**2 + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (3, 0, 0, 0, 2, 0):
        return -1/4*np.pi**(3/2)*Dx*(AB_product*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + 3*Dy**2) + 3*AB_sum**2) + 3*AB_sum**4 + 3*AB_sum**2*alpha*(2*AB_product*Dy**2 - AB_sum) - AB_sum**2*beta*(-2*AB_product*Dx**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*alpha)
    if (i, j, k, l, m, n) == (3, 0, 0, 0, 1, 1):
        return -1/2*np.pi**(3/2)*AB_product*Dx*Dy*Dz*(3*AB_sum**2 - beta*(-2*AB_product*Dx**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (3, 0, 0, 0, 0, 2):
        return -1/4*np.pi**(3/2)*Dx*(AB_product*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + 3*Dz**2) + 3*AB_sum**2) + 3*AB_sum**4 + 3*AB_sum**2*alpha*(2*AB_product*Dz**2 - AB_sum) - AB_sum**2*beta*(-2*AB_product*Dx**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*alpha)
    if (i, j, k, l, m, n) == (3, 0, 0, 3, 0, 0):
        return -1/8*np.pi**(3/2)*(AB_product*(8*AB_product**3*Dx**6 - 60*AB_product**2*Dx**4*(alpha + beta) + 90*AB_product*AB_sum**2*Dx**2 - 15*AB_sum**3) + 9*AB_sum**4*(2*AB_product*Dx**2 - AB_sum) + 3*AB_sum**2*alpha*(4*AB_product**2*Dx**4 - 12*AB_product*AB_sum*Dx**2 + 3*AB_sum**2) + 3*AB_sum**2*beta*(4*AB_product**2*Dx**4 - 12*AB_product*AB_sum*Dx**2 + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(15/2))
    if (i, j, k, l, m, n) == (3, 0, 0, 2, 1, 0):
        return -1/4*np.pi**(3/2)*Dx*Dy*(AB_product*(4*AB_product**2*Dx**4 - 20*AB_product*AB_sum*Dx**2 + 15*AB_sum**2) + 3*AB_sum**4 - AB_sum**2*(3*alpha + beta)*(-2*AB_product*Dx**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (3, 0, 0, 2, 0, 1):
        return -1/4*np.pi**(3/2)*Dx*Dz*(AB_product*(4*AB_product**2*Dx**4 - 20*AB_product*AB_sum*Dx**2 + 15*AB_sum**2) + 3*AB_sum**4 - AB_sum**2*(3*alpha + beta)*(-2*AB_product*Dx**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (3, 0, 0, 1, 2, 0):
        return (1/8)*np.pi**(3/2)*(-AB_product*(8*AB_product**3*Dx**4*Dy**2 - 4*AB_product**2*AB_sum*Dx**2*(Dx**2 + 6*Dy**2) + 6*AB_product*AB_sum**2*(2*Dx**2 + Dy**2) - 3*AB_sum**3) - 3*AB_sum**4*(2*AB_product*Dx**2 - AB_sum) - 3*AB_sum**2*alpha*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + Dy**2) + AB_sum**2) - AB_sum**2*beta*(4*AB_product**2*Dx**4 - 12*AB_product*AB_sum*Dx**2 + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(15/2))
    if (i, j, k, l, m, n) == (3, 0, 0, 1, 1, 1):
        return -1/4*np.pi**(3/2)*Dy*Dz*alpha*(3*AB_sum**2*(2*AB_product*Dx**2 - AB_sum) + beta*(4*AB_product**2*Dx**4 - 12*AB_product*AB_sum*Dx**2 + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (3, 0, 0, 1, 0, 2):
        return (1/8)*np.pi**(3/2)*(-AB_product*(8*AB_product**3*Dx**4*Dz**2 - 4*AB_product**2*AB_sum*Dx**2*(Dx**2 + 6*Dz**2) + 6*AB_product*AB_sum**2*(2*Dx**2 + Dz**2) - 3*AB_sum**3) - 3*AB_sum**4*(2*AB_product*Dx**2 - AB_sum) - 3*AB_sum**2*alpha*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + Dz**2) + AB_sum**2) - AB_sum**2*beta*(4*AB_product**2*Dx**4 - 12*AB_product*AB_sum*Dx**2 + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(15/2))
    if (i, j, k, l, m, n) == (3, 0, 0, 0, 3, 0):
        return -1/4*np.pi**(3/2)*Dx*Dy*(AB_product*(4*AB_product**2*Dx**2*Dy**2 - 6*AB_product*AB_sum*(Dx**2 + Dy**2) + 9*AB_sum**2) + 9*AB_sum**4 - 3*AB_sum**2*(alpha*(-2*AB_product*Dy**2 + 3*alpha + 3*beta) + beta*(-2*AB_product*Dx**2 + 3*alpha + 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (3, 0, 0, 0, 2, 1):
        return -1/4*np.pi**(3/2)*Dx*Dz*(AB_product*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + 3*Dy**2) + 3*AB_sum**2) + 3*AB_sum**4 + AB_sum**2*(3*alpha*(2*AB_product*Dy**2 - AB_sum) - beta*(-2*AB_product*Dx**2 + 3*alpha + 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (3, 0, 0, 0, 1, 2):
        return -1/4*np.pi**(3/2)*Dx*Dy*(AB_product*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + 3*Dz**2) + 3*AB_sum**2) + 3*AB_sum**4 + AB_sum**2*(3*alpha*(2*AB_product*Dz**2 - AB_sum) - beta*(-2*AB_product*Dx**2 + 3*alpha + 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (3, 0, 0, 0, 0, 3):
        return -1/4*np.pi**(3/2)*Dx*Dz*(AB_product*(4*AB_product**2*Dx**2*Dz**2 - 6*AB_product*AB_sum*(Dx**2 + Dz**2) + 9*AB_sum**2) + 9*AB_sum**4 - 3*AB_sum**2*(alpha*(-2*AB_product*Dz**2 + 3*alpha + 3*beta) + beta*(-2*AB_product*Dx**2 + 3*alpha + 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (2, 1, 0, 0, 0, 0):
        return -1/2*np.pi**(3/2)*Dy*beta*(AB_sum**2 + beta*(2*AB_product*Dx**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(9/2)*alpha)
    if (i, j, k, l, m, n) == (2, 1, 0, 1, 0, 0):
        return -1/2*np.pi**(3/2)*Dx*Dy*beta*(AB_sum**2 - beta*(-2*AB_product*Dx**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (2, 1, 0, 0, 1, 0):
        return (1/4)*np.pi**(3/2)*(-AB_sum**2*(2*AB_product*Dy**2 - AB_sum) - beta*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + Dy**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(11/2)*alpha)
    if (i, j, k, l, m, n) == (2, 1, 0, 0, 0, 1):
        return -1/2*np.pi**(3/2)*Dy*Dz*beta*(AB_sum**2 + beta*(2*AB_product*Dx**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (2, 1, 0, 2, 0, 0):
        return -1/4*np.pi**(3/2)*Dy*(AB_product*(4*AB_product**2*Dx**4 - 12*AB_product*AB_sum*Dx**2 + 3*AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dx**2 - AB_sum) + AB_sum**2*beta*(2*AB_product*Dx**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*alpha)
    if (i, j, k, l, m, n) == (2, 1, 0, 1, 1, 0):
        return -1/4*np.pi**(3/2)*Dx*(AB_sum**2*(2*AB_product*Dy**2 - AB_sum) + beta*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + 3*Dy**2) + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (2, 1, 0, 1, 0, 1):
        return -1/2*np.pi**(3/2)*AB_product*Dx*Dy*Dz*(AB_sum**2 - beta*(-2*AB_product*Dx**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (2, 1, 0, 0, 2, 0):
        return -1/4*np.pi**(3/2)*Dy*(AB_product*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(3*Dx**2 + Dy**2) + 3*AB_sum**2) + AB_sum**4 - AB_sum**2*alpha*(-2*AB_product*Dy**2 + 3*alpha + 3*beta) + AB_sum**2*beta*(2*AB_product*Dx**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*alpha)
    if (i, j, k, l, m, n) == (2, 1, 0, 0, 1, 1):
        return -1/4*np.pi**(3/2)*Dz*(AB_sum**2*(2*AB_product*Dy**2 - AB_sum) + beta*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + Dy**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (2, 1, 0, 0, 0, 2):
        return -1/4*np.pi**(3/2)*Dy*(AB_product*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + Dz**2) + AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dz**2 - AB_sum) + AB_sum**2*beta*(2*AB_product*Dx**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*alpha)
    if (i, j, k, l, m, n) == (2, 1, 0, 3, 0, 0):
        return -1/4*np.pi**(3/2)*Dx*Dy*(AB_product*(4*AB_product**2*Dx**4 - 20*AB_product*AB_sum*Dx**2 + 15*AB_sum**2) + 3*AB_sum**4 - AB_sum**2*(alpha + 3*beta)*(-2*AB_product*Dx**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (2, 1, 0, 2, 1, 0):
        return (1/8)*np.pi**(3/2)*(-AB_product*(8*AB_product**3*Dx**4*Dy**2 - 4*AB_product**2*AB_sum*Dx**2*(Dx**2 + 6*Dy**2) + 6*AB_product*AB_sum**2*(2*Dx**2 + Dy**2) - 3*AB_sum**3) - AB_sum**4*(2*AB_product*Dy**2 - AB_sum) - AB_sum**2*alpha*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + Dy**2) + AB_sum**2) - AB_sum**2*beta*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + Dy**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(15/2))
    if (i, j, k, l, m, n) == (2, 1, 0, 2, 0, 1):
        return (1/4)*np.pi**(3/2)*AB_product*Dy*Dz*(-4*AB_product**2*Dx**4 + 12*AB_product*AB_sum*Dx**2 - 2*AB_sum**3*Dx**2 - 3*AB_sum**2)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (2, 1, 0, 1, 2, 0):
        return -1/4*np.pi**(3/2)*Dx*Dy*(AB_product*(4*AB_product**2*Dx**2*Dy**2 - 6*AB_product*AB_sum*(Dx**2 + Dy**2) + 9*AB_sum**2) + AB_sum**4 - AB_sum**2*(alpha*(-2*AB_product*Dy**2 + 3*alpha + 3*beta) + beta*(-2*AB_product*Dx**2 + 3*alpha + 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (2, 1, 0, 1, 1, 1):
        return -1/4*np.pi**(3/2)*Dx*Dz*alpha*(AB_sum**2*(2*AB_product*Dy**2 - AB_sum) + beta*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + 3*Dy**2) + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (2, 1, 0, 1, 0, 2):
        return -1/4*np.pi**(3/2)*Dx*Dy*(AB_product*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + 3*Dz**2) + 3*AB_sum**2) + AB_sum**4 + AB_sum**2*(alpha*(2*AB_product*Dz**2 - AB_sum) - beta*(-2*AB_product*Dx**2 + 3*alpha + 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (2, 1, 0, 0, 3, 0):
        return (1/8)*np.pi**(3/2)*(-AB_product*(8*AB_product**3*Dx**2*Dy**4 - 4*AB_product**2*AB_sum*Dy**2*(6*Dx**2 + Dy**2) + 6*AB_product*AB_sum**2*(Dx**2 + 2*Dy**2) - 3*AB_sum**3) - 3*AB_sum**4*(2*AB_product*Dy**2 - AB_sum) - AB_sum**2*alpha*(4*AB_product**2*Dy**4 - 12*AB_product*AB_sum*Dy**2 + 3*AB_sum**2) - 3*AB_sum**2*beta*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + Dy**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(15/2))
    if (i, j, k, l, m, n) == (2, 1, 0, 0, 2, 1):
        return -1/4*np.pi**(3/2)*Dy*Dz*(AB_product*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(3*Dx**2 + Dy**2) + 3*AB_sum**2) + AB_sum**4 - AB_sum**2*(alpha*(-2*AB_product*Dy**2 + 3*alpha + 3*beta) - beta*(2*AB_product*Dx**2 - AB_sum)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (2, 1, 0, 0, 1, 2):
        return (1/8)*np.pi**(3/2)*(-AB_product*(8*AB_product**3*Dx**2*Dy**2*Dz**2 - 4*AB_product**2*AB_sum*(Dx**2*Dy**2 + Dx**2*Dz**2 + Dy**2*Dz**2) + 2*AB_product*AB_sum**2*D2 - AB_sum**3) - AB_sum**4*(2*AB_product*Dy**2 - AB_sum) - AB_sum**2*alpha*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + Dz**2) + AB_sum**2) - AB_sum**2*beta*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + Dy**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(15/2))
    if (i, j, k, l, m, n) == (2, 1, 0, 0, 0, 3):
        return -1/4*np.pi**(3/2)*Dy*Dz*(AB_product*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(3*Dx**2 + Dz**2) + 3*AB_sum**2) + 3*AB_sum**4 - AB_sum**2*(alpha*(-2*AB_product*Dz**2 + 3*alpha + 3*beta) - 3*beta*(2*AB_product*Dx**2 - AB_sum)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (2, 0, 1, 0, 0, 0):
        return -1/2*np.pi**(3/2)*Dz*beta*(AB_sum**2 + beta*(2*AB_product*Dx**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(9/2)*alpha)
    if (i, j, k, l, m, n) == (2, 0, 1, 1, 0, 0):
        return -1/2*np.pi**(3/2)*Dx*Dz*beta*(AB_sum**2 - beta*(-2*AB_product*Dx**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (2, 0, 1, 0, 1, 0):
        return -1/2*np.pi**(3/2)*Dy*Dz*beta*(AB_sum**2 + beta*(2*AB_product*Dx**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (2, 0, 1, 0, 0, 1):
        return (1/4)*np.pi**(3/2)*(-AB_sum**2*(2*AB_product*Dz**2 - AB_sum) - beta*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + Dz**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(11/2)*alpha)
    if (i, j, k, l, m, n) == (2, 0, 1, 2, 0, 0):
        return -1/4*np.pi**(3/2)*Dz*(AB_product*(4*AB_product**2*Dx**4 - 12*AB_product*AB_sum*Dx**2 + 3*AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dx**2 - AB_sum) + AB_sum**2*beta*(2*AB_product*Dx**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*alpha)
    if (i, j, k, l, m, n) == (2, 0, 1, 1, 1, 0):
        return -1/2*np.pi**(3/2)*AB_product*Dx*Dy*Dz*(AB_sum**2 - beta*(-2*AB_product*Dx**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (2, 0, 1, 1, 0, 1):
        return -1/4*np.pi**(3/2)*Dx*(AB_sum**2*(2*AB_product*Dz**2 - AB_sum) + beta*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + 3*Dz**2) + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (2, 0, 1, 0, 2, 0):
        return -1/4*np.pi**(3/2)*Dz*(AB_product*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + Dy**2) + AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dy**2 - AB_sum) + AB_sum**2*beta*(2*AB_product*Dx**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*alpha)
    if (i, j, k, l, m, n) == (2, 0, 1, 0, 1, 1):
        return -1/4*np.pi**(3/2)*Dy*(AB_sum**2*(2*AB_product*Dz**2 - AB_sum) + beta*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + Dz**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (2, 0, 1, 0, 0, 2):
        return -1/4*np.pi**(3/2)*Dz*(AB_product*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(3*Dx**2 + Dz**2) + 3*AB_sum**2) + AB_sum**4 - AB_sum**2*alpha*(-2*AB_product*Dz**2 + 3*alpha + 3*beta) + AB_sum**2*beta*(2*AB_product*Dx**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*alpha)
    if (i, j, k, l, m, n) == (2, 0, 1, 3, 0, 0):
        return -1/4*np.pi**(3/2)*Dx*Dz*(AB_product*(4*AB_product**2*Dx**4 - 20*AB_product*AB_sum*Dx**2 + 15*AB_sum**2) + 3*AB_sum**4 - AB_sum**2*(alpha + 3*beta)*(-2*AB_product*Dx**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (2, 0, 1, 2, 1, 0):
        return (1/4)*np.pi**(3/2)*AB_product*Dy*Dz*(-4*AB_product**2*Dx**4 + 12*AB_product*AB_sum*Dx**2 - 2*AB_sum**3*Dx**2 - 3*AB_sum**2)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (2, 0, 1, 2, 0, 1):
        return (1/8)*np.pi**(3/2)*(-AB_product*(8*AB_product**3*Dx**4*Dz**2 - 4*AB_product**2*AB_sum*Dx**2*(Dx**2 + 6*Dz**2) + 6*AB_product*AB_sum**2*(2*Dx**2 + Dz**2) - 3*AB_sum**3) - AB_sum**4*(2*AB_product*Dz**2 - AB_sum) - AB_sum**2*alpha*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + Dz**2) + AB_sum**2) - AB_sum**2*beta*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + Dz**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(15/2))
    if (i, j, k, l, m, n) == (2, 0, 1, 1, 2, 0):
        return -1/4*np.pi**(3/2)*Dx*Dz*(AB_product*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + 3*Dy**2) + 3*AB_sum**2) + AB_sum**4 + AB_sum**2*(alpha*(2*AB_product*Dy**2 - AB_sum) - beta*(-2*AB_product*Dx**2 + 3*alpha + 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (2, 0, 1, 1, 1, 1):
        return -1/4*np.pi**(3/2)*Dx*Dy*alpha*(AB_sum**2*(2*AB_product*Dz**2 - AB_sum) + beta*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + 3*Dz**2) + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (2, 0, 1, 1, 0, 2):
        return -1/4*np.pi**(3/2)*Dx*Dz*(AB_product*(4*AB_product**2*Dx**2*Dz**2 - 6*AB_product*AB_sum*(Dx**2 + Dz**2) + 9*AB_sum**2) + AB_sum**4 - AB_sum**2*(alpha*(-2*AB_product*Dz**2 + 3*alpha + 3*beta) + beta*(-2*AB_product*Dx**2 + 3*alpha + 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (2, 0, 1, 0, 3, 0):
        return -1/4*np.pi**(3/2)*Dy*Dz*(AB_product*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(3*Dx**2 + Dy**2) + 3*AB_sum**2) + 3*AB_sum**4 - AB_sum**2*(alpha*(-2*AB_product*Dy**2 + 3*alpha + 3*beta) - 3*beta*(2*AB_product*Dx**2 - AB_sum)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (2, 0, 1, 0, 2, 1):
        return (1/8)*np.pi**(3/2)*(-AB_product*(8*AB_product**3*Dx**2*Dy**2*Dz**2 - 4*AB_product**2*AB_sum*(Dx**2*Dy**2 + Dx**2*Dz**2 + Dy**2*Dz**2) + 2*AB_product*AB_sum**2*D2 - AB_sum**3) - AB_sum**4*(2*AB_product*Dz**2 - AB_sum) - AB_sum**2*alpha*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + Dz**2) + AB_sum**2) - AB_sum**2*beta*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + Dz**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(15/2))
    if (i, j, k, l, m, n) == (2, 0, 1, 0, 1, 2):
        return -1/4*np.pi**(3/2)*Dy*Dz*(AB_product*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(3*Dx**2 + Dz**2) + 3*AB_sum**2) + AB_sum**4 - AB_sum**2*(alpha*(-2*AB_product*Dz**2 + 3*alpha + 3*beta) - beta*(2*AB_product*Dx**2 - AB_sum)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (2, 0, 1, 0, 0, 3):
        return (1/8)*np.pi**(3/2)*(-AB_product*(8*AB_product**3*Dx**2*Dz**4 - 4*AB_product**2*AB_sum*Dz**2*(6*Dx**2 + Dz**2) + 6*AB_product*AB_sum**2*(Dx**2 + 2*Dz**2) - 3*AB_sum**3) - 3*AB_sum**4*(2*AB_product*Dz**2 - AB_sum) - AB_sum**2*alpha*(4*AB_product**2*Dz**4 - 12*AB_product*AB_sum*Dz**2 + 3*AB_sum**2) - 3*AB_sum**2*beta*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + Dz**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(15/2))
    if (i, j, k, l, m, n) == (1, 2, 0, 0, 0, 0):
        return -1/2*np.pi**(3/2)*Dx*beta*(AB_sum**2 + beta*(2*AB_product*Dy**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(9/2)*alpha)
    if (i, j, k, l, m, n) == (1, 2, 0, 1, 0, 0):
        return (1/4)*np.pi**(3/2)*(-AB_sum**2*(2*AB_product*Dx**2 - AB_sum) - beta*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + Dy**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(11/2)*alpha)
    if (i, j, k, l, m, n) == (1, 2, 0, 0, 1, 0):
        return -1/2*np.pi**(3/2)*Dx*Dy*beta*(AB_sum**2 - beta*(-2*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (1, 2, 0, 0, 0, 1):
        return -1/2*np.pi**(3/2)*Dx*Dz*beta*(AB_sum**2 + beta*(2*AB_product*Dy**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (1, 2, 0, 2, 0, 0):
        return -1/4*np.pi**(3/2)*Dx*(AB_product*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + 3*Dy**2) + 3*AB_sum**2) + AB_sum**4 - AB_sum**2*alpha*(-2*AB_product*Dx**2 + 3*alpha + 3*beta) + AB_sum**2*beta*(2*AB_product*Dy**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*alpha)
    if (i, j, k, l, m, n) == (1, 2, 0, 1, 1, 0):
        return -1/4*np.pi**(3/2)*Dy*(AB_sum**2*(2*AB_product*Dx**2 - AB_sum) + beta*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(3*Dx**2 + Dy**2) + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 2, 0, 1, 0, 1):
        return -1/4*np.pi**(3/2)*Dz*(AB_sum**2*(2*AB_product*Dx**2 - AB_sum) + beta*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + Dy**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 2, 0, 0, 2, 0):
        return -1/4*np.pi**(3/2)*Dx*(AB_product*(4*AB_product**2*Dy**4 - 12*AB_product*AB_sum*Dy**2 + 3*AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dy**2 - AB_sum) + AB_sum**2*beta*(2*AB_product*Dy**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*alpha)
    if (i, j, k, l, m, n) == (1, 2, 0, 0, 1, 1):
        return -1/2*np.pi**(3/2)*AB_product*Dx*Dy*Dz*(AB_sum**2 - beta*(-2*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 2, 0, 0, 0, 2):
        return -1/4*np.pi**(3/2)*Dx*(AB_product*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + Dz**2) + AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dz**2 - AB_sum) + AB_sum**2*beta*(2*AB_product*Dy**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*alpha)
    if (i, j, k, l, m, n) == (1, 2, 0, 3, 0, 0):
        return (1/8)*np.pi**(3/2)*(-AB_product*(8*AB_product**3*Dx**4*Dy**2 - 4*AB_product**2*AB_sum*Dx**2*(Dx**2 + 6*Dy**2) + 6*AB_product*AB_sum**2*(2*Dx**2 + Dy**2) - 3*AB_sum**3) - 3*AB_sum**4*(2*AB_product*Dx**2 - AB_sum) - AB_sum**2*alpha*(4*AB_product**2*Dx**4 - 12*AB_product*AB_sum*Dx**2 + 3*AB_sum**2) - 3*AB_sum**2*beta*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + Dy**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(15/2))
    if (i, j, k, l, m, n) == (1, 2, 0, 2, 1, 0):
        return -1/4*np.pi**(3/2)*Dx*Dy*(AB_product*(4*AB_product**2*Dx**2*Dy**2 - 6*AB_product*AB_sum*(Dx**2 + Dy**2) + 9*AB_sum**2) + AB_sum**4 - AB_sum**2*(alpha*(-2*AB_product*Dx**2 + 3*alpha + 3*beta) + beta*(-2*AB_product*Dy**2 + 3*alpha + 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 2, 0, 2, 0, 1):
        return -1/4*np.pi**(3/2)*Dx*Dz*(AB_product*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + 3*Dy**2) + 3*AB_sum**2) + AB_sum**4 - AB_sum**2*(alpha*(-2*AB_product*Dx**2 + 3*alpha + 3*beta) - beta*(2*AB_product*Dy**2 - AB_sum)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 2, 0, 1, 2, 0):
        return (1/8)*np.pi**(3/2)*(-AB_product*(8*AB_product**3*Dx**2*Dy**4 - 4*AB_product**2*AB_sum*Dy**2*(6*Dx**2 + Dy**2) + 6*AB_product*AB_sum**2*(Dx**2 + 2*Dy**2) - 3*AB_sum**3) - AB_sum**4*(2*AB_product*Dx**2 - AB_sum) - AB_sum**2*alpha*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + Dy**2) + AB_sum**2) - AB_sum**2*beta*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + Dy**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(15/2))
    if (i, j, k, l, m, n) == (1, 2, 0, 1, 1, 1):
        return -1/4*np.pi**(3/2)*Dy*Dz*alpha*(AB_sum**2*(2*AB_product*Dx**2 - AB_sum) + beta*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(3*Dx**2 + Dy**2) + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 2, 0, 1, 0, 2):
        return (1/8)*np.pi**(3/2)*(-AB_product*(8*AB_product**3*Dx**2*Dy**2*Dz**2 - 4*AB_product**2*AB_sum*(Dx**2*Dy**2 + Dx**2*Dz**2 + Dy**2*Dz**2) + 2*AB_product*AB_sum**2*D2 - AB_sum**3) - AB_sum**4*(2*AB_product*Dx**2 - AB_sum) - AB_sum**2*alpha*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + Dz**2) + AB_sum**2) - AB_sum**2*beta*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + Dy**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(15/2))
    if (i, j, k, l, m, n) == (1, 2, 0, 0, 3, 0):
        return -1/4*np.pi**(3/2)*Dx*Dy*(AB_product*(4*AB_product**2*Dy**4 - 20*AB_product*AB_sum*Dy**2 + 15*AB_sum**2) + 3*AB_sum**4 - AB_sum**2*(alpha + 3*beta)*(-2*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 2, 0, 0, 2, 1):
        return (1/4)*np.pi**(3/2)*AB_product*Dx*Dz*(-4*AB_product**2*Dy**4 + 12*AB_product*AB_sum*Dy**2 - 2*AB_sum**3*Dy**2 - 3*AB_sum**2)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 2, 0, 0, 1, 2):
        return -1/4*np.pi**(3/2)*Dx*Dy*(AB_product*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + 3*Dz**2) + 3*AB_sum**2) + AB_sum**4 + AB_sum**2*(alpha*(2*AB_product*Dz**2 - AB_sum) - beta*(-2*AB_product*Dy**2 + 3*alpha + 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 2, 0, 0, 0, 3):
        return -1/4*np.pi**(3/2)*Dx*Dz*(AB_product*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(3*Dy**2 + Dz**2) + 3*AB_sum**2) + 3*AB_sum**4 - AB_sum**2*(alpha*(-2*AB_product*Dz**2 + 3*alpha + 3*beta) - 3*beta*(2*AB_product*Dy**2 - AB_sum)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 1, 1, 0, 0, 0):
        return -np.pi**(3/2)*Dx*Dy*Dz*beta**3*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (1, 1, 1, 1, 0, 0):
        return (1/2)*np.pi**(3/2)*Dy*Dz*beta**2*(-2*AB_product*Dx**2 + AB_sum)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (1, 1, 1, 0, 1, 0):
        return (1/2)*np.pi**(3/2)*Dx*Dz*beta**2*(-2*AB_product*Dy**2 + AB_sum)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (1, 1, 1, 0, 0, 1):
        return (1/2)*np.pi**(3/2)*Dx*Dy*beta**2*(-2*AB_product*Dz**2 + AB_sum)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (1, 1, 1, 2, 0, 0):
        return -1/2*np.pi**(3/2)*Dx*Dy*Dz*beta**2*(AB_sum**2 - alpha*(-2*AB_product*Dx**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 1, 1, 1, 1, 0):
        return -1/4*np.pi**(3/2)*Dz*beta*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + Dy**2) + AB_sum**2)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 1, 1, 1, 0, 1):
        return -1/4*np.pi**(3/2)*Dy*beta*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + Dz**2) + AB_sum**2)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 1, 1, 0, 2, 0):
        return -1/2*np.pi**(3/2)*Dx*Dy*Dz*beta**2*(AB_sum**2 - alpha*(-2*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 1, 1, 0, 1, 1):
        return -1/4*np.pi**(3/2)*Dx*beta*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + Dz**2) + AB_sum**2)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 1, 1, 0, 0, 2):
        return -1/2*np.pi**(3/2)*Dx*Dy*Dz*beta**2*(AB_sum**2 - alpha*(-2*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 1, 1, 3, 0, 0):
        return -1/4*np.pi**(3/2)*Dy*Dz*beta*(3*AB_sum**2*(2*AB_product*Dx**2 - AB_sum) + alpha*(4*AB_product**2*Dx**4 - 12*AB_product*AB_sum*Dx**2 + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 1, 1, 2, 1, 0):
        return -1/4*np.pi**(3/2)*Dx*Dz*beta*(AB_sum**2*(2*AB_product*Dy**2 - AB_sum) + alpha*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + 3*Dy**2) + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 1, 1, 2, 0, 1):
        return -1/4*np.pi**(3/2)*Dx*Dy*beta*(AB_sum**2*(2*AB_product*Dz**2 - AB_sum) + alpha*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + 3*Dz**2) + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 1, 1, 1, 2, 0):
        return -1/4*np.pi**(3/2)*Dy*Dz*beta*(AB_sum**2*(2*AB_product*Dx**2 - AB_sum) + alpha*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(3*Dx**2 + Dy**2) + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 1, 1, 1, 1, 1):
        return np.pi**(3/2)*(-AB_product**3*Dx**2*Dy**2*Dz**2 + (1/2)*AB_product**2*AB_sum*(Dx**2*Dy**2 + Dx**2*Dz**2 + Dy**2*Dz**2) - 1/4*AB_product*AB_sum**2*D2 + (1/8)*AB_sum**3)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 1, 1, 1, 0, 2):
        return -1/4*np.pi**(3/2)*Dy*Dz*beta*(AB_sum**2*(2*AB_product*Dx**2 - AB_sum) + alpha*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(3*Dx**2 + Dz**2) + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 1, 1, 0, 3, 0):
        return -1/4*np.pi**(3/2)*Dx*Dz*beta*(3*AB_sum**2*(2*AB_product*Dy**2 - AB_sum) + alpha*(4*AB_product**2*Dy**4 - 12*AB_product*AB_sum*Dy**2 + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 1, 1, 0, 2, 1):
        return -1/4*np.pi**(3/2)*Dx*Dy*beta*(AB_sum**2*(2*AB_product*Dz**2 - AB_sum) + alpha*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + 3*Dz**2) + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 1, 1, 0, 1, 2):
        return -1/4*np.pi**(3/2)*Dx*Dz*beta*(AB_sum**2*(2*AB_product*Dy**2 - AB_sum) + alpha*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(3*Dy**2 + Dz**2) + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 1, 1, 0, 0, 3):
        return -1/4*np.pi**(3/2)*Dx*Dy*beta*(3*AB_sum**2*(2*AB_product*Dz**2 - AB_sum) + alpha*(4*AB_product**2*Dz**4 - 12*AB_product*AB_sum*Dz**2 + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 0, 2, 0, 0, 0):
        return -1/2*np.pi**(3/2)*Dx*beta*(AB_sum**2 + beta*(2*AB_product*Dz**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(9/2)*alpha)
    if (i, j, k, l, m, n) == (1, 0, 2, 1, 0, 0):
        return (1/4)*np.pi**(3/2)*(-AB_sum**2*(2*AB_product*Dx**2 - AB_sum) - beta*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + Dz**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(11/2)*alpha)
    if (i, j, k, l, m, n) == (1, 0, 2, 0, 1, 0):
        return -1/2*np.pi**(3/2)*Dx*Dy*beta*(AB_sum**2 + beta*(2*AB_product*Dz**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (1, 0, 2, 0, 0, 1):
        return -1/2*np.pi**(3/2)*Dx*Dz*beta*(AB_sum**2 - beta*(-2*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (1, 0, 2, 2, 0, 0):
        return -1/4*np.pi**(3/2)*Dx*(AB_product*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + 3*Dz**2) + 3*AB_sum**2) + AB_sum**4 - AB_sum**2*alpha*(-2*AB_product*Dx**2 + 3*alpha + 3*beta) + AB_sum**2*beta*(2*AB_product*Dz**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*alpha)
    if (i, j, k, l, m, n) == (1, 0, 2, 1, 1, 0):
        return -1/4*np.pi**(3/2)*Dy*(AB_sum**2*(2*AB_product*Dx**2 - AB_sum) + beta*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + Dz**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 0, 2, 1, 0, 1):
        return -1/4*np.pi**(3/2)*Dz*(AB_sum**2*(2*AB_product*Dx**2 - AB_sum) + beta*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(3*Dx**2 + Dz**2) + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 0, 2, 0, 2, 0):
        return -1/4*np.pi**(3/2)*Dx*(AB_product*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + Dz**2) + AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dy**2 - AB_sum) + AB_sum**2*beta*(2*AB_product*Dz**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*alpha)
    if (i, j, k, l, m, n) == (1, 0, 2, 0, 1, 1):
        return -1/2*np.pi**(3/2)*AB_product*Dx*Dy*Dz*(AB_sum**2 - beta*(-2*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 0, 2, 0, 0, 2):
        return -1/4*np.pi**(3/2)*Dx*(AB_product*(4*AB_product**2*Dz**4 - 12*AB_product*AB_sum*Dz**2 + 3*AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dz**2 - AB_sum) + AB_sum**2*beta*(2*AB_product*Dz**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*alpha)
    if (i, j, k, l, m, n) == (1, 0, 2, 3, 0, 0):
        return (1/8)*np.pi**(3/2)*(-AB_product*(8*AB_product**3*Dx**4*Dz**2 - 4*AB_product**2*AB_sum*Dx**2*(Dx**2 + 6*Dz**2) + 6*AB_product*AB_sum**2*(2*Dx**2 + Dz**2) - 3*AB_sum**3) - 3*AB_sum**4*(2*AB_product*Dx**2 - AB_sum) - AB_sum**2*alpha*(4*AB_product**2*Dx**4 - 12*AB_product*AB_sum*Dx**2 + 3*AB_sum**2) - 3*AB_sum**2*beta*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + Dz**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(15/2))
    if (i, j, k, l, m, n) == (1, 0, 2, 2, 1, 0):
        return -1/4*np.pi**(3/2)*Dx*Dy*(AB_product*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + 3*Dz**2) + 3*AB_sum**2) + AB_sum**4 - AB_sum**2*(alpha*(-2*AB_product*Dx**2 + 3*alpha + 3*beta) - beta*(2*AB_product*Dz**2 - AB_sum)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 0, 2, 2, 0, 1):
        return -1/4*np.pi**(3/2)*Dx*Dz*(AB_product*(4*AB_product**2*Dx**2*Dz**2 - 6*AB_product*AB_sum*(Dx**2 + Dz**2) + 9*AB_sum**2) + AB_sum**4 - AB_sum**2*(alpha*(-2*AB_product*Dx**2 + 3*alpha + 3*beta) + beta*(-2*AB_product*Dz**2 + 3*alpha + 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 0, 2, 1, 2, 0):
        return (1/8)*np.pi**(3/2)*(-AB_product*(8*AB_product**3*Dx**2*Dy**2*Dz**2 - 4*AB_product**2*AB_sum*(Dx**2*Dy**2 + Dx**2*Dz**2 + Dy**2*Dz**2) + 2*AB_product*AB_sum**2*D2 - AB_sum**3) - AB_sum**4*(2*AB_product*Dx**2 - AB_sum) - AB_sum**2*alpha*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + Dy**2) + AB_sum**2) - AB_sum**2*beta*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + Dz**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(15/2))
    if (i, j, k, l, m, n) == (1, 0, 2, 1, 1, 1):
        return -1/4*np.pi**(3/2)*Dy*Dz*alpha*(AB_sum**2*(2*AB_product*Dx**2 - AB_sum) + beta*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(3*Dx**2 + Dz**2) + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 0, 2, 1, 0, 2):
        return (1/8)*np.pi**(3/2)*(-AB_product*(8*AB_product**3*Dx**2*Dz**4 - 4*AB_product**2*AB_sum*Dz**2*(6*Dx**2 + Dz**2) + 6*AB_product*AB_sum**2*(Dx**2 + 2*Dz**2) - 3*AB_sum**3) - AB_sum**4*(2*AB_product*Dx**2 - AB_sum) - AB_sum**2*alpha*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + Dz**2) + AB_sum**2) - AB_sum**2*beta*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + Dz**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(15/2))
    if (i, j, k, l, m, n) == (1, 0, 2, 0, 3, 0):
        return -1/4*np.pi**(3/2)*Dx*Dy*(AB_product*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + 3*Dz**2) + 3*AB_sum**2) + 3*AB_sum**4 - AB_sum**2*(alpha*(-2*AB_product*Dy**2 + 3*alpha + 3*beta) - 3*beta*(2*AB_product*Dz**2 - AB_sum)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 0, 2, 0, 2, 1):
        return -1/4*np.pi**(3/2)*Dx*Dz*(AB_product*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(3*Dy**2 + Dz**2) + 3*AB_sum**2) + AB_sum**4 + AB_sum**2*(alpha*(2*AB_product*Dy**2 - AB_sum) - beta*(-2*AB_product*Dz**2 + 3*alpha + 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 0, 2, 0, 1, 2):
        return (1/4)*np.pi**(3/2)*AB_product*Dx*Dy*(-4*AB_product**2*Dz**4 + 12*AB_product*AB_sum*Dz**2 - 2*AB_sum**3*Dz**2 - 3*AB_sum**2)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 0, 2, 0, 0, 3):
        return -1/4*np.pi**(3/2)*Dx*Dz*(AB_product*(4*AB_product**2*Dz**4 - 20*AB_product*AB_sum*Dz**2 + 15*AB_sum**2) + 3*AB_sum**4 - AB_sum**2*(alpha + 3*beta)*(-2*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 3, 0, 0, 0, 0):
        return -1/2*np.pi**(3/2)*Dy*beta*(3*AB_sum**2 - beta*(-2*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(9/2)*alpha)
    if (i, j, k, l, m, n) == (0, 3, 0, 1, 0, 0):
        return -1/2*np.pi**(3/2)*Dx*Dy*beta*(3*AB_sum**2 - beta*(-2*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 3, 0, 0, 1, 0):
        return (1/4)*np.pi**(3/2)*(-3*AB_sum**2*(2*AB_product*Dy**2 - AB_sum) - beta*(4*AB_product**2*Dy**4 - 12*AB_product*AB_sum*Dy**2 + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(11/2)*alpha)
    if (i, j, k, l, m, n) == (0, 3, 0, 0, 0, 1):
        return -1/2*np.pi**(3/2)*Dy*Dz*beta*(3*AB_sum**2 - beta*(-2*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 3, 0, 2, 0, 0):
        return -1/4*np.pi**(3/2)*Dy*(AB_product*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(3*Dx**2 + Dy**2) + 3*AB_sum**2) + 3*AB_sum**4 + 3*AB_sum**2*alpha*(2*AB_product*Dx**2 - AB_sum) - AB_sum**2*beta*(-2*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*alpha)
    if (i, j, k, l, m, n) == (0, 3, 0, 1, 1, 0):
        return -1/4*np.pi**(3/2)*Dx*(3*AB_sum**2*(2*AB_product*Dy**2 - AB_sum) + beta*(4*AB_product**2*Dy**4 - 12*AB_product*AB_sum*Dy**2 + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 3, 0, 1, 0, 1):
        return -1/2*np.pi**(3/2)*AB_product*Dx*Dy*Dz*(3*AB_sum**2 - beta*(-2*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 3, 0, 0, 2, 0):
        return -1/4*np.pi**(3/2)*Dy*(AB_product*(4*AB_product**2*Dy**4 - 20*AB_product*AB_sum*Dy**2 + 15*AB_sum**2) + 3*AB_sum**4 - 3*AB_sum**2*alpha*(-2*AB_product*Dy**2 + 3*alpha + 3*beta) - AB_sum**2*beta*(-2*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*alpha)
    if (i, j, k, l, m, n) == (0, 3, 0, 0, 1, 1):
        return -1/4*np.pi**(3/2)*Dz*(3*AB_sum**2*(2*AB_product*Dy**2 - AB_sum) + beta*(4*AB_product**2*Dy**4 - 12*AB_product*AB_sum*Dy**2 + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 3, 0, 0, 0, 2):
        return -1/4*np.pi**(3/2)*Dy*(AB_product*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + 3*Dz**2) + 3*AB_sum**2) + 3*AB_sum**4 + 3*AB_sum**2*alpha*(2*AB_product*Dz**2 - AB_sum) - AB_sum**2*beta*(-2*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*alpha)
    if (i, j, k, l, m, n) == (0, 3, 0, 3, 0, 0):
        return -1/4*np.pi**(3/2)*Dx*Dy*(AB_product*(4*AB_product**2*Dx**2*Dy**2 - 6*AB_product*AB_sum*(Dx**2 + Dy**2) + 9*AB_sum**2) + 9*AB_sum**4 - 3*AB_sum**2*(alpha*(-2*AB_product*Dx**2 + 3*alpha + 3*beta) + beta*(-2*AB_product*Dy**2 + 3*alpha + 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 3, 0, 2, 1, 0):
        return (1/8)*np.pi**(3/2)*(-AB_product*(8*AB_product**3*Dx**2*Dy**4 - 4*AB_product**2*AB_sum*Dy**2*(6*Dx**2 + Dy**2) + 6*AB_product*AB_sum**2*(Dx**2 + 2*Dy**2) - 3*AB_sum**3) - 3*AB_sum**4*(2*AB_product*Dy**2 - AB_sum) - 3*AB_sum**2*alpha*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + Dy**2) + AB_sum**2) - AB_sum**2*beta*(4*AB_product**2*Dy**4 - 12*AB_product*AB_sum*Dy**2 + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(15/2))
    if (i, j, k, l, m, n) == (0, 3, 0, 2, 0, 1):
        return -1/4*np.pi**(3/2)*Dy*Dz*(AB_product*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(3*Dx**2 + Dy**2) + 3*AB_sum**2) + 3*AB_sum**4 + AB_sum**2*(3*alpha*(2*AB_product*Dx**2 - AB_sum) - beta*(-2*AB_product*Dy**2 + 3*alpha + 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 3, 0, 1, 2, 0):
        return -1/4*np.pi**(3/2)*Dx*Dy*(AB_product*(4*AB_product**2*Dy**4 - 20*AB_product*AB_sum*Dy**2 + 15*AB_sum**2) + 3*AB_sum**4 - AB_sum**2*(3*alpha + beta)*(-2*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 3, 0, 1, 1, 1):
        return -1/4*np.pi**(3/2)*Dx*Dz*alpha*(3*AB_sum**2*(2*AB_product*Dy**2 - AB_sum) + beta*(4*AB_product**2*Dy**4 - 12*AB_product*AB_sum*Dy**2 + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 3, 0, 1, 0, 2):
        return -1/4*np.pi**(3/2)*Dx*Dy*(AB_product*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + 3*Dz**2) + 3*AB_sum**2) + 3*AB_sum**4 + AB_sum**2*(3*alpha*(2*AB_product*Dz**2 - AB_sum) - beta*(-2*AB_product*Dy**2 + 3*alpha + 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 3, 0, 0, 3, 0):
        return -1/8*np.pi**(3/2)*(AB_product*(8*AB_product**3*Dy**6 - 60*AB_product**2*Dy**4*(alpha + beta) + 90*AB_product*AB_sum**2*Dy**2 - 15*AB_sum**3) + 9*AB_sum**4*(2*AB_product*Dy**2 - AB_sum) + 3*AB_sum**2*alpha*(4*AB_product**2*Dy**4 - 12*AB_product*AB_sum*Dy**2 + 3*AB_sum**2) + 3*AB_sum**2*beta*(4*AB_product**2*Dy**4 - 12*AB_product*AB_sum*Dy**2 + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(15/2))
    if (i, j, k, l, m, n) == (0, 3, 0, 0, 2, 1):
        return -1/4*np.pi**(3/2)*Dy*Dz*(AB_product*(4*AB_product**2*Dy**4 - 20*AB_product*AB_sum*Dy**2 + 15*AB_sum**2) + 3*AB_sum**4 - AB_sum**2*(3*alpha + beta)*(-2*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 3, 0, 0, 1, 2):
        return (1/8)*np.pi**(3/2)*(-AB_product*(8*AB_product**3*Dy**4*Dz**2 - 4*AB_product**2*AB_sum*Dy**2*(Dy**2 + 6*Dz**2) + 6*AB_product*AB_sum**2*(2*Dy**2 + Dz**2) - 3*AB_sum**3) - 3*AB_sum**4*(2*AB_product*Dy**2 - AB_sum) - 3*AB_sum**2*alpha*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + Dz**2) + AB_sum**2) - AB_sum**2*beta*(4*AB_product**2*Dy**4 - 12*AB_product*AB_sum*Dy**2 + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(15/2))
    if (i, j, k, l, m, n) == (0, 3, 0, 0, 0, 3):
        return -1/4*np.pi**(3/2)*Dy*Dz*(AB_product*(4*AB_product**2*Dy**2*Dz**2 - 6*AB_product*AB_sum*(Dy**2 + Dz**2) + 9*AB_sum**2) + 9*AB_sum**4 - 3*AB_sum**2*(alpha*(-2*AB_product*Dz**2 + 3*alpha + 3*beta) + beta*(-2*AB_product*Dy**2 + 3*alpha + 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 2, 1, 0, 0, 0):
        return -1/2*np.pi**(3/2)*Dz*beta*(AB_sum**2 + beta*(2*AB_product*Dy**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(9/2)*alpha)
    if (i, j, k, l, m, n) == (0, 2, 1, 1, 0, 0):
        return -1/2*np.pi**(3/2)*Dx*Dz*beta*(AB_sum**2 + beta*(2*AB_product*Dy**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 2, 1, 0, 1, 0):
        return -1/2*np.pi**(3/2)*Dy*Dz*beta*(AB_sum**2 - beta*(-2*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 2, 1, 0, 0, 1):
        return (1/4)*np.pi**(3/2)*(-AB_sum**2*(2*AB_product*Dz**2 - AB_sum) - beta*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + Dz**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(11/2)*alpha)
    if (i, j, k, l, m, n) == (0, 2, 1, 2, 0, 0):
        return -1/4*np.pi**(3/2)*Dz*(AB_product*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + Dy**2) + AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dx**2 - AB_sum) + AB_sum**2*beta*(2*AB_product*Dy**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*alpha)
    if (i, j, k, l, m, n) == (0, 2, 1, 1, 1, 0):
        return -1/2*np.pi**(3/2)*AB_product*Dx*Dy*Dz*(AB_sum**2 - beta*(-2*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 2, 1, 1, 0, 1):
        return -1/4*np.pi**(3/2)*Dx*(AB_sum**2*(2*AB_product*Dz**2 - AB_sum) + beta*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + Dz**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 2, 1, 0, 2, 0):
        return -1/4*np.pi**(3/2)*Dz*(AB_product*(4*AB_product**2*Dy**4 - 12*AB_product*AB_sum*Dy**2 + 3*AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dy**2 - AB_sum) + AB_sum**2*beta*(2*AB_product*Dy**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*alpha)
    if (i, j, k, l, m, n) == (0, 2, 1, 0, 1, 1):
        return -1/4*np.pi**(3/2)*Dy*(AB_sum**2*(2*AB_product*Dz**2 - AB_sum) + beta*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + 3*Dz**2) + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 2, 1, 0, 0, 2):
        return -1/4*np.pi**(3/2)*Dz*(AB_product*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(3*Dy**2 + Dz**2) + 3*AB_sum**2) + AB_sum**4 - AB_sum**2*alpha*(-2*AB_product*Dz**2 + 3*alpha + 3*beta) + AB_sum**2*beta*(2*AB_product*Dy**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*alpha)
    if (i, j, k, l, m, n) == (0, 2, 1, 3, 0, 0):
        return -1/4*np.pi**(3/2)*Dx*Dz*(AB_product*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + 3*Dy**2) + 3*AB_sum**2) + 3*AB_sum**4 - AB_sum**2*(alpha*(-2*AB_product*Dx**2 + 3*alpha + 3*beta) - 3*beta*(2*AB_product*Dy**2 - AB_sum)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 2, 1, 2, 1, 0):
        return -1/4*np.pi**(3/2)*Dy*Dz*(AB_product*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(3*Dx**2 + Dy**2) + 3*AB_sum**2) + AB_sum**4 + AB_sum**2*(alpha*(2*AB_product*Dx**2 - AB_sum) - beta*(-2*AB_product*Dy**2 + 3*alpha + 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 2, 1, 2, 0, 1):
        return (1/8)*np.pi**(3/2)*(-AB_product*(8*AB_product**3*Dx**2*Dy**2*Dz**2 - 4*AB_product**2*AB_sum*(Dx**2*Dy**2 + Dx**2*Dz**2 + Dy**2*Dz**2) + 2*AB_product*AB_sum**2*D2 - AB_sum**3) - AB_sum**4*(2*AB_product*Dz**2 - AB_sum) - AB_sum**2*alpha*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + Dz**2) + AB_sum**2) - AB_sum**2*beta*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + Dz**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(15/2))
    if (i, j, k, l, m, n) == (0, 2, 1, 1, 2, 0):
        return (1/4)*np.pi**(3/2)*AB_product*Dx*Dz*(-4*AB_product**2*Dy**4 + 12*AB_product*AB_sum*Dy**2 - 2*AB_sum**3*Dy**2 - 3*AB_sum**2)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 2, 1, 1, 1, 1):
        return -1/4*np.pi**(3/2)*Dx*Dy*alpha*(AB_sum**2*(2*AB_product*Dz**2 - AB_sum) + beta*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + 3*Dz**2) + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 2, 1, 1, 0, 2):
        return -1/4*np.pi**(3/2)*Dx*Dz*(AB_product*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(3*Dy**2 + Dz**2) + 3*AB_sum**2) + AB_sum**4 - AB_sum**2*(alpha*(-2*AB_product*Dz**2 + 3*alpha + 3*beta) - beta*(2*AB_product*Dy**2 - AB_sum)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 2, 1, 0, 3, 0):
        return -1/4*np.pi**(3/2)*Dy*Dz*(AB_product*(4*AB_product**2*Dy**4 - 20*AB_product*AB_sum*Dy**2 + 15*AB_sum**2) + 3*AB_sum**4 - AB_sum**2*(alpha + 3*beta)*(-2*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 2, 1, 0, 2, 1):
        return (1/8)*np.pi**(3/2)*(-AB_product*(8*AB_product**3*Dy**4*Dz**2 - 4*AB_product**2*AB_sum*Dy**2*(Dy**2 + 6*Dz**2) + 6*AB_product*AB_sum**2*(2*Dy**2 + Dz**2) - 3*AB_sum**3) - AB_sum**4*(2*AB_product*Dz**2 - AB_sum) - AB_sum**2*alpha*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + Dz**2) + AB_sum**2) - AB_sum**2*beta*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + Dz**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(15/2))
    if (i, j, k, l, m, n) == (0, 2, 1, 0, 1, 2):
        return -1/4*np.pi**(3/2)*Dy*Dz*(AB_product*(4*AB_product**2*Dy**2*Dz**2 - 6*AB_product*AB_sum*(Dy**2 + Dz**2) + 9*AB_sum**2) + AB_sum**4 - AB_sum**2*(alpha*(-2*AB_product*Dz**2 + 3*alpha + 3*beta) + beta*(-2*AB_product*Dy**2 + 3*alpha + 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 2, 1, 0, 0, 3):
        return (1/8)*np.pi**(3/2)*(-AB_product*(8*AB_product**3*Dy**2*Dz**4 - 4*AB_product**2*AB_sum*Dz**2*(6*Dy**2 + Dz**2) + 6*AB_product*AB_sum**2*(Dy**2 + 2*Dz**2) - 3*AB_sum**3) - 3*AB_sum**4*(2*AB_product*Dz**2 - AB_sum) - AB_sum**2*alpha*(4*AB_product**2*Dz**4 - 12*AB_product*AB_sum*Dz**2 + 3*AB_sum**2) - 3*AB_sum**2*beta*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + Dz**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(15/2))
    if (i, j, k, l, m, n) == (0, 1, 2, 0, 0, 0):
        return -1/2*np.pi**(3/2)*Dy*beta*(AB_sum**2 + beta*(2*AB_product*Dz**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(9/2)*alpha)
    if (i, j, k, l, m, n) == (0, 1, 2, 1, 0, 0):
        return -1/2*np.pi**(3/2)*Dx*Dy*beta*(AB_sum**2 + beta*(2*AB_product*Dz**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 1, 2, 0, 1, 0):
        return (1/4)*np.pi**(3/2)*(-AB_sum**2*(2*AB_product*Dy**2 - AB_sum) - beta*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + Dz**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(11/2)*alpha)
    if (i, j, k, l, m, n) == (0, 1, 2, 0, 0, 1):
        return -1/2*np.pi**(3/2)*Dy*Dz*beta*(AB_sum**2 - beta*(-2*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 1, 2, 2, 0, 0):
        return -1/4*np.pi**(3/2)*Dy*(AB_product*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + Dz**2) + AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dx**2 - AB_sum) + AB_sum**2*beta*(2*AB_product*Dz**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*alpha)
    if (i, j, k, l, m, n) == (0, 1, 2, 1, 1, 0):
        return -1/4*np.pi**(3/2)*Dx*(AB_sum**2*(2*AB_product*Dy**2 - AB_sum) + beta*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + Dz**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 1, 2, 1, 0, 1):
        return -1/2*np.pi**(3/2)*AB_product*Dx*Dy*Dz*(AB_sum**2 - beta*(-2*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 1, 2, 0, 2, 0):
        return -1/4*np.pi**(3/2)*Dy*(AB_product*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + 3*Dz**2) + 3*AB_sum**2) + AB_sum**4 - AB_sum**2*alpha*(-2*AB_product*Dy**2 + 3*alpha + 3*beta) + AB_sum**2*beta*(2*AB_product*Dz**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*alpha)
    if (i, j, k, l, m, n) == (0, 1, 2, 0, 1, 1):
        return -1/4*np.pi**(3/2)*Dz*(AB_sum**2*(2*AB_product*Dy**2 - AB_sum) + beta*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(3*Dy**2 + Dz**2) + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 1, 2, 0, 0, 2):
        return -1/4*np.pi**(3/2)*Dy*(AB_product*(4*AB_product**2*Dz**4 - 12*AB_product*AB_sum*Dz**2 + 3*AB_sum**2) + AB_sum**4 + AB_sum**2*alpha*(2*AB_product*Dz**2 - AB_sum) + AB_sum**2*beta*(2*AB_product*Dz**2 - AB_sum))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*alpha)
    if (i, j, k, l, m, n) == (0, 1, 2, 3, 0, 0):
        return -1/4*np.pi**(3/2)*Dx*Dy*(AB_product*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + 3*Dz**2) + 3*AB_sum**2) + 3*AB_sum**4 - AB_sum**2*(alpha*(-2*AB_product*Dx**2 + 3*alpha + 3*beta) - 3*beta*(2*AB_product*Dz**2 - AB_sum)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 1, 2, 2, 1, 0):
        return (1/8)*np.pi**(3/2)*(-AB_product*(8*AB_product**3*Dx**2*Dy**2*Dz**2 - 4*AB_product**2*AB_sum*(Dx**2*Dy**2 + Dx**2*Dz**2 + Dy**2*Dz**2) + 2*AB_product*AB_sum**2*D2 - AB_sum**3) - AB_sum**4*(2*AB_product*Dy**2 - AB_sum) - AB_sum**2*alpha*(4*AB_product**2*Dx**2*Dy**2 - 2*AB_product*AB_sum*(Dx**2 + Dy**2) + AB_sum**2) - AB_sum**2*beta*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + Dz**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(15/2))
    if (i, j, k, l, m, n) == (0, 1, 2, 2, 0, 1):
        return -1/4*np.pi**(3/2)*Dy*Dz*(AB_product*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(3*Dx**2 + Dz**2) + 3*AB_sum**2) + AB_sum**4 + AB_sum**2*(alpha*(2*AB_product*Dx**2 - AB_sum) - beta*(-2*AB_product*Dz**2 + 3*alpha + 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 1, 2, 1, 2, 0):
        return -1/4*np.pi**(3/2)*Dx*Dy*(AB_product*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + 3*Dz**2) + 3*AB_sum**2) + AB_sum**4 - AB_sum**2*(alpha*(-2*AB_product*Dy**2 + 3*alpha + 3*beta) - beta*(2*AB_product*Dz**2 - AB_sum)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 1, 2, 1, 1, 1):
        return -1/4*np.pi**(3/2)*Dx*Dz*alpha*(AB_sum**2*(2*AB_product*Dy**2 - AB_sum) + beta*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(3*Dy**2 + Dz**2) + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 1, 2, 1, 0, 2):
        return (1/4)*np.pi**(3/2)*AB_product*Dx*Dy*(-4*AB_product**2*Dz**4 + 12*AB_product*AB_sum*Dz**2 - 2*AB_sum**3*Dz**2 - 3*AB_sum**2)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 1, 2, 0, 3, 0):
        return (1/8)*np.pi**(3/2)*(-AB_product*(8*AB_product**3*Dy**4*Dz**2 - 4*AB_product**2*AB_sum*Dy**2*(Dy**2 + 6*Dz**2) + 6*AB_product*AB_sum**2*(2*Dy**2 + Dz**2) - 3*AB_sum**3) - 3*AB_sum**4*(2*AB_product*Dy**2 - AB_sum) - AB_sum**2*alpha*(4*AB_product**2*Dy**4 - 12*AB_product*AB_sum*Dy**2 + 3*AB_sum**2) - 3*AB_sum**2*beta*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + Dz**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(15/2))
    if (i, j, k, l, m, n) == (0, 1, 2, 0, 2, 1):
        return -1/4*np.pi**(3/2)*Dy*Dz*(AB_product*(4*AB_product**2*Dy**2*Dz**2 - 6*AB_product*AB_sum*(Dy**2 + Dz**2) + 9*AB_sum**2) + AB_sum**4 - AB_sum**2*(alpha*(-2*AB_product*Dy**2 + 3*alpha + 3*beta) + beta*(-2*AB_product*Dz**2 + 3*alpha + 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 1, 2, 0, 1, 2):
        return (1/8)*np.pi**(3/2)*(-AB_product*(8*AB_product**3*Dy**2*Dz**4 - 4*AB_product**2*AB_sum*Dz**2*(6*Dy**2 + Dz**2) + 6*AB_product*AB_sum**2*(Dy**2 + 2*Dz**2) - 3*AB_sum**3) - AB_sum**4*(2*AB_product*Dy**2 - AB_sum) - AB_sum**2*alpha*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + Dz**2) + AB_sum**2) - AB_sum**2*beta*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + Dz**2) + AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(15/2))
    if (i, j, k, l, m, n) == (0, 1, 2, 0, 0, 3):
        return -1/4*np.pi**(3/2)*Dy*Dz*(AB_product*(4*AB_product**2*Dz**4 - 20*AB_product*AB_sum*Dz**2 + 15*AB_sum**2) + 3*AB_sum**4 - AB_sum**2*(alpha + 3*beta)*(-2*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 0, 3, 0, 0, 0):
        return -1/2*np.pi**(3/2)*Dz*beta*(3*AB_sum**2 - beta*(-2*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(9/2)*alpha)
    if (i, j, k, l, m, n) == (0, 0, 3, 1, 0, 0):
        return -1/2*np.pi**(3/2)*Dx*Dz*beta*(3*AB_sum**2 - beta*(-2*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 0, 3, 0, 1, 0):
        return -1/2*np.pi**(3/2)*Dy*Dz*beta*(3*AB_sum**2 - beta*(-2*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 0, 3, 0, 0, 1):
        return (1/4)*np.pi**(3/2)*(-3*AB_sum**2*(2*AB_product*Dz**2 - AB_sum) - beta*(4*AB_product**2*Dz**4 - 12*AB_product*AB_sum*Dz**2 + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(11/2)*alpha)
    if (i, j, k, l, m, n) == (0, 0, 3, 2, 0, 0):
        return -1/4*np.pi**(3/2)*Dz*(AB_product*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(3*Dx**2 + Dz**2) + 3*AB_sum**2) + 3*AB_sum**4 + 3*AB_sum**2*alpha*(2*AB_product*Dx**2 - AB_sum) - AB_sum**2*beta*(-2*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*alpha)
    if (i, j, k, l, m, n) == (0, 0, 3, 1, 1, 0):
        return -1/2*np.pi**(3/2)*AB_product*Dx*Dy*Dz*(3*AB_sum**2 - beta*(-2*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 0, 3, 1, 0, 1):
        return -1/4*np.pi**(3/2)*Dx*(3*AB_sum**2*(2*AB_product*Dz**2 - AB_sum) + beta*(4*AB_product**2*Dz**4 - 12*AB_product*AB_sum*Dz**2 + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 0, 3, 0, 2, 0):
        return -1/4*np.pi**(3/2)*Dz*(AB_product*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(3*Dy**2 + Dz**2) + 3*AB_sum**2) + 3*AB_sum**4 + 3*AB_sum**2*alpha*(2*AB_product*Dy**2 - AB_sum) - AB_sum**2*beta*(-2*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*alpha)
    if (i, j, k, l, m, n) == (0, 0, 3, 0, 1, 1):
        return -1/4*np.pi**(3/2)*Dy*(3*AB_sum**2*(2*AB_product*Dz**2 - AB_sum) + beta*(4*AB_product**2*Dz**4 - 12*AB_product*AB_sum*Dz**2 + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 0, 3, 0, 0, 2):
        return -1/4*np.pi**(3/2)*Dz*(AB_product*(4*AB_product**2*Dz**4 - 20*AB_product*AB_sum*Dz**2 + 15*AB_sum**2) + 3*AB_sum**4 - 3*AB_sum**2*alpha*(-2*AB_product*Dz**2 + 3*alpha + 3*beta) - AB_sum**2*beta*(-2*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/(AB_sum**(13/2)*alpha)
    if (i, j, k, l, m, n) == (0, 0, 3, 3, 0, 0):
        return -1/4*np.pi**(3/2)*Dx*Dz*(AB_product*(4*AB_product**2*Dx**2*Dz**2 - 6*AB_product*AB_sum*(Dx**2 + Dz**2) + 9*AB_sum**2) + 9*AB_sum**4 - 3*AB_sum**2*(alpha*(-2*AB_product*Dx**2 + 3*alpha + 3*beta) + beta*(-2*AB_product*Dz**2 + 3*alpha + 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 0, 3, 2, 1, 0):
        return -1/4*np.pi**(3/2)*Dy*Dz*(AB_product*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(3*Dx**2 + Dz**2) + 3*AB_sum**2) + 3*AB_sum**4 + AB_sum**2*(3*alpha*(2*AB_product*Dx**2 - AB_sum) - beta*(-2*AB_product*Dz**2 + 3*alpha + 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 0, 3, 2, 0, 1):
        return (1/8)*np.pi**(3/2)*(-AB_product*(8*AB_product**3*Dx**2*Dz**4 - 4*AB_product**2*AB_sum*Dz**2*(6*Dx**2 + Dz**2) + 6*AB_product*AB_sum**2*(Dx**2 + 2*Dz**2) - 3*AB_sum**3) - 3*AB_sum**4*(2*AB_product*Dz**2 - AB_sum) - 3*AB_sum**2*alpha*(4*AB_product**2*Dx**2*Dz**2 - 2*AB_product*AB_sum*(Dx**2 + Dz**2) + AB_sum**2) - AB_sum**2*beta*(4*AB_product**2*Dz**4 - 12*AB_product*AB_sum*Dz**2 + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(15/2))
    if (i, j, k, l, m, n) == (0, 0, 3, 1, 2, 0):
        return -1/4*np.pi**(3/2)*Dx*Dz*(AB_product*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(3*Dy**2 + Dz**2) + 3*AB_sum**2) + 3*AB_sum**4 + AB_sum**2*(3*alpha*(2*AB_product*Dy**2 - AB_sum) - beta*(-2*AB_product*Dz**2 + 3*alpha + 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 0, 3, 1, 1, 1):
        return -1/4*np.pi**(3/2)*Dx*Dy*alpha*(3*AB_sum**2*(2*AB_product*Dz**2 - AB_sum) + beta*(4*AB_product**2*Dz**4 - 12*AB_product*AB_sum*Dz**2 + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 0, 3, 1, 0, 2):
        return -1/4*np.pi**(3/2)*Dx*Dz*(AB_product*(4*AB_product**2*Dz**4 - 20*AB_product*AB_sum*Dz**2 + 15*AB_sum**2) + 3*AB_sum**4 - AB_sum**2*(3*alpha + beta)*(-2*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 0, 3, 0, 3, 0):
        return -1/4*np.pi**(3/2)*Dy*Dz*(AB_product*(4*AB_product**2*Dy**2*Dz**2 - 6*AB_product*AB_sum*(Dy**2 + Dz**2) + 9*AB_sum**2) + 9*AB_sum**4 - 3*AB_sum**2*(alpha*(-2*AB_product*Dy**2 + 3*alpha + 3*beta) + beta*(-2*AB_product*Dz**2 + 3*alpha + 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 0, 3, 0, 2, 1):
        return (1/8)*np.pi**(3/2)*(-AB_product*(8*AB_product**3*Dy**2*Dz**4 - 4*AB_product**2*AB_sum*Dz**2*(6*Dy**2 + Dz**2) + 6*AB_product*AB_sum**2*(Dy**2 + 2*Dz**2) - 3*AB_sum**3) - 3*AB_sum**4*(2*AB_product*Dz**2 - AB_sum) - 3*AB_sum**2*alpha*(4*AB_product**2*Dy**2*Dz**2 - 2*AB_product*AB_sum*(Dy**2 + Dz**2) + AB_sum**2) - AB_sum**2*beta*(4*AB_product**2*Dz**4 - 12*AB_product*AB_sum*Dz**2 + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(15/2))
    if (i, j, k, l, m, n) == (0, 0, 3, 0, 1, 2):
        return -1/4*np.pi**(3/2)*Dy*Dz*(AB_product*(4*AB_product**2*Dz**4 - 20*AB_product*AB_sum*Dz**2 + 15*AB_sum**2) + 3*AB_sum**4 - AB_sum**2*(3*alpha + beta)*(-2*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 0, 3, 0, 0, 3):
        return -1/8*np.pi**(3/2)*(AB_product*(8*AB_product**3*Dz**6 - 60*AB_product**2*Dz**4*(alpha + beta) + 90*AB_product*AB_sum**2*Dz**2 - 15*AB_sum**3) + 9*AB_sum**4*(2*AB_product*Dz**2 - AB_sum) + 3*AB_sum**2*alpha*(4*AB_product**2*Dz**4 - 12*AB_product*AB_sum*Dz**2 + 3*AB_sum**2) + 3*AB_sum**2*beta*(4*AB_product**2*Dz**4 - 12*AB_product*AB_sum*Dz**2 + 3*AB_sum**2))*np.exp(-AB_product*D2/AB_sum)/(AB_product*AB_sum**(15/2))
    raise KeyError((i, j, k, l, m, n))
