import numpy as np
from numba import njit
@njit(cache=True, fastmath=True)
def T(i, j, k, l, m, n, Dx, Dy, Dz, D2, AB_sum, AB_product,alpha, beta):
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 0, 0):
        return np.pi**(3/2)*AB_product*(-2*AB_product*D2 + 3*alpha + 3*beta)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(7/2)
    if (i, j, k, l, m, n) == (0, 0, 0, 1, 0, 0):
        return np.pi**(3/2)*AB_product*Dx*alpha*(-2*AB_product*D2 + 5*alpha + 5*beta)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 1, 0):
        return np.pi**(3/2)*AB_product*Dy*alpha*(-2*AB_product*D2 + 5*alpha + 5*beta)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 0, 1):
        return np.pi**(3/2)*AB_product*Dz*alpha*(-2*AB_product*D2 + 5*alpha + 5*beta)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (0, 0, 0, 2, 0, 0):
        return (1/2)*np.pi**(3/2)*alpha*(AB_sum**2*(-2*AB_product*D2 + 3*alpha + 3*beta) + alpha*(2*AB_product*Dx**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 2*AB_sum**2 + AB_sum*(2*AB_product*D2 + 8*AB_product*Dx**2 - 3*alpha - 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 0, 0, 1, 1, 0):
        return np.pi**(3/2)*AB_product*Dx*Dy*alpha**2*(-2*AB_product*D2 + 7*alpha + 7*beta)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 0, 0, 1, 0, 1):
        return np.pi**(3/2)*AB_product*Dx*Dz*alpha**2*(-2*AB_product*D2 + 7*alpha + 7*beta)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 2, 0):
        return (1/2)*np.pi**(3/2)*alpha*(AB_sum**2*(-2*AB_product*D2 + 3*alpha + 3*beta) + alpha*(2*AB_product*Dy**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 2*AB_sum**2 + AB_sum*(2*AB_product*D2 + 8*AB_product*Dy**2 - 3*alpha - 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 1, 1):
        return np.pi**(3/2)*AB_product*Dy*Dz*alpha**2*(-2*AB_product*D2 + 7*alpha + 7*beta)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 0, 0, 0, 0, 2):
        return (1/2)*np.pi**(3/2)*alpha*(AB_sum**2*(-2*AB_product*D2 + 3*alpha + 3*beta) + alpha*(2*AB_product*Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 2*AB_sum**2 + AB_sum*(2*AB_product*D2 + 8*AB_product*Dz**2 - 3*alpha - 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 0, 0):
        return np.pi**(3/2)*AB_product*Dx*beta*(2*AB_product*D2 - 5*alpha - 5*beta)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 1, 0, 0):
        return -1/2*np.pi**(3/2)*AB_product*(2*AB_product*Dx**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 2*AB_sum**2 + AB_sum*(2*AB_product*D2 + 8*AB_product*Dx**2 - 3*alpha - 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 1, 0):
        return np.pi**(3/2)*AB_product**2*Dx*Dy*(2*AB_product*D2 - 7*alpha - 7*beta)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 0, 1):
        return np.pi**(3/2)*AB_product**2*Dx*Dz*(2*AB_product*D2 - 7*alpha - 7*beta)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 2, 0, 0):
        return -1/2*np.pi**(3/2)*AB_product*Dx*(AB_sum**2*(-2*AB_product*D2 + 5*alpha + 5*beta) + alpha*(2*AB_product*Dx**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 12*AB_sum**2 + 3*AB_sum*(2*AB_product*D2 + 4*AB_product*Dx**2 - 3*alpha - 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 1, 1, 0):
        return -1/2*np.pi**(3/2)*AB_product*Dy*alpha*(2*AB_product*Dx**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 4*AB_sum**2 + AB_sum*(2*AB_product*D2 + 12*AB_product*Dx**2 - 3*alpha - 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 1, 0, 1):
        return -1/2*np.pi**(3/2)*AB_product*Dz*alpha*(2*AB_product*Dx**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 4*AB_sum**2 + AB_sum*(2*AB_product*D2 + 12*AB_product*Dx**2 - 3*alpha - 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 2, 0):
        return -1/2*np.pi**(3/2)*AB_product*Dx*(AB_sum**2*(-2*AB_product*D2 + 5*alpha + 5*beta) + alpha*(2*AB_product*Dy**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 4*AB_sum**2 + AB_sum*(2*AB_product*D2 + 12*AB_product*Dy**2 - 3*alpha - 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 1, 1):
        return np.pi**(3/2)*AB_product**2*Dx*Dy*Dz*alpha*(2*AB_product*D2 - 9*alpha - 9*beta)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 0, 0, 0, 0, 2):
        return -1/2*np.pi**(3/2)*AB_product*Dx*(AB_sum**2*(-2*AB_product*D2 + 5*alpha + 5*beta) + alpha*(2*AB_product*Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 4*AB_sum**2 + AB_sum*(2*AB_product*D2 + 12*AB_product*Dz**2 - 3*alpha - 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 0, 0):
        return np.pi**(3/2)*AB_product*Dy*beta*(2*AB_product*D2 - 5*alpha - 5*beta)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 1, 0, 0):
        return np.pi**(3/2)*AB_product**2*Dx*Dy*(2*AB_product*D2 - 7*alpha - 7*beta)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 1, 0):
        return -1/2*np.pi**(3/2)*AB_product*(2*AB_product*Dy**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 2*AB_sum**2 + AB_sum*(2*AB_product*D2 + 8*AB_product*Dy**2 - 3*alpha - 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 0, 1):
        return np.pi**(3/2)*AB_product**2*Dy*Dz*(2*AB_product*D2 - 7*alpha - 7*beta)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 2, 0, 0):
        return -1/2*np.pi**(3/2)*AB_product*Dy*(AB_sum**2*(-2*AB_product*D2 + 5*alpha + 5*beta) + alpha*(2*AB_product*Dx**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 4*AB_sum**2 + AB_sum*(2*AB_product*D2 + 12*AB_product*Dx**2 - 3*alpha - 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 1, 1, 0):
        return -1/2*np.pi**(3/2)*AB_product*Dx*alpha*(2*AB_product*Dy**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 4*AB_sum**2 + AB_sum*(2*AB_product*D2 + 12*AB_product*Dy**2 - 3*alpha - 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 1, 0, 1):
        return np.pi**(3/2)*AB_product**2*Dx*Dy*Dz*alpha*(2*AB_product*D2 - 9*alpha - 9*beta)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 2, 0):
        return -1/2*np.pi**(3/2)*AB_product*Dy*(AB_sum**2*(-2*AB_product*D2 + 5*alpha + 5*beta) + alpha*(2*AB_product*Dy**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 12*AB_sum**2 + 3*AB_sum*(2*AB_product*D2 + 4*AB_product*Dy**2 - 3*alpha - 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 1, 1):
        return -1/2*np.pi**(3/2)*AB_product*Dz*alpha*(2*AB_product*Dy**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 4*AB_sum**2 + AB_sum*(2*AB_product*D2 + 12*AB_product*Dy**2 - 3*alpha - 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 1, 0, 0, 0, 2):
        return -1/2*np.pi**(3/2)*AB_product*Dy*(AB_sum**2*(-2*AB_product*D2 + 5*alpha + 5*beta) + alpha*(2*AB_product*Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 4*AB_sum**2 + AB_sum*(2*AB_product*D2 + 12*AB_product*Dz**2 - 3*alpha - 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 0, 0):
        return np.pi**(3/2)*AB_product*Dz*beta*(2*AB_product*D2 - 5*alpha - 5*beta)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(9/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 1, 0, 0):
        return np.pi**(3/2)*AB_product**2*Dx*Dz*(2*AB_product*D2 - 7*alpha - 7*beta)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 1, 0):
        return np.pi**(3/2)*AB_product**2*Dy*Dz*(2*AB_product*D2 - 7*alpha - 7*beta)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 0, 1):
        return -1/2*np.pi**(3/2)*AB_product*(2*AB_product*Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 2*AB_sum**2 + AB_sum*(2*AB_product*D2 + 8*AB_product*Dz**2 - 3*alpha - 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 2, 0, 0):
        return -1/2*np.pi**(3/2)*AB_product*Dz*(AB_sum**2*(-2*AB_product*D2 + 5*alpha + 5*beta) + alpha*(2*AB_product*Dx**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 4*AB_sum**2 + AB_sum*(2*AB_product*D2 + 12*AB_product*Dx**2 - 3*alpha - 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 1, 1, 0):
        return np.pi**(3/2)*AB_product**2*Dx*Dy*Dz*alpha*(2*AB_product*D2 - 9*alpha - 9*beta)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 1, 0, 1):
        return -1/2*np.pi**(3/2)*AB_product*Dx*alpha*(2*AB_product*Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 4*AB_sum**2 + AB_sum*(2*AB_product*D2 + 12*AB_product*Dz**2 - 3*alpha - 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 2, 0):
        return -1/2*np.pi**(3/2)*AB_product*Dz*(AB_sum**2*(-2*AB_product*D2 + 5*alpha + 5*beta) + alpha*(2*AB_product*Dy**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 4*AB_sum**2 + AB_sum*(2*AB_product*D2 + 12*AB_product*Dy**2 - 3*alpha - 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 1, 1):
        return -1/2*np.pi**(3/2)*AB_product*Dy*alpha*(2*AB_product*Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 4*AB_sum**2 + AB_sum*(2*AB_product*D2 + 12*AB_product*Dz**2 - 3*alpha - 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 0, 1, 0, 0, 2):
        return -1/2*np.pi**(3/2)*AB_product*Dz*(AB_sum**2*(-2*AB_product*D2 + 5*alpha + 5*beta) + alpha*(2*AB_product*Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 12*AB_sum**2 + 3*AB_sum*(2*AB_product*D2 + 4*AB_product*Dz**2 - 3*alpha - 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (2, 0, 0, 0, 0, 0):
        return (1/2)*np.pi**(3/2)*beta*(AB_sum**2*(-2*AB_product*D2 + 3*alpha + 3*beta) + beta*(2*AB_product*Dx**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 2*AB_sum**2 + AB_sum*(2*AB_product*D2 + 8*AB_product*Dx**2 - 3*alpha - 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (2, 0, 0, 1, 0, 0):
        return (1/2)*np.pi**(3/2)*AB_product*Dx*(AB_sum**2*(-2*AB_product*D2 + 5*alpha + 5*beta) + beta*(2*AB_product*Dx**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 12*AB_sum**2 + 3*AB_sum*(2*AB_product*D2 + 4*AB_product*Dx**2 - 3*alpha - 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (2, 0, 0, 0, 1, 0):
        return (1/2)*np.pi**(3/2)*AB_product*Dy*(AB_sum**2*(-2*AB_product*D2 + 5*alpha + 5*beta) + beta*(2*AB_product*Dx**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 4*AB_sum**2 + AB_sum*(2*AB_product*D2 + 12*AB_product*Dx**2 - 3*alpha - 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (2, 0, 0, 0, 0, 1):
        return (1/2)*np.pi**(3/2)*AB_product*Dz*(AB_sum**2*(-2*AB_product*D2 + 5*alpha + 5*beta) + beta*(2*AB_product*Dx**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 4*AB_sum**2 + AB_sum*(2*AB_product*D2 + 12*AB_product*Dx**2 - 3*alpha - 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (2, 0, 0, 2, 0, 0):
        return (1/4)*np.pi**(3/2)*(AB_product*(4*AB_product**2*Dx**4*(-2*AB_product*D2 + 3*alpha + 3*beta) + 4*AB_product*AB_sum*Dx**2*(6*AB_product*D2 + 8*AB_product*Dx**2 - 9*alpha - 9*beta) + 12*AB_sum**3 + 3*AB_sum**2*(-2*AB_product*D2 - 24*AB_product*Dx**2 + 3*alpha + 3*beta)) + AB_sum**4*(-2*AB_product*D2 + 3*alpha + 3*beta) + AB_sum**3*(2*AB_product*Dx**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 2*AB_sum**2 + AB_sum*(2*AB_product*D2 + 8*AB_product*Dx**2 - 3*alpha - 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (2, 0, 0, 1, 1, 0):
        return (1/2)*np.pi**(3/2)*AB_product*Dx*Dy*alpha*(AB_sum**2*(-2*AB_product*D2 + 7*alpha + 7*beta) + beta*(2*AB_product*Dx**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 18*AB_sum**2 + AB_sum*(6*AB_product*D2 + 16*AB_product*Dx**2 - 9*alpha - 9*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (2, 0, 0, 1, 0, 1):
        return (1/2)*np.pi**(3/2)*AB_product*Dx*Dz*alpha*(AB_sum**2*(-2*AB_product*D2 + 7*alpha + 7*beta) + beta*(2*AB_product*Dx**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 18*AB_sum**2 + AB_sum*(6*AB_product*D2 + 16*AB_product*Dx**2 - 9*alpha - 9*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (2, 0, 0, 0, 2, 0):
        return (1/4)*np.pi**(3/2)*(AB_product*(4*AB_product**2*Dx**2*Dy**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 2*AB_product*AB_sum*(-16*AB_product*Dx**2*Dy**2 + Dx**2*(-2*AB_product*D2 + 3*alpha + 3*beta) + Dy**2*(-2*AB_product*D2 + 3*alpha + 3*beta)) + 4*AB_sum**3 + AB_sum**2*(-2*AB_product*D2 - 12*AB_product*Dx**2 - 12*AB_product*Dy**2 + 3*alpha + 3*beta)) + AB_sum**4*(-2*AB_product*D2 + 3*alpha + 3*beta) + AB_sum**2*(alpha*(2*AB_product*Dy**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 2*AB_sum**2 + AB_sum*(2*AB_product*D2 + 8*AB_product*Dy**2 - 3*alpha - 3*beta)) + beta*(2*AB_product*Dx**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 2*AB_sum**2 + AB_sum*(2*AB_product*D2 + 8*AB_product*Dx**2 - 3*alpha - 3*beta))))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (2, 0, 0, 0, 1, 1):
        return (1/2)*np.pi**(3/2)*AB_product*Dy*Dz*alpha*(AB_sum**2*(-2*AB_product*D2 + 7*alpha + 7*beta) + beta*(2*AB_product*Dx**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 6*AB_sum**2 + AB_sum*(2*AB_product*D2 + 16*AB_product*Dx**2 - 3*alpha - 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (2, 0, 0, 0, 0, 2):
        return (1/4)*np.pi**(3/2)*(AB_product*(4*AB_product**2*Dx**2*Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 2*AB_product*AB_sum*(-16*AB_product*Dx**2*Dz**2 + Dx**2*(-2*AB_product*D2 + 3*alpha + 3*beta) + Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta)) + 4*AB_sum**3 + AB_sum**2*(-2*AB_product*D2 - 12*AB_product*Dx**2 - 12*AB_product*Dz**2 + 3*alpha + 3*beta)) + AB_sum**4*(-2*AB_product*D2 + 3*alpha + 3*beta) + AB_sum**2*(alpha*(2*AB_product*Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 2*AB_sum**2 + AB_sum*(2*AB_product*D2 + 8*AB_product*Dz**2 - 3*alpha - 3*beta)) + beta*(2*AB_product*Dx**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 2*AB_sum**2 + AB_sum*(2*AB_product*D2 + 8*AB_product*Dx**2 - 3*alpha - 3*beta))))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 0, 0, 0):
        return np.pi**(3/2)*AB_product*Dx*Dy*beta**2*(-2*AB_product*D2 + 7*alpha + 7*beta)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 1, 0, 0):
        return (1/2)*np.pi**(3/2)*AB_product*Dy*beta*(2*AB_product*Dx**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 4*AB_sum**2 + AB_sum*(2*AB_product*D2 + 12*AB_product*Dx**2 - 3*alpha - 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 0, 1, 0):
        return (1/2)*np.pi**(3/2)*AB_product*Dx*beta*(2*AB_product*Dy**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 4*AB_sum**2 + AB_sum*(2*AB_product*D2 + 12*AB_product*Dy**2 - 3*alpha - 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 0, 0, 1):
        return np.pi**(3/2)*AB_product**2*Dx*Dy*Dz*beta*(-2*AB_product*D2 + 9*alpha + 9*beta)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 2, 0, 0):
        return (1/2)*np.pi**(3/2)*AB_product*Dx*Dy*beta*(AB_sum**2*(-2*AB_product*D2 + 7*alpha + 7*beta) + alpha*(2*AB_product*Dx**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 18*AB_sum**2 + AB_sum*(6*AB_product*D2 + 16*AB_product*Dx**2 - 9*alpha - 9*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 1, 1, 0):
        return (1/4)*np.pi**(3/2)*AB_product*(AB_product**2*Dx**2*Dy**2*(-8*AB_product*D2 + 12*alpha + 12*beta) - 2*AB_product*AB_sum*(-16*AB_product*Dx**2*Dy**2 + Dx**2*(-2*AB_product*D2 + 3*alpha + 3*beta) + Dy**2*(-2*AB_product*D2 + 3*alpha + 3*beta)) + 4*AB_sum**3 + AB_sum**2*(-2*AB_product*D2 - 12*AB_product*Dx**2 - 12*AB_product*Dy**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 1, 0, 1):
        return (1/2)*np.pi**(3/2)*AB_product**2*Dy*Dz*(2*AB_product*Dx**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 6*AB_sum**2 + AB_sum*(2*AB_product*D2 + 16*AB_product*Dx**2 - 3*alpha - 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 0, 2, 0):
        return (1/2)*np.pi**(3/2)*AB_product*Dx*Dy*beta*(AB_sum**2*(-2*AB_product*D2 + 7*alpha + 7*beta) + alpha*(2*AB_product*Dy**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 18*AB_sum**2 + AB_sum*(6*AB_product*D2 + 16*AB_product*Dy**2 - 9*alpha - 9*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 0, 1, 1):
        return (1/2)*np.pi**(3/2)*AB_product**2*Dx*Dz*(2*AB_product*Dy**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 6*AB_sum**2 + AB_sum*(2*AB_product*D2 + 16*AB_product*Dy**2 - 3*alpha - 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 1, 0, 0, 0, 2):
        return (1/2)*np.pi**(3/2)*AB_product*Dx*Dy*beta*(AB_sum**2*(-2*AB_product*D2 + 7*alpha + 7*beta) + alpha*(2*AB_product*Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 6*AB_sum**2 + AB_sum*(2*AB_product*D2 + 16*AB_product*Dz**2 - 3*alpha - 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 0, 0, 0):
        return np.pi**(3/2)*AB_product*Dx*Dz*beta**2*(-2*AB_product*D2 + 7*alpha + 7*beta)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 1, 0, 0):
        return (1/2)*np.pi**(3/2)*AB_product*Dz*beta*(2*AB_product*Dx**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 4*AB_sum**2 + AB_sum*(2*AB_product*D2 + 12*AB_product*Dx**2 - 3*alpha - 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 0, 1, 0):
        return np.pi**(3/2)*AB_product**2*Dx*Dy*Dz*beta*(-2*AB_product*D2 + 9*alpha + 9*beta)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 0, 0, 1):
        return (1/2)*np.pi**(3/2)*AB_product*Dx*beta*(2*AB_product*Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 4*AB_sum**2 + AB_sum*(2*AB_product*D2 + 12*AB_product*Dz**2 - 3*alpha - 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 2, 0, 0):
        return (1/2)*np.pi**(3/2)*AB_product*Dx*Dz*beta*(AB_sum**2*(-2*AB_product*D2 + 7*alpha + 7*beta) + alpha*(2*AB_product*Dx**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 18*AB_sum**2 + AB_sum*(6*AB_product*D2 + 16*AB_product*Dx**2 - 9*alpha - 9*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 1, 1, 0):
        return (1/2)*np.pi**(3/2)*AB_product**2*Dy*Dz*(2*AB_product*Dx**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 6*AB_sum**2 + AB_sum*(2*AB_product*D2 + 16*AB_product*Dx**2 - 3*alpha - 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 1, 0, 1):
        return (1/4)*np.pi**(3/2)*AB_product*(AB_product**2*Dx**2*Dz**2*(-8*AB_product*D2 + 12*alpha + 12*beta) - 2*AB_product*AB_sum*(-16*AB_product*Dx**2*Dz**2 + Dx**2*(-2*AB_product*D2 + 3*alpha + 3*beta) + Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta)) + 4*AB_sum**3 + AB_sum**2*(-2*AB_product*D2 - 12*AB_product*Dx**2 - 12*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 0, 2, 0):
        return (1/2)*np.pi**(3/2)*AB_product*Dx*Dz*beta*(AB_sum**2*(-2*AB_product*D2 + 7*alpha + 7*beta) + alpha*(2*AB_product*Dy**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 6*AB_sum**2 + AB_sum*(2*AB_product*D2 + 16*AB_product*Dy**2 - 3*alpha - 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 0, 1, 1):
        return (1/2)*np.pi**(3/2)*AB_product**2*Dx*Dy*(2*AB_product*Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 6*AB_sum**2 + AB_sum*(2*AB_product*D2 + 16*AB_product*Dz**2 - 3*alpha - 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (1, 0, 1, 0, 0, 2):
        return (1/2)*np.pi**(3/2)*AB_product*Dx*Dz*beta*(AB_sum**2*(-2*AB_product*D2 + 7*alpha + 7*beta) + alpha*(2*AB_product*Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 18*AB_sum**2 + AB_sum*(6*AB_product*D2 + 16*AB_product*Dz**2 - 9*alpha - 9*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 2, 0, 0, 0, 0):
        return (1/2)*np.pi**(3/2)*beta*(AB_sum**2*(-2*AB_product*D2 + 3*alpha + 3*beta) + beta*(2*AB_product*Dy**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 2*AB_sum**2 + AB_sum*(2*AB_product*D2 + 8*AB_product*Dy**2 - 3*alpha - 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 2, 0, 1, 0, 0):
        return (1/2)*np.pi**(3/2)*AB_product*Dx*(AB_sum**2*(-2*AB_product*D2 + 5*alpha + 5*beta) + beta*(2*AB_product*Dy**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 4*AB_sum**2 + AB_sum*(2*AB_product*D2 + 12*AB_product*Dy**2 - 3*alpha - 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 2, 0, 0, 1, 0):
        return (1/2)*np.pi**(3/2)*AB_product*Dy*(AB_sum**2*(-2*AB_product*D2 + 5*alpha + 5*beta) + beta*(2*AB_product*Dy**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 12*AB_sum**2 + 3*AB_sum*(2*AB_product*D2 + 4*AB_product*Dy**2 - 3*alpha - 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 2, 0, 0, 0, 1):
        return (1/2)*np.pi**(3/2)*AB_product*Dz*(AB_sum**2*(-2*AB_product*D2 + 5*alpha + 5*beta) + beta*(2*AB_product*Dy**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 4*AB_sum**2 + AB_sum*(2*AB_product*D2 + 12*AB_product*Dy**2 - 3*alpha - 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 2, 0, 2, 0, 0):
        return (1/4)*np.pi**(3/2)*(AB_product*(4*AB_product**2*Dx**2*Dy**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 2*AB_product*AB_sum*(-16*AB_product*Dx**2*Dy**2 + Dx**2*(-2*AB_product*D2 + 3*alpha + 3*beta) + Dy**2*(-2*AB_product*D2 + 3*alpha + 3*beta)) + 4*AB_sum**3 + AB_sum**2*(-2*AB_product*D2 - 12*AB_product*Dx**2 - 12*AB_product*Dy**2 + 3*alpha + 3*beta)) + AB_sum**4*(-2*AB_product*D2 + 3*alpha + 3*beta) + AB_sum**2*(alpha*(2*AB_product*Dx**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 2*AB_sum**2 + AB_sum*(2*AB_product*D2 + 8*AB_product*Dx**2 - 3*alpha - 3*beta)) + beta*(2*AB_product*Dy**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 2*AB_sum**2 + AB_sum*(2*AB_product*D2 + 8*AB_product*Dy**2 - 3*alpha - 3*beta))))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 2, 0, 1, 1, 0):
        return (1/2)*np.pi**(3/2)*AB_product*Dx*Dy*alpha*(AB_sum**2*(-2*AB_product*D2 + 7*alpha + 7*beta) + beta*(2*AB_product*Dy**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 18*AB_sum**2 + AB_sum*(6*AB_product*D2 + 16*AB_product*Dy**2 - 9*alpha - 9*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 2, 0, 1, 0, 1):
        return (1/2)*np.pi**(3/2)*AB_product*Dx*Dz*alpha*(AB_sum**2*(-2*AB_product*D2 + 7*alpha + 7*beta) + beta*(2*AB_product*Dy**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 6*AB_sum**2 + AB_sum*(2*AB_product*D2 + 16*AB_product*Dy**2 - 3*alpha - 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 2, 0, 0, 2, 0):
        return (1/4)*np.pi**(3/2)*(AB_product*(4*AB_product**2*Dy**4*(-2*AB_product*D2 + 3*alpha + 3*beta) + 4*AB_product*AB_sum*Dy**2*(6*AB_product*D2 + 8*AB_product*Dy**2 - 9*alpha - 9*beta) + 12*AB_sum**3 + 3*AB_sum**2*(-2*AB_product*D2 - 24*AB_product*Dy**2 + 3*alpha + 3*beta)) + AB_sum**4*(-2*AB_product*D2 + 3*alpha + 3*beta) + AB_sum**3*(2*AB_product*Dy**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 2*AB_sum**2 + AB_sum*(2*AB_product*D2 + 8*AB_product*Dy**2 - 3*alpha - 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 2, 0, 0, 1, 1):
        return (1/2)*np.pi**(3/2)*AB_product*Dy*Dz*alpha*(AB_sum**2*(-2*AB_product*D2 + 7*alpha + 7*beta) + beta*(2*AB_product*Dy**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 18*AB_sum**2 + AB_sum*(6*AB_product*D2 + 16*AB_product*Dy**2 - 9*alpha - 9*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 2, 0, 0, 0, 2):
        return (1/4)*np.pi**(3/2)*(AB_product*(4*AB_product**2*Dy**2*Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 2*AB_product*AB_sum*(-16*AB_product*Dy**2*Dz**2 + Dy**2*(-2*AB_product*D2 + 3*alpha + 3*beta) + Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta)) + 4*AB_sum**3 + AB_sum**2*(-2*AB_product*D2 - 12*AB_product*Dy**2 - 12*AB_product*Dz**2 + 3*alpha + 3*beta)) + AB_sum**4*(-2*AB_product*D2 + 3*alpha + 3*beta) + AB_sum**2*(alpha*(2*AB_product*Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 2*AB_sum**2 + AB_sum*(2*AB_product*D2 + 8*AB_product*Dz**2 - 3*alpha - 3*beta)) + beta*(2*AB_product*Dy**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 2*AB_sum**2 + AB_sum*(2*AB_product*D2 + 8*AB_product*Dy**2 - 3*alpha - 3*beta))))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 0, 0, 0):
        return np.pi**(3/2)*AB_product*Dy*Dz*beta**2*(-2*AB_product*D2 + 7*alpha + 7*beta)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 1, 0, 0):
        return np.pi**(3/2)*AB_product**2*Dx*Dy*Dz*beta*(-2*AB_product*D2 + 9*alpha + 9*beta)*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 0, 1, 0):
        return (1/2)*np.pi**(3/2)*AB_product*Dz*beta*(2*AB_product*Dy**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 4*AB_sum**2 + AB_sum*(2*AB_product*D2 + 12*AB_product*Dy**2 - 3*alpha - 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 0, 0, 1):
        return (1/2)*np.pi**(3/2)*AB_product*Dy*beta*(2*AB_product*Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 4*AB_sum**2 + AB_sum*(2*AB_product*D2 + 12*AB_product*Dz**2 - 3*alpha - 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 2, 0, 0):
        return (1/2)*np.pi**(3/2)*AB_product*Dy*Dz*beta*(AB_sum**2*(-2*AB_product*D2 + 7*alpha + 7*beta) + alpha*(2*AB_product*Dx**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 6*AB_sum**2 + AB_sum*(2*AB_product*D2 + 16*AB_product*Dx**2 - 3*alpha - 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 1, 1, 0):
        return (1/2)*np.pi**(3/2)*AB_product**2*Dx*Dz*(2*AB_product*Dy**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 6*AB_sum**2 + AB_sum*(2*AB_product*D2 + 16*AB_product*Dy**2 - 3*alpha - 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 1, 0, 1):
        return (1/2)*np.pi**(3/2)*AB_product**2*Dx*Dy*(2*AB_product*Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 6*AB_sum**2 + AB_sum*(2*AB_product*D2 + 16*AB_product*Dz**2 - 3*alpha - 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 0, 2, 0):
        return (1/2)*np.pi**(3/2)*AB_product*Dy*Dz*beta*(AB_sum**2*(-2*AB_product*D2 + 7*alpha + 7*beta) + alpha*(2*AB_product*Dy**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 18*AB_sum**2 + AB_sum*(6*AB_product*D2 + 16*AB_product*Dy**2 - 9*alpha - 9*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 0, 1, 1):
        return (1/4)*np.pi**(3/2)*AB_product*(AB_product**2*Dy**2*Dz**2*(-8*AB_product*D2 + 12*alpha + 12*beta) - 2*AB_product*AB_sum*(-16*AB_product*Dy**2*Dz**2 + Dy**2*(-2*AB_product*D2 + 3*alpha + 3*beta) + Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta)) + 4*AB_sum**3 + AB_sum**2*(-2*AB_product*D2 - 12*AB_product*Dy**2 - 12*AB_product*Dz**2 + 3*alpha + 3*beta))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 1, 1, 0, 0, 2):
        return (1/2)*np.pi**(3/2)*AB_product*Dy*Dz*beta*(AB_sum**2*(-2*AB_product*D2 + 7*alpha + 7*beta) + alpha*(2*AB_product*Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 18*AB_sum**2 + AB_sum*(6*AB_product*D2 + 16*AB_product*Dz**2 - 9*alpha - 9*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 0, 2, 0, 0, 0):
        return (1/2)*np.pi**(3/2)*beta*(AB_sum**2*(-2*AB_product*D2 + 3*alpha + 3*beta) + beta*(2*AB_product*Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 2*AB_sum**2 + AB_sum*(2*AB_product*D2 + 8*AB_product*Dz**2 - 3*alpha - 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(11/2)
    if (i, j, k, l, m, n) == (0, 0, 2, 1, 0, 0):
        return (1/2)*np.pi**(3/2)*AB_product*Dx*(AB_sum**2*(-2*AB_product*D2 + 5*alpha + 5*beta) + beta*(2*AB_product*Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 4*AB_sum**2 + AB_sum*(2*AB_product*D2 + 12*AB_product*Dz**2 - 3*alpha - 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 0, 2, 0, 1, 0):
        return (1/2)*np.pi**(3/2)*AB_product*Dy*(AB_sum**2*(-2*AB_product*D2 + 5*alpha + 5*beta) + beta*(2*AB_product*Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 4*AB_sum**2 + AB_sum*(2*AB_product*D2 + 12*AB_product*Dz**2 - 3*alpha - 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 0, 2, 0, 0, 1):
        return (1/2)*np.pi**(3/2)*AB_product*Dz*(AB_sum**2*(-2*AB_product*D2 + 5*alpha + 5*beta) + beta*(2*AB_product*Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 12*AB_sum**2 + 3*AB_sum*(2*AB_product*D2 + 4*AB_product*Dz**2 - 3*alpha - 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(13/2)
    if (i, j, k, l, m, n) == (0, 0, 2, 2, 0, 0):
        return (1/4)*np.pi**(3/2)*(AB_product*(4*AB_product**2*Dx**2*Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 2*AB_product*AB_sum*(-16*AB_product*Dx**2*Dz**2 + Dx**2*(-2*AB_product*D2 + 3*alpha + 3*beta) + Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta)) + 4*AB_sum**3 + AB_sum**2*(-2*AB_product*D2 - 12*AB_product*Dx**2 - 12*AB_product*Dz**2 + 3*alpha + 3*beta)) + AB_sum**4*(-2*AB_product*D2 + 3*alpha + 3*beta) + AB_sum**2*(alpha*(2*AB_product*Dx**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 2*AB_sum**2 + AB_sum*(2*AB_product*D2 + 8*AB_product*Dx**2 - 3*alpha - 3*beta)) + beta*(2*AB_product*Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 2*AB_sum**2 + AB_sum*(2*AB_product*D2 + 8*AB_product*Dz**2 - 3*alpha - 3*beta))))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 0, 2, 1, 1, 0):
        return (1/2)*np.pi**(3/2)*AB_product*Dx*Dy*alpha*(AB_sum**2*(-2*AB_product*D2 + 7*alpha + 7*beta) + beta*(2*AB_product*Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 6*AB_sum**2 + AB_sum*(2*AB_product*D2 + 16*AB_product*Dz**2 - 3*alpha - 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 0, 2, 1, 0, 1):
        return (1/2)*np.pi**(3/2)*AB_product*Dx*Dz*alpha*(AB_sum**2*(-2*AB_product*D2 + 7*alpha + 7*beta) + beta*(2*AB_product*Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 18*AB_sum**2 + AB_sum*(6*AB_product*D2 + 16*AB_product*Dz**2 - 9*alpha - 9*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 0, 2, 0, 2, 0):
        return (1/4)*np.pi**(3/2)*(AB_product*(4*AB_product**2*Dy**2*Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 2*AB_product*AB_sum*(-16*AB_product*Dy**2*Dz**2 + Dy**2*(-2*AB_product*D2 + 3*alpha + 3*beta) + Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta)) + 4*AB_sum**3 + AB_sum**2*(-2*AB_product*D2 - 12*AB_product*Dy**2 - 12*AB_product*Dz**2 + 3*alpha + 3*beta)) + AB_sum**4*(-2*AB_product*D2 + 3*alpha + 3*beta) + AB_sum**2*(alpha*(2*AB_product*Dy**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 2*AB_sum**2 + AB_sum*(2*AB_product*D2 + 8*AB_product*Dy**2 - 3*alpha - 3*beta)) + beta*(2*AB_product*Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 2*AB_sum**2 + AB_sum*(2*AB_product*D2 + 8*AB_product*Dz**2 - 3*alpha - 3*beta))))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 0, 2, 0, 1, 1):
        return (1/2)*np.pi**(3/2)*AB_product*Dy*Dz*alpha*(AB_sum**2*(-2*AB_product*D2 + 7*alpha + 7*beta) + beta*(2*AB_product*Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 18*AB_sum**2 + AB_sum*(6*AB_product*D2 + 16*AB_product*Dz**2 - 9*alpha - 9*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    if (i, j, k, l, m, n) == (0, 0, 2, 0, 0, 2):
        return (1/4)*np.pi**(3/2)*(AB_product*(4*AB_product**2*Dz**4*(-2*AB_product*D2 + 3*alpha + 3*beta) + 4*AB_product*AB_sum*Dz**2*(6*AB_product*D2 + 8*AB_product*Dz**2 - 9*alpha - 9*beta) + 12*AB_sum**3 + 3*AB_sum**2*(-2*AB_product*D2 - 24*AB_product*Dz**2 + 3*alpha + 3*beta)) + AB_sum**4*(-2*AB_product*D2 + 3*alpha + 3*beta) + AB_sum**3*(2*AB_product*Dz**2*(-2*AB_product*D2 + 3*alpha + 3*beta) - 2*AB_sum**2 + AB_sum*(2*AB_product*D2 + 8*AB_product*Dz**2 - 3*alpha - 3*beta)))*np.exp(-AB_product*D2/AB_sum)/AB_sum**(15/2)
    raise KeyError((i, j, k, l, m, n))
