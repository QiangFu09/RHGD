"""Synthetic quadratic problems used in the numerical experiments."""

import numpy as np


def num_iter(kappa):
    if kappa == 1000.0:
        return 1000.0
    elif kappa == 100000.0:
        return 10000.0
    elif kappa == 10000000.0:
        return 1000000.0
    else:
        return 10000.0

def generate_spd_matrix_fixed_max(dim, cond_num, lambda_max):
    (Q, _) = np.linalg.qr(np.random.randn(dim, dim))
    lambda_min = lambda_max / cond_num
    eigvals = np.linspace(lambda_max, lambda_min, dim)
    A = Q @ np.diag(eigvals) @ Q.T
    return A

def generate_symmetric_matrix_with_zero_min_eig(dim, lambda_max, seed=None):
    if seed is not None:
        np.random.seed(seed)
    (Q, _) = np.linalg.qr(np.random.randn(dim, dim))
    eigvals = np.linspace(0, lambda_max, dim)
    eigvals[-1] = lambda_max
    eigvals[0] = 0
    A = Q @ np.diag(eigvals) @ Q.T
    A = 0.5 * (A + A.T)
    return A
