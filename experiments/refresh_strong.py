"""Refresh strong experiment."""

import numpy as np
import matplotlib.pyplot as plt

from algorithms import constant_refresh_quadratic
from problems import generate_spd_matrix_fixed_max
from plotting import plot_rhed_vs_poisson_refresh


def run(config):
    plt.style.use('default')
    np.random.seed(42)
    dim = config.dimension
    x0 = np.random.randn(dim)
    num_iters = config.iterations if config.iterations is not None else int(100000.0)
    kappa = config.kappa
    A = generate_spd_matrix_fixed_max(dim, kappa, config.smoothness)
    alpha = config.smoothness / kappa
    methods = constant_refresh_quadratic(A, dim)
    quad_loss = methods.loss
    quad_grad = methods.gradient
    run_rhgd = methods.rhgd
    gamma = config.step_size if config.step_size is not None else 0.002
    h = np.sqrt(gamma)
    se = np.random.default_rng(seed=0)
    (_, lrhgd, refresh1) = run_rhgd(x0, num_iters, h, alpha, rng=se)
    plot_rhed_vs_poisson_refresh(lrhgd, refresh1, h, alpha, label='RHGD')
    fields = ("lgd", "lnag", "losses_cnag_all", "losses_rhgd_all", "losses_rhgd1_all", "lrhgd", "refresh1", "f_star", "alpha", "L", "eta", "h", "num_iters", "comparisons")
    state = locals()
    return {key: state[key] for key in fields if key in state}
