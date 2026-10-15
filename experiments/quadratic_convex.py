"""Quadratic convex experiment."""

import numpy as np
import matplotlib.pyplot as plt

from algorithms import convex_quadratic
from problems import generate_symmetric_matrix_with_zero_min_eig
from plotting import format_sci_latex, compute_log_curve_stats


def run(config):
    plt.style.use('default')
    np.random.seed(42)
    dim = config.dimension
    x0 = np.random.randn(dim)
    num_iters = config.iterations if config.iterations is not None else int(10000.0)
    lamda_max = config.smoothness
    A = generate_symmetric_matrix_with_zero_min_eig(dim, lamda_max, seed=42)
    L = lamda_max
    methods = convex_quadratic(A, dim)
    quad_loss = methods.loss
    quad_grad = methods.gradient
    run_gd = methods.gd
    run_nag = methods.agd
    stochastic_accelerated_descent = methods.cagd
    run_rhgd = methods.rhgd
    eta = config.step_size if config.step_size is not None else 0.002
    h = np.sqrt(eta)
    (losses_gd_all, losses_nag_all, losses_cnag_all, losses_rhgd_all) = ([], [], [], [])
    for i in range(config.trials):
        rng = np.random.default_rng(seed=i)
        (_, lcnag) = stochastic_accelerated_descent(x0, eta, num_iters, rng=rng)
        (_, lrhgd) = run_rhgd(x0, num_iters, h, rng=rng)
        losses_cnag_all.append(lcnag)
        losses_rhgd_all.append(lrhgd)
    (_, lgd) = run_gd(x0, num_iters, eta)
    (_, lnag) = run_nag(x0, num_iters, eta)
    losses_cnag_all = np.array(losses_cnag_all)
    losses_rhgd_all = np.array(losses_rhgd_all)
    (mean_cnag, lower_cnag, upper_cnag) = compute_log_curve_stats(losses_cnag_all)
    (mean_rhgd, lower_rhgd, upper_rhgd) = compute_log_curve_stats(losses_rhgd_all)
    for loss_curve in [lgd, lnag, mean_cnag, mean_rhgd]:
        loss_curve[0] = quad_loss(x0)
    iters = np.arange(num_iters)
    plt.figure(figsize=(10, 6))
    plt.plot(iters, lgd, linestyle='-.', label='GD', color='firebrick', linewidth=5)
    plt.plot(iters, lnag, linestyle='-.', label='AGD', color='royalblue', linewidth=5)
    plt.plot(iters, mean_cnag, linestyle='-.', label='CAGD', color='seagreen', linewidth=5)
    plt.fill_between(iters, lower_cnag, upper_cnag, color='seagreen', alpha=0.3)
    plt.plot(iters, mean_rhgd, label='RHGD (ours)', color='darkorange', linewidth=5)
    plt.fill_between(iters, lower_rhgd, upper_rhgd, color='darkorange', alpha=0.3)
    plt.xlabel('Iteration $k$', fontsize=25)
    plt.ylabel('$f(x_k)-f(x^*)$', fontsize=25)
    plt.title(f'$\\alpha=0,\\ L={format_sci_latex(L)}$', fontsize=25)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.legend(fontsize=23, frameon=True, fancybox=True, facecolor='white', edgecolor='black', framealpha=1.0)
    plt.tight_layout()
    fields = ("lgd", "lnag", "losses_cnag_all", "losses_rhgd_all", "losses_rhgd1_all", "lrhgd", "refresh1", "f_star", "alpha", "L", "eta", "h", "num_iters", "comparisons")
    state = locals()
    return {key: state[key] for key in fields if key in state}
