"""Logistic convex experiment."""

import numpy as np
import matplotlib.pyplot as plt

from algorithms import convex_logistic
from plotting import compute_log_curve_stats


def run(config):
    plt.style.use('default')
    dim = config.dimension
    num_samples = 500
    num_iters = config.iterations if config.iterations is not None else int(200.0)
    np.random.seed(42)
    x0 = np.random.randn(dim)
    A = np.random.randn(num_samples, dim)
    true_x = np.random.randn(dim)
    b = np.sign(A @ true_x + 0.1 * np.random.randn(num_samples))
    methods = convex_logistic(A, b, dim)
    logistic_loss = methods.loss
    logistic_grad = methods.gradient
    run_gd = methods.gd
    run_nag = methods.agd
    stochastic_accelerated_descent = methods.cagd
    run_rhgd = methods.rhgd
    eta = config.step_size if config.step_size is not None else 1
    h = np.sqrt(eta)
    (losses_gd_all, losses_nag_all, losses_cnag_all, losses_rhgd_all, losses_rhgd1_all) = ([], [], [], [], [])
    for i in range(config.trials):
        rng = np.random.default_rng(seed=i)
        (_, lrhgd) = run_rhgd(x0, num_iters, h, rng=rng)
        (_, lcnag) = stochastic_accelerated_descent(x0, eta, num_iters, rng=rng)
        losses_rhgd_all.append(lrhgd)
        losses_cnag_all.append(lcnag)
    (_, lgd) = run_gd(x0, num_iters, eta)
    (_, lnag) = run_nag(x0, num_iters, eta)
    losses_cnag_all = np.array(losses_cnag_all)
    losses_rhgd_all = np.array(losses_rhgd_all)
    (mean_cnag, lower_cnag, upper_cnag) = compute_log_curve_stats(losses_cnag_all)
    (mean_rhgd, lower_rhgd, upper_rhgd) = compute_log_curve_stats(losses_rhgd_all)
    for loss_curve in [lgd, lnag, mean_cnag, mean_rhgd]:
        loss_curve[0] = logistic_loss(x0)
    iters = np.arange(num_iters)
    plt.figure(figsize=(10, 6))
    plt.plot(iters, lgd, linestyle='-.', label='GD', color='firebrick', linewidth=5)
    plt.plot(iters, lnag, linestyle='-.', label='AGD', color='royalblue', linewidth=5)
    plt.plot(iters, mean_cnag, linestyle='-.', label='CAGD', color='seagreen', linewidth=5)
    plt.fill_between(iters, lower_cnag, upper_cnag, color='seagreen', alpha=0.3)
    plt.plot(iters, mean_rhgd, label='RHGD (ours)', color='darkorange', linewidth=5)
    plt.fill_between(iters, lower_rhgd, upper_rhgd, color='darkorange', alpha=0.3)
    plt.xlabel('Iteration $k$', fontsize=25)
    plt.ylabel('$f(x_k)-f^*$', fontsize=25)
    plt.title(f'$\\alpha=0$', fontsize=25)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.tick_params(axis='both', which='both', labelsize=16, length=6, width=1.5)
    plt.legend(fontsize=20)
    plt.legend(fontsize=23, frameon=True, fancybox=True, facecolor='white', edgecolor='black', framealpha=1.0)
    plt.tight_layout()
    fields = ("lgd", "lnag", "losses_cnag_all", "losses_rhgd_all", "losses_rhgd1_all", "lrhgd", "refresh1", "f_star", "alpha", "L", "eta", "h", "num_iters", "comparisons")
    state = locals()
    return {key: state[key] for key in fields if key in state}
