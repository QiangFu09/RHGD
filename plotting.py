"""Loss aggregation and refresh-event diagnostics."""

import numpy as np
import matplotlib.pyplot as plt


def format_sci_latex(val):
    coeff = val / 10 ** int(np.floor(np.log10(val)))
    exp = int(np.floor(np.log10(val)))
    return f'{coeff:.1f}\\times 10^{{{exp}}}'

def compute_log_curve_stats(losses_all, f_star=None):
    """Return the arithmetic mean and delta-method envelopes in log10 space."""
    losses_all = np.array(losses_all)
    if f_star is not None:
        shifted = losses_all - f_star
        shifted = np.clip(shifted, 1e-20, None)
    else:
        shifted = losses_all
    mean_loss = np.mean(shifted, axis=0)
    std_loss = np.std(shifted, axis=0)
    logmean = np.log10(mean_loss)
    logstd = std_loss / (mean_loss * np.log(10))
    log_lower = logmean - logstd
    log_upper = logmean + logstd
    lower = 10 ** log_lower
    upper = 10 ** log_upper
    return (mean_loss, lower, upper)

def plot_rhed_vs_poisson_refresh(losses, refresh_points, h, mu, label='RHGD'):
    """Plots RHGD loss curve with actual refresh points and Poisson jump times on the same x-axis."""
    num_iters = len(losses)
    time_points = np.arange(num_iters) * h
    rate = np.sqrt(mu)
    T_max = time_points[-1]
    poisson_jumps = []
    t = 0
    while t < T_max:
        dt = np.random.exponential(scale=1 / rate)
        t += dt
        poisson_jumps.append(t)
    refresh_times = [time_points[i] for i in refresh_points]
    refresh_losses = [losses[i] for i in refresh_points]
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, losses, label=label, color='darkorange', linewidth=5, zorder=1)
    plt.scatter(refresh_times, refresh_losses, color='blue', s=40, label='Refresh (actual)', zorder=2)
    for (i, t) in enumerate(poisson_jumps):
        if t <= T_max:
            plt.axvline(x=t, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Homogeneous Poisson Jump' if i == 0 else '')
    plt.xlabel('Time$=$iteration $\\times$ stepsize $h$', fontsize=20)
    plt.ylabel('Objective', fontsize=20)
    plt.yscale('log')
    plt.grid(True)
    plt.tick_params(axis='both', which='both', labelsize=16, length=6, width=1.5)
    plt.legend(fontsize=23, frameon=True, fancybox=True, facecolor='white', edgecolor='black', framealpha=1.0)
    plt.title('RHGD Refreshment vs. Homogeneous Poisson Jump', fontsize=20)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    pass

def simulate_inhomogeneous_poisson(T_max, h, lambda_max=1.0):
    """Simulate an inhomogeneous Poisson process with rate λ(t) = 1 / (t + 2h)"""
    t = 0
    jump_times = []
    while t < T_max:
        u = np.random.exponential(1 / lambda_max)
        t += u
        if t >= T_max:
            break
        lambda_t = 17 / (2 * t + 18 * h)
        if np.random.rand() < lambda_t / lambda_max:
            jump_times.append(t)
    return jump_times

def plot_rhed_vs_inhomogeneous_poisson(losses, refresh_points, h, label='RHGD'):
    num_iters = len(losses)
    time_points = np.arange(num_iters) * h
    T_max = time_points[-1]
    poisson_jumps = simulate_inhomogeneous_poisson(T_max, h)
    refresh_times = [time_points[i] for i in refresh_points]
    refresh_losses = [losses[i] for i in refresh_points]
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, losses, label=label, color='darkorange', linewidth=5, zorder=1)
    plt.scatter(refresh_times, refresh_losses, color='blue', s=40, label='Refresh (actual)', zorder=2)
    for (i, t) in enumerate(poisson_jumps):
        plt.axvline(x=t, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Inhomogeneous Poisson Jump' if i == 0 else '')
    plt.xlabel('Time$=$iteration $\\times$ stepsize $h$', fontsize=20)
    plt.ylabel('Objective', fontsize=20)
    plt.yscale('log')
    plt.grid(True)
    plt.tick_params(axis='both', which='both', labelsize=16, length=6, width=1.5)
    plt.legend(fontsize=23, frameon=True, fancybox=True, facecolor='white', edgecolor='black', framealpha=1.0)
    plt.title('RHGD Refreshment vs. Inhomogeneous Poisson Jump', fontsize=20)
    plt.tight_layout()
    pass
