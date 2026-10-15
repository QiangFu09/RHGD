"""Problem-bound GD, AGD, CAGD and RHGD implementations.

Factories accept the problem data and return callables with the signatures used
by the experiments. Strong, convex, and adaptive variants retain their distinct
update rules.
"""

from types import SimpleNamespace

import numpy as np


def strong_quadratic(A, dim):
    def quad_loss(x):
        return 0.5 * x @ A @ x

    def quad_grad(x):
        return A @ x

    def run_gd(x0, num_iters, gamma):
        x = x0.copy()
        losses = []
        for _ in range(num_iters):
            losses.append(quad_loss(x))
            x_new = x - gamma * quad_grad(x)
            x = x_new
        return (x, losses)

    def run_nag(x0, num_iters, mu, gamma):
        kappa = 1 / (mu * gamma)
        theta = (1 - np.sqrt(mu * gamma)) / (1 + np.sqrt(mu * gamma))
        x = x0.copy()
        y = x0.copy()
        losses = []
        for _ in range(num_iters):
            losses.append(quad_loss(x))
            x_old = x.copy()
            x = y - gamma * quad_grad(y)
            y = x + theta * (x - x_old)
        return (x, losses)

    def stochastic_accelerated_descent(x0, gamma, mu, num_iters, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        d = len(x0)
        x = x0.copy()
        z = x0.copy()
        x_list = [x.copy()]
        fx_list = []
        T = np.cumsum(rng.exponential(scale=1.0, size=num_iters + 1))
        sqrt_mu_L = np.sqrt(mu * gamma)
        for k in range(num_iters):
            fx_list.append(quad_loss(x))
            delta_t = T[k + 1] - T[k]
            tau_k = 0.5 * (1 - np.exp(-2 * sqrt_mu_L * delta_t))
            tau_k_prime = np.tanh(sqrt_mu_L * delta_t)
            gamma_k = gamma
            gamma_k_prime = 1 / np.sqrt(mu / gamma)
            y = x + tau_k * (z - x)
            x_next = y - gamma_k * quad_grad(y)
            z_next = z + tau_k_prime * (y - z) - gamma_k_prime * quad_grad(y)
            x = x_next
            z = z_next
        return (x, fx_list)

    def run_rhgd(x0, num_iters, h, mu, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        x = x0.copy()
        y = np.zeros(dim)
        losses = []
        for k in range(num_iters):
            losses.append(quad_loss(x))
            x_half = x + h * y
            x_new = x_half - h ** 2 * quad_grad(x_half)
            y_tilde = y - h * quad_grad(x_new)
            prob = np.sqrt(mu) * h
            if rng.random() < prob:
                y = np.zeros(dim)
            else:
                y = y_tilde
            x = x_new
        return (x, losses)

    return SimpleNamespace(loss=quad_loss, gradient=quad_grad, gd=run_gd, agd=run_nag, cagd=stochastic_accelerated_descent, rhgd=run_rhgd)

def convex_quadratic(A, dim):
    def quad_loss(x):
        return 0.5 * x @ A @ x

    def quad_grad(x):
        return A @ x

    def run_gd(x0, num_iters, gamma):
        x = x0.copy()
        losses = []
        for _ in range(num_iters):
            losses.append(quad_loss(x))
            x_new = x - gamma * quad_grad(x)
            x = x_new
        return (x, losses)

    def run_nag(x0, num_iters, gamma):
        x = x0.copy()
        y = x0.copy()
        t = 1.0
        losses = []
        for _ in range(num_iters):
            losses.append(quad_loss(x))
            x_new = y - gamma * quad_grad(y)
            t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
            y = x_new + (t - 1) / t_new * (x_new - x)
            (x, t) = (x_new, t_new)
        return (x, losses)

    def stochastic_accelerated_descent(x0, gamma, num_iters, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        d = len(x0)
        x = x0.copy()
        z = x0.copy()
        x_list = [x.copy()]
        fx_list = []
        Tk = np.cumsum(rng.exponential(scale=1.0, size=num_iters + 1))
        for k in range(num_iters):
            fx_list.append(quad_loss(x))
            tau_k = 1 - (Tk[k] / Tk[k + 1]) ** 2
            tau_k_prime = 0
            gamma_k = gamma
            gamma_k_prime = Tk[k] * gamma / 2
            y = x + tau_k * (z - x)
            x_next = y - gamma_k * quad_grad(y)
            z_next = z + tau_k_prime * (y - z) - gamma_k_prime * quad_grad(y)
            x = x_next
            z = z_next
        return (x, fx_list)

    def run_rhgd(x0, num_iters, h, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        x = x0.copy()
        y = np.zeros(dim)
        losses = []
        for k in range(num_iters):
            losses.append(quad_loss(x))
            x_half = x + h * y
            x_new = x_half - h ** 2 * quad_grad(x_half)
            y_tilde = y - h * quad_grad(x_new)
            prob = 17 / (2 * k + 18)
            if rng.random() < prob:
                y = np.zeros(dim)
            else:
                y = y_tilde
            x = x_new
        return (x, losses)

    return SimpleNamespace(loss=quad_loss, gradient=quad_grad, gd=run_gd, agd=run_nag, cagd=stochastic_accelerated_descent, rhgd=run_rhgd)

def regularized_logistic(A, b, lambda_reg, dim):
    num_samples = A.shape[0]

    def logistic_loss(x):
        exp_term = np.exp(-b * (A @ x))
        return np.mean(np.log(1 + exp_term)) + lambda_reg / 2 * np.linalg.norm(x) ** 2

    def logistic_grad(x):
        exp_term = np.exp(b * (A @ x))
        return -(A.T @ (b / (1 + exp_term))) / num_samples + lambda_reg * x

    def run_gd(x0, num_iters, gamma):
        x = x0.copy()
        losses = []
        for _ in range(num_iters):
            losses.append(logistic_loss(x))
            x_trial = x - gamma * logistic_grad(x)
            if logistic_loss(x_trial) <= logistic_loss(x) - gamma / 2 * np.linalg.norm(logistic_grad(x)) ** 2:
                gamma = 1.1 * gamma
                x_new = x_trial
            else:
                gamma = 0.6 * gamma
                x_new = x
            x = x_new
        return (x, losses)

    def run_nag(x0, num_iters, gamma, mu):
        x = x0.copy()
        y = x0.copy()
        losses = []
        for _ in range(num_iters):
            losses.append(logistic_loss(x))
            x_old = x.copy()
            x_trial = y - gamma * logistic_grad(y)
            if logistic_loss(x_trial) <= logistic_loss(y) - gamma / 2 * np.linalg.norm(logistic_grad(y)) ** 2:
                gamma = 1.1 * gamma
                x = x_trial
            else:
                gamma = 0.6 * gamma
                x = x_old
            kappa = 1 / (gamma * mu)
            theta = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)
            y_new = x + theta * (x - x_old)
            y = y_new
        return (x, losses)

    def stochastic_accelerated_descent(x0, gamma, mu, num_iters, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        d = len(x0)
        x = x0.copy()
        z = x0.copy()
        x_list = [x.copy()]
        fx_list = []
        T = np.cumsum(rng.exponential(scale=1.0, size=num_iters + 1))
        for k in range(num_iters):
            fx_list.append(logistic_loss(x))
            sqrt_mu_L = np.sqrt(mu * gamma)
            delta_t = T[k + 1] - T[k]
            tau_k = 0.5 * (1 - np.exp(-2 * sqrt_mu_L * delta_t))
            gamma_k = gamma
            y = x + tau_k * (z - x)
            x_trial = y - gamma_k * logistic_grad(y)
            if logistic_loss(x_trial) <= logistic_loss(y) - gamma_k / 2 * np.linalg.norm(logistic_grad(y)) ** 2:
                gamma = 1.1 * gamma
                x_next = x_trial
            else:
                gamma = 0.6 * gamma
                x_next = x
            sqrt_mu_L = np.sqrt(mu * gamma)
            tau_k_prime = np.tanh(sqrt_mu_L * delta_t)
            gamma_k_prime = np.sqrt(gamma) / np.sqrt(mu)
            z_next = z + tau_k_prime * (y - z) - gamma_k_prime * logistic_grad(y)
            x = x_next
            z = z_next
        return (x, fx_list)

    def run_rhgd(x0, num_iters, h, mu, c, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        x = x0.copy()
        y = np.zeros(dim)
        losses = []
        for k in range(num_iters):
            losses.append(logistic_loss(x))
            x_half = x + h * y
            x_trial = x_half - h ** 2 * logistic_grad(x_half)
            if logistic_loss(x_trial) <= logistic_loss(x_half) - h ** 2 / 2 * np.linalg.norm(logistic_grad(x_half)) ** 2:
                h = np.sqrt(1.1) * h
                x_new = x_trial
            else:
                h = np.sqrt(0.6) * h
                x_new = x
            y_tilde = y - h * logistic_grad(x_new)
            prob = np.sqrt(mu) * h * c
            if rng.random() < prob:
                y = np.zeros(dim)
            else:
                y = y_tilde
            x = x_new
        return (x, losses)

    return SimpleNamespace(loss=logistic_loss, gradient=logistic_grad, gd=run_gd, agd=run_nag, cagd=stochastic_accelerated_descent, rhgd=run_rhgd)

def convex_logistic(A, b, dim):
    num_samples = A.shape[0]

    def logistic_loss(x):
        exp_term = np.exp(-b * (A @ x))
        return np.mean(np.log(1 + exp_term))

    def logistic_grad(x):
        exp_term = np.exp(b * (A @ x))
        return -(A.T @ (b / (1 + exp_term))) / num_samples

    def run_gd(x0, num_iters, gamma):
        x = x0.copy()
        losses = []
        for _ in range(num_iters):
            losses.append(logistic_loss(x))
            x_trial = x - gamma * logistic_grad(x)
            if logistic_loss(x_trial) <= logistic_loss(x) - gamma / 2 * np.linalg.norm(logistic_grad(x)) ** 2:
                gamma = 1.1 * gamma
                x_new = x_trial
            else:
                gamma = 0.6 * gamma
                x_new = x
            x = x_new
        return (x, losses)

    def run_nag(x0, num_iters, gamma):
        x = x0.copy()
        y = x0.copy()
        t = 1.0
        losses = []
        for _ in range(num_iters):
            losses.append(logistic_loss(x))
            x_trial = y - gamma * logistic_grad(y)
            if logistic_loss(x_trial) <= logistic_loss(y) - gamma / 2 * np.linalg.norm(logistic_grad(y)) ** 2:
                gamma = 1.1 * gamma
                x_new = x_trial
            else:
                gamma = 0.6 * gamma
                x_new = x
            t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
            y_new = x_new + (t - 1) / t_new * (x_new - x)
            (x, y, t) = (x_new, y_new, t_new)
        return (x, losses)

    def stochastic_accelerated_descent(x0, gamma, num_iters, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        d = len(x0)
        x = x0.copy()
        z = x0.copy()
        x_list = [x.copy()]
        fx_list = []
        Tk = np.cumsum(rng.exponential(scale=1.0, size=num_iters + 1))
        for k in range(num_iters):
            fx_list.append(logistic_loss(x))
            tau_k = 1 - (Tk[k] / Tk[k + 1]) ** 2
            tau_k_prime = 0
            gamma_k = gamma
            y = x + tau_k * (z - x)
            x_trial = y - gamma_k * logistic_grad(y)
            if logistic_loss(x_trial) <= logistic_loss(y) - gamma_k / 2 * np.linalg.norm(logistic_grad(y)) ** 2:
                gamma = 1.1 * gamma
                x_next = x_trial
            else:
                gamma = 0.6 * gamma
                x_next = x
            gamma_k_prime = Tk[k] * gamma / 2
            z_next = z + tau_k_prime * (y - z) - gamma_k_prime * logistic_grad(y)
            x = x_next
            z = z_next
        return (x, fx_list)

    def run_rhgd(x0, num_iters, h, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        x = x0.copy()
        y = np.zeros(dim)
        losses = []
        for k in range(num_iters):
            losses.append(logistic_loss(x))
            x_half = x + h * y
            x_trial = x_half - h ** 2 * logistic_grad(x_half)
            if logistic_loss(x_trial) <= logistic_loss(x_half) - h ** 2 / 2 * np.linalg.norm(logistic_grad(x_half)) ** 2:
                h = np.sqrt(1.1) * h
                x_new = x_trial
            else:
                h = np.sqrt(0.6) * h
                x_new = x
            y_tilde = y - h * logistic_grad(x_new)
            prob = 17 / (2 * k + 18)
            if rng.random() < prob:
                y = np.zeros(dim)
            else:
                y = y_tilde
            x = x_new
        return (x, losses)

    return SimpleNamespace(loss=logistic_loss, gradient=logistic_grad, gd=run_gd, agd=run_nag, cagd=stochastic_accelerated_descent, rhgd=run_rhgd)

def constant_refresh_quadratic(A, dim):
    def quad_loss(x):
        return 0.5 * x @ A @ x

    def quad_grad(x):
        return A @ x

    def run_rhgd(x0, num_iters, h, mu, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        x = x0.copy()
        y = np.zeros(dim)
        losses = []
        refresh_points = []
        for k in range(num_iters):
            losses.append(quad_loss(x))
            x_half = x + h * y
            x_new = x_half - h ** 2 * quad_grad(x_half)
            y_tilde = y - h * quad_grad(x_new)
            prob = np.sqrt(mu) * h
            if rng.random() < prob:
                y = np.zeros(dim)
                refresh_points.append(k)
            else:
                y = y_tilde
            x = x_new
        return (x, losses, refresh_points)

    return SimpleNamespace(loss=quad_loss, gradient=quad_grad, rhgd=run_rhgd)

def varying_refresh_quadratic(A, dim):
    def quad_loss(x):
        return 0.5 * x @ A @ x

    def quad_grad(x):
        return A @ x

    def run_rhgd(x0, num_iters, h, L, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        x = x0.copy()
        y = np.zeros(dim)
        losses = []
        refresh_points = []
        for k in range(num_iters):
            losses.append(quad_loss(x))
            x_half = x + h * y
            x_new = x_half - h ** 2 * quad_grad(x_half)
            y_tilde = y - h * quad_grad(x_new)
            prob = 17 / (2 * k + 18)
            if rng.random() < prob:
                y = np.zeros(dim)
                refresh_points.append(k)
            else:
                y = y_tilde
            x = x_new
        return (x, losses, refresh_points)

    return SimpleNamespace(loss=quad_loss, gradient=quad_grad, rhgd=run_rhgd)
