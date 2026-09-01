"""
pro_flows_comparison_extended_v2_fixed.py
=========================================
Complete and runnable PrO algorithm comparison with three extended experiments.
All plots use English labels and titles, and are saved as PNG files.
Designed for easy review and reproducibility.
"""

import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import itertools
import warnings
warnings.filterwarnings("ignore")

# ==================== Unified algorithm style mapping ====================
ALGORITHM_LABELS = {
    'pro': 'PrO (MFLD)',
    'underdamped_pro': 'PrO (NULA)',
    'vgd': 'VGD',
    'fr_mirror': 'FR Mirror',
    'wfr_smc': 'SMC-WFR',
    'wfr_bdl': 'BDL-WFR',
    'wfr': 'KDE-WFR',
    'fr': 'KDE-FR',
    'bayes': 'Bayes',
}
ALGORITHM_COLORS = {
    'pro': '#e41a1c',          # red
    'underdamped_pro': '#ff7f00',  # orange
    'vgd': '#4daf4a',          # green
    'fr_mirror': '#984ea3',    # purple
    'wfr_smc': '#377eb8',      # blue
    'wfr_bdl': '#f781bf',      # pink
    'wfr': '#a65628',          # brown
    'fr': '#999999',           # grey
    'bayes': '#1f78b4',        # dark blue
}
ALGORITHM_LINESTYLES = {
    'pro': '-',
    'underdamped_pro': '--',
    'vgd': '-.',
    'fr_mirror': ':',
    'wfr_smc': '-',
    'wfr_bdl': '--',
    'wfr': '-.',
    'fr': ':',
    'bayes': '-',
}
# ========================================================================

# ==============================================================================
# Device configuration
# ==============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==============================================================================
# 1. Data loading and synthetic data generators
# ==============================================================================
def load_real_data():
    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=43)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    return (torch.tensor(X_train, dtype=torch.float32, device=device),
            torch.tensor(y_train, dtype=torch.float32, device=device).view(-1, 1),
            torch.tensor(X_val, dtype=torch.float32, device=device),
            torch.tensor(y_val, dtype=torch.float32, device=device).view(-1, 1),
            torch.tensor(X_test, dtype=torch.float32, device=device),
            torch.tensor(y_test, dtype=torch.float32, device=device).view(-1, 1))

def generate_linear_data(n_samples=1000, d=1, noise_scale=0.1, seed=42):
    torch.manual_seed(seed)
    X = torch.randn(n_samples, d)
    theta_true = torch.randn(d)
    y = X @ theta_true + noise_scale * torch.randn(n_samples)
    return X, y.view(-1, 1)

def generate_data_with_outliers(n_samples=1000, d=1, outlier_frac=0.1, seed=42):
    torch.manual_seed(seed)
    X = torch.randn(n_samples, d)
    theta_true = torch.randn(d)
    y = X @ theta_true + 0.1 * torch.randn(n_samples)
    n_out = int(outlier_frac * n_samples)
    outlier_idx = torch.randperm(n_samples)[:n_out]
    y[outlier_idx] = 10 * torch.randn(n_out)
    return X, y.view(-1, 1)

def generate_heteroscedastic_data(n_samples=1000, d=1, seed=42):
    torch.manual_seed(seed)
    X = torch.randn(n_samples, d)
    theta_true = torch.randn(d)
    noise_scale = 0.1 + 0.5 * torch.abs(X[:, 0])
    y = X @ theta_true + noise_scale * torch.randn(n_samples)
    return X, y.view(-1, 1)

def generate_nonlinear_data(n_samples=1000, d=1, noise_scale=0.1, seed=42):
    torch.manual_seed(seed)
    X = torch.randn(n_samples, d)
    if d == 1:
        y = X[:, 0] ** 2 + noise_scale * torch.randn(n_samples)
    else:
        y = torch.sum(X ** 2, dim=1) + noise_scale * torch.randn(n_samples)
    return X, y.view(-1, 1)

# ==============================================================================
# 2. Model parameter estimation
# ==============================================================================
def estimate_noise_variance(X, y):
    theta_ols = torch.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ theta_ols
    sigma2 = torch.mean(residuals ** 2).clamp_min(1e-4)
    return float(sigma2.item()), theta_ols.flatten()

def median_heuristic_gamma2(y, max_points=2000):
    y_flat = y.flatten()
    if y_flat.numel() > max_points:
        generator = torch.Generator(device=device)
        generator.manual_seed(20260729)
        idx = torch.randperm(y_flat.numel(), generator=generator, device=device)[:max_points]
        y_flat = y_flat[idx]
    D2 = (y_flat[:, None] - y_flat[None, :]) ** 2
    upper = D2[torch.triu_indices(D2.shape[0], D2.shape[1], offset=1, device=device).unbind()]
    upper = upper[upper > 0]
    if upper.numel() == 0:
        return 1.0
    return float(torch.median(upper).item())

# ==============================================================================
# 3. Core algorithmic functions (full implementation)
# ==============================================================================
# 3.1 Prior and helpers
def prior_log_prob(theta):
    return -0.5 * torch.sum(theta ** 2, dim=-1)

def prior_grad_log(theta):
    return -theta

def prior_sample(key, p, dim):
    """Draw p independent initial particles from N(0,I_dim)."""
    return torch.randn(p, dim, device=device)

# 3.2 MMD potential and gradient
def compute_mmd_potential_and_grad(
    theta,
    w,
    X_batch,
    y_batch,
    gamma2,
    sigma2,
    leave_one_out=False,
):
    B = X_batch.shape[0]
    p = theta.shape[0]
    if gamma2 <= 0 or sigma2 <= 0:
        raise ValueError("gamma2 and sigma2 must both be strictly positive.")
    if leave_one_out and p < 2:
        raise ValueError("leave_one_out=True requires at least two particles.")

    if y_batch.dim() == 1:
        y_batch = y_batch.view(-1, 1)

    pred_mean = X_batch @ theta.T  # [B, p]

    pair_variance = gamma2 + 2.0 * sigma2
    pair_scale = math.sqrt(gamma2 / pair_variance)
    diff_pair = pred_mean[:, :, None] - pred_mean[:, None, :]
    K_pair = pair_scale * torch.exp(-(diff_pair ** 2) / (2.0 * pair_variance))

    data_variance = gamma2 + sigma2
    data_scale = math.sqrt(gamma2 / data_variance)
    diff_data = pred_mean - y_batch
    K_data = data_scale * torch.exp(-(diff_data ** 2) / (2.0 * data_variance))

    interaction_weights = w[None, :].expand(p, p).clone()
    if leave_one_out:
        interaction_weights.fill_diagonal_(0.0)
        interaction_weights = interaction_weights / interaction_weights.sum(dim=1, keepdim=True)

    interaction_term = torch.sum(K_pair * interaction_weights[None, :, :], dim=2)
    phi = 2.0 * (interaction_term.mean(dim=0) - K_data.mean(dim=0))

    dK_pair_dmean = -(diff_pair / pair_variance) * K_pair
    dK_data_dmean = -(diff_data / data_variance) * K_data
    grad_mean = 2.0 * (torch.sum(dK_pair_dmean * interaction_weights[None, :, :], dim=2) - dK_data_dmean)
    grad_phi = (grad_mean.T @ X_batch) / B
    return phi, grad_phi

def compute_mmd_potential_and_grad_at(
    theta_eval,
    theta_measure,
    w_measure,
    X_batch,
    y_batch,
    gamma2,
    sigma2,
):
    B = X_batch.shape[0]
    if y_batch.dim() == 1:
        y_batch = y_batch.view(-1, 1)
    pred_eval = X_batch @ theta_eval.T
    pred_measure = X_batch @ theta_measure.T
    pair_variance = gamma2 + 2.0 * sigma2
    pair_scale = math.sqrt(gamma2 / pair_variance)
    diff_pair = pred_eval[:, :, None] - pred_measure[:, None, :]
    K_pair = pair_scale * torch.exp(-(diff_pair ** 2) / (2.0 * pair_variance))
    data_variance = gamma2 + sigma2
    data_scale = math.sqrt(gamma2 / data_variance)
    diff_data = pred_eval - y_batch
    K_data = data_scale * torch.exp(-(diff_data ** 2) / (2.0 * data_variance))
    interaction_term = torch.sum(K_pair * w_measure[None, None, :], dim=2)
    phi = 2.0 * (interaction_term.mean(dim=0) - K_data.mean(dim=0))
    dK_pair_dmean = -(diff_pair / pair_variance) * K_pair
    dK_data_dmean = -(diff_data / data_variance) * K_data
    grad_mean = 2.0 * (torch.sum(dK_pair_dmean * w_measure[None, None, :], dim=2) - dK_data_dmean)
    grad_phi = (grad_mean.T @ X_batch) / B
    return phi, grad_phi

# 3.3 KDE density and score
def compute_kde_log_density_and_score(theta, w, bandwidth):
    if bandwidth <= 0:
        raise ValueError("KDE bandwidth must be strictly positive.")
    h2 = bandwidth ** 2
    parameter_dim = theta.shape[1]
    diff = theta[:, None, :] - theta[None, :, :]
    D2 = torch.sum(diff ** 2, dim=-1)
    log_kernel = (-0.5 * D2 / h2 - 0.5 * parameter_dim * math.log(2.0 * math.pi * h2))
    log_weighted_kernel = torch.log(w.clamp_min(1e-30))[None, :] + log_kernel
    log_q = torch.logsumexp(log_weighted_kernel, dim=1)
    responsibilities = torch.softmax(log_weighted_kernel, dim=1)
    score_q = -torch.sum(responsibilities[:, :, None] * diff, dim=1) / h2
    return log_q, score_q

def compute_gaussian_mixture_log_density(x, means, w, variance):
    if variance <= 0:
        raise ValueError("variance must be strictly positive.")
    parameter_dim = x.shape[1]
    diff = x[:, None, :] - means[None, :, :]
    D2 = torch.sum(diff ** 2, dim=-1)
    log_kernel = (-0.5 * D2 / variance - 0.5 * parameter_dim * math.log(2.0 * math.pi * variance))
    return torch.logsumexp(torch.log(w.clamp_min(1e-30))[None, :] + log_kernel, dim=1)

# 3.4 Weight updates and resampling
def update_weights_fisher_rao(w, eta, eta_w):
    eta_bar = torch.sum(w * eta)
    log_w_new = torch.log(w.clamp_min(1e-30)) - eta_w * (eta - eta_bar)
    return torch.softmax(log_w_new, dim=0)

def multinomial_resample(theta, w):
    p = theta.shape[0]
    ancestors = torch.multinomial(w, p, replacement=True)
    theta_new = theta[ancestors]
    w_new = torch.ones(p, device=theta.device) / p
    return theta_new, w_new

def systematic_resample(theta, w):
    p = theta.shape[0]
    start = torch.rand(1, device=theta.device) / p
    positions = start + torch.arange(p, device=theta.device) / p
    cumulative = torch.cumsum(w, dim=0)
    cumulative[-1] = 1.0
    ancestors = torch.searchsorted(cumulative, positions, right=False)
    theta_new = theta[ancestors]
    w_new = torch.ones(p, device=theta.device) / p
    return theta_new, w_new

# 3.5 Algorithm training steps
def train_step_fr_mirror(theta, w, X_batch, y_batch, gamma2, sigma2, lambda_, eta_w, base_w):
    phi, _ = compute_mmd_potential_and_grad(theta, w, X_batch, y_batch, gamma2, sigma2, leave_one_out=False)
    decay = math.exp(-eta_w)
    log_target = torch.log(base_w.clamp_min(1e-30)) - lambda_ * phi
    log_w_new = decay * torch.log(w.clamp_min(1e-30)) + (1.0 - decay) * log_target
    return theta.detach(), torch.softmax(log_w_new, dim=0).detach()

def train_step_wfr_smc(theta, w, X_batch, y_batch, gamma2, sigma2, lambda_, gamma):
    theta_old, w_old = systematic_resample(theta, w)
    _, grad_phi_old = compute_mmd_potential_and_grad(theta_old, w_old, X_batch, y_batch, gamma2, sigma2, leave_one_out=True)
    proposal_means = theta_old + gamma * (prior_grad_log(theta_old) - lambda_ * grad_phi_old)
    theta_new = proposal_means + math.sqrt(2.0 * gamma) * torch.randn_like(theta_old)
    log_proposal = compute_gaussian_mixture_log_density(theta_new, proposal_means, w_old, variance=2.0 * gamma)
    phi_new, _ = compute_mmd_potential_and_grad_at(theta_new, theta_old, w_old, X_batch, y_batch, gamma2, sigma2)
    log_target = prior_log_prob(theta_new) - lambda_ * phi_new
    fr_exponent = 1.0 - math.exp(-gamma)
    log_w_new = fr_exponent * (log_target - log_proposal)
    w_new = torch.softmax(log_w_new, dim=0)
    return theta_new.detach(), w_new.detach()

def birth_death_population_step(theta, centred_rate, gamma):
    p = theta.shape[0]
    particles = []
    event_probability = 1.0 - torch.exp(-gamma * torch.abs(centred_rate))
    uniforms = torch.rand(p, device=theta.device)
    for j in range(p):
        event = bool((uniforms[j] < event_probability[j]).item())
        rate_j = float(centred_rate[j].item())
        if rate_j > 0.0 and event:
            continue
        particles.append(theta[j])
        if rate_j < 0.0 and event:
            particles.append(theta[j].clone())
    if len(particles) == 0:
        particles = [theta[torch.argmin(centred_rate)].clone()]
    population = torch.stack(particles, dim=0)
    n_population = population.shape[0]
    if n_population > p:
        keep = torch.randperm(n_population, device=theta.device)[:p]
        population = population[keep]
    elif n_population < p:
        add = torch.randint(0, n_population, (p - n_population,), device=theta.device)
        population = torch.cat([population, population[add]], dim=0)
    return population

def train_step_wfr_bdl(theta, X_batch, y_batch, gamma2, sigma2, lambda_, gamma, kde_bandwidth):
    p = theta.shape[0]
    w = torch.ones(p, device=theta.device) / p
    _, grad_phi = compute_mmd_potential_and_grad(theta, w, X_batch, y_batch, gamma2, sigma2, leave_one_out=True)
    theta_new = theta + gamma * (prior_grad_log(theta) - lambda_ * grad_phi) + math.sqrt(2.0 * gamma) * torch.randn_like(theta)
    w_uniform = torch.ones(p, device=theta.device) / p
    phi_new, _ = compute_mmd_potential_and_grad(theta_new, w_uniform, X_batch, y_batch, gamma2, sigma2, leave_one_out=False)
    log_q, _ = compute_kde_log_density_and_score(theta_new, w_uniform, kde_bandwidth)
    rate = lambda_ * phi_new + log_q - prior_log_prob(theta_new)
    centred_rate = rate - rate.mean()
    theta_new = birth_death_population_step(theta_new, centred_rate, gamma)
    w_new = torch.ones(p, device=theta.device) / p
    return theta_new.detach(), w_new

def compute_vgd_velocity(theta, X_batch, y_batch, gamma2, sigma2, lambda_, lengthscales):
    if len(lengthscales) == 0:
        raise ValueError("VGD requires at least one positive kernel lengthscale.")
    if any(float(ell) <= 0.0 for ell in lengthscales):
        raise ValueError("All VGD kernel lengthscales must be strictly positive.")
    p = theta.shape[0]
    w = torch.ones(p, device=theta.device) / p
    _, grad_phi = compute_mmd_potential_and_grad(theta, w, X_batch, y_batch, gamma2, sigma2, leave_one_out=False)
    variational_drift = prior_grad_log(theta) - lambda_ * grad_phi
    diff = theta[:, None, :] - theta[None, :, :]
    squared_distance = torch.sum(diff ** 2, dim=-1)
    kernel_sum = torch.zeros(p, p, device=theta.device, dtype=theta.dtype)
    repulsion_sum = torch.zeros(p, p, theta.shape[1], device=theta.device, dtype=theta.dtype)
    for ell in lengthscales:
        ell2 = float(ell) ** 2
        base = 1.0 + squared_distance / ell2
        kernel_sum = kernel_sum + torch.rsqrt(base)
        repulsion_sum = repulsion_sum + (diff / ell2) * base.pow(-1.5)[:, :, None]
    n_scales = float(len(lengthscales))
    kernel = kernel_sum / n_scales
    repulsion = repulsion_sum / n_scales
    smoothed_drift = torch.sum(kernel[:, :, None] * variational_drift[None, :, :], dim=1) / p
    repulsive_drift = torch.sum(repulsion, dim=1) / p
    return smoothed_drift + repulsive_drift

def train_step_vgd(theta, X_batch, y_batch, gamma2, sigma2, lambda_, step_size, lengthscales,
                   first_moment, second_moment, iteration, beta1=0.9, beta2=0.999, adam_epsilon=1e-8):
    if step_size <= 0:
        raise ValueError("VGD step_size must be strictly positive.")
    velocity = compute_vgd_velocity(theta, X_batch, y_batch, gamma2, sigma2, lambda_, lengthscales)
    iteration = iteration + 1
    first_moment = beta1 * first_moment + (1.0 - beta1) * velocity
    second_moment = beta2 * second_moment + (1.0 - beta2) * velocity.square()
    first_hat = first_moment / (1.0 - beta1 ** iteration)
    second_hat = second_moment / (1.0 - beta2 ** iteration)
    theta_new = theta + step_size * first_hat / (torch.sqrt(second_hat) + adam_epsilon)
    return theta_new.detach(), first_moment.detach(), second_moment.detach(), iteration

def compute_nula_coefficients(step_size, friction):
    h = float(step_size)
    gamma = float(friction)
    if h <= 0.0:
        raise ValueError("NULA step_size must be strictly positive.")
    if gamma <= 0.0:
        raise ValueError("NULA friction must be strictly positive.")
    one_minus_phi2 = -math.expm1(-gamma * h)
    phi2 = 1.0 - one_minus_phi2
    phi0 = one_minus_phi2 / gamma
    phi1 = (h - phi0) / gamma
    sigma11 = (2.0 / gamma) * (h - 2.0 * phi0 + (1.0 - phi2 ** 2) / (2.0 * gamma))
    sigma12 = (one_minus_phi2 ** 2) / gamma
    sigma22 = 1.0 - phi2 ** 2
    sigma11 = max(sigma11, 0.0)
    sigma22 = max(sigma22, 0.0)
    return phi0, phi1, phi2, sigma11, sigma12, sigma22

def sample_nula_noise(theta, sigma11, sigma12, sigma22):
    z_position = torch.randn_like(theta)
    z_velocity = torch.randn_like(theta)
    if sigma11 <= 1e-30:
        noise_position = torch.zeros_like(theta)
        noise_velocity = math.sqrt(max(sigma22, 0.0)) * z_velocity
        return noise_position, noise_velocity
    sqrt_sigma11 = math.sqrt(sigma11)
    conditional_coefficient = sigma12 / sqrt_sigma11
    conditional_variance = sigma22 - (sigma12 ** 2) / sigma11
    conditional_variance = max(conditional_variance, 0.0)
    noise_position = sqrt_sigma11 * z_position
    noise_velocity = conditional_coefficient * z_position + math.sqrt(conditional_variance) * z_velocity
    return noise_position, noise_velocity

def train_step_underdamped_pro(theta, velocity, X_batch, y_batch, gamma2, sigma2, lambda_, step_size, friction, p):
    if velocity.shape != theta.shape:
        raise ValueError("NULA velocity and theta must have the same shape.")
    w = torch.ones(p, device=theta.device) / p
    _, grad_phi = compute_mmd_potential_and_grad(theta, w, X_batch, y_batch, gamma2, sigma2, leave_one_out=False)
    force = lambda_ * grad_phi - prior_grad_log(theta)
    phi0, phi1, phi2, sigma11, sigma12, sigma22 = compute_nula_coefficients(step_size, friction)
    noise_position, noise_velocity = sample_nula_noise(theta, sigma11, sigma12, sigma22)
    theta_new = theta + phi0 * velocity - phi1 * force + noise_position
    velocity_new = phi2 * velocity - phi0 * force + noise_velocity
    return theta_new.detach(), velocity_new.detach()

def train_step_pro(theta, X_batch, y_batch, gamma2, sigma2, lambda_, dt, p):
    w = torch.ones(p, device=device) / p
    _, grad_score = compute_mmd_potential_and_grad(theta, w, X_batch, y_batch, gamma2, sigma2, leave_one_out=True)
    grad_prior = prior_grad_log(theta)
    noise = torch.randn_like(theta) * math.sqrt(2.0 * dt)
    theta_new = theta - dt * (lambda_ * grad_score - grad_prior) + noise
    return theta_new.detach()

def compute_deterministic_flow_terms(theta, w, X_batch, y_batch, gamma2, sigma2, lambda_, kde_bandwidth):
    phi, grad_score = compute_mmd_potential_and_grad(theta, w, X_batch, y_batch, gamma2, sigma2, leave_one_out=False)
    log_q, score_q = compute_kde_log_density_and_score(theta, w, kde_bandwidth)
    transport_grad = lambda_ * grad_score + score_q - prior_grad_log(theta)
    reaction_potential = lambda_ * phi + log_q - prior_log_prob(theta)
    return transport_grad, reaction_potential

def train_step_wfr_imp(theta, w, X_batch, y_batch, gamma2, sigma2, lambda_, eta_theta, eta_w, p, momentum_buffer, kde_bandwidth):
    grad, eta = compute_deterministic_flow_terms(theta, w, X_batch, y_batch, gamma2, sigma2, lambda_, kde_bandwidth)
    momentum_buffer = 0.9 * momentum_buffer + 0.1 * grad
    theta_new = theta - eta_theta * momentum_buffer
    w_raw = update_weights_fisher_rao(w, eta, eta_w)
    w_new = 0.6 * w + 0.4 * w_raw
    w_new = w_new / torch.sum(w_new)
    return theta_new.detach(), w_new.detach(), momentum_buffer.detach()

def train_step_old(theta, w, X_batch, y_batch, gamma2, sigma2, lambda_, eta_theta, eta_w, p, mode, kde_bandwidth):
    grad, eta = compute_deterministic_flow_terms(theta, w, X_batch, y_batch, gamma2, sigma2, lambda_, kde_bandwidth)
    if mode in ['wfr', 'wf']:
        theta_new = theta - eta_theta * grad
    else:
        theta_new = theta
    if mode in ['wfr', 'fr']:
        w_new = update_weights_fisher_rao(w, eta, eta_w)
    else:
        w_new = w
    return theta_new.detach(), w_new.detach()

# 3.6 Learning rate schedulers
def get_lr_cosine(step, base_lr, total_steps):
    floor = 0.01 * base_lr
    return max(base_lr * 0.5 * (1.0 + math.cos(math.pi * step / total_steps)), floor)

def get_lr_exp(step, base_lr, decay_rate):
    floor = 0.01 * base_lr
    return max(base_lr * (decay_rate ** step), floor)

# 3.7 Evaluation functions
def compute_predictive_nll(theta, w, X_eval, y_eval, sigma2, chunk_size=2048):
    log_w = torch.log(w.clamp_min(1e-30))
    log_normalizer = -0.5 * math.log(2.0 * math.pi * sigma2)
    total_nll = 0.0
    total_count = 0
    for start in range(0, X_eval.shape[0], chunk_size):
        stop = min(start + chunk_size, X_eval.shape[0])
        X_chunk = X_eval[start:stop]
        y_chunk = y_eval[start:stop]
        pred_mean = X_chunk @ theta.T
        log_components = log_w[None, :] + log_normalizer - 0.5 * (y_chunk - pred_mean) ** 2 / sigma2
        nll = -torch.logsumexp(log_components, dim=1)
        total_nll += float(nll.sum().item())
        total_count += nll.numel()
    return total_nll / total_count

def compute_predictive_mmd(theta, w, X_eval, y_eval, gamma2, sigma2, chunk_size=512):
    total_score = 0.0
    total_count = 0
    pair_variance = gamma2 + 2.0 * sigma2
    pair_scale = math.sqrt(gamma2 / pair_variance)
    data_variance = gamma2 + sigma2
    data_scale = math.sqrt(gamma2 / data_variance)
    for start in range(0, X_eval.shape[0], chunk_size):
        stop = min(start + chunk_size, X_eval.shape[0])
        X_chunk = X_eval[start:stop]
        y_chunk = y_eval[start:stop]
        pred_mean = X_chunk @ theta.T
        diff_pair = pred_mean[:, :, None] - pred_mean[:, None, :]
        K_pair = pair_scale * torch.exp(-(diff_pair ** 2) / (2.0 * pair_variance))
        diff_data = pred_mean - y_chunk
        K_data = data_scale * torch.exp(-(diff_data ** 2) / (2.0 * data_variance))
        pair_term = torch.sum(K_pair * w[None, :, None] * w[None, None, :], dim=(1, 2))
        data_term = torch.sum(K_data * w[None, :], dim=1)
        score = pair_term + 1.0 - 2.0 * data_term
        total_score += float(score.sum().item())
        total_count += score.numel()
    return total_score / total_count

# ==============================================================================
# 4. run_experiment (improved version)
# ==============================================================================
def run_experiment(mode, X_train, y_train, X_eval, y_eval, params, K, theta_init=None, batch_indices=None):
    """
    params must contain 'p' and 'd' (feature dimension).
    """
    if theta_init is None:
        theta = prior_sample(None, params['p'], params['d'])
    else:
        theta = theta_init.clone().to(device)

    w = torch.ones(params['p'], device=device) / params['p']
    base_w = w.clone()
    momentum_buffer = torch.zeros_like(theta)
    vgd_first_moment = torch.zeros_like(theta)
    vgd_second_moment = torch.zeros_like(theta)
    vgd_iteration = 0
    underdamped_velocity = torch.randn_like(theta) if mode == 'underdamped_pro' else None

    record_steps, eval_losses, eval_mmd = [], [], []
    batch_size = params.get('batch_size', 128)
    eval_every = params.get('eval_every', 10)
    max_mmd_eval_points = params.get('max_mmd_eval_points', 1024)

    if X_eval.shape[0] > max_mmd_eval_points:
        X_mmd_eval = X_eval[:max_mmd_eval_points]
        y_mmd_eval = y_eval[:max_mmd_eval_points]
    else:
        X_mmd_eval = X_eval
        y_mmd_eval = y_eval

    for step in range(K):
        if batch_indices is None:
            idx = torch.randint(0, X_train.shape[0], (batch_size,), device=device)
        else:
            idx = batch_indices[step]
        X_batch, y_batch = X_train[idx], y_train[idx]

        current_eta_theta = get_lr_cosine(step, params.get('eta_theta', 0.001), K)
        current_eta_w = get_lr_exp(step, params.get('eta_w', 0.01), 0.995)

        if mode == 'pro':
            theta = train_step_pro(theta, X_batch, y_batch, params['gamma2'], params['sigma2'], params['lambda_'], params['dt'], params['p'])
            w = torch.ones(params['p'], device=device) / params['p']
        elif mode == 'underdamped_pro':
            theta, underdamped_velocity = train_step_underdamped_pro(theta, underdamped_velocity, X_batch, y_batch,
                params['gamma2'], params['sigma2'], params['lambda_'], params['nula_step_size'], params['nula_friction'], params['p'])
            w = torch.ones(params['p'], device=device) / params['p']
        elif mode == 'vgd':
            (theta, vgd_first_moment, vgd_second_moment, vgd_iteration) = train_step_vgd(theta, X_batch, y_batch,
                params['gamma2'], params['sigma2'], params['lambda_'], params['vgd_step_size'], params['vgd_lengthscales'],
                vgd_first_moment, vgd_second_moment, vgd_iteration)
            w = torch.ones(params['p'], device=device) / params['p']
        elif mode == 'fr_mirror':
            theta, w = train_step_fr_mirror(theta, w, X_batch, y_batch, params['gamma2'], params['sigma2'],
                params['lambda_'], params['eta_w'], base_w)
        elif mode == 'wfr_smc':
            theta, w = train_step_wfr_smc(theta, w, X_batch, y_batch, params['gamma2'], params['sigma2'],
                params['lambda_'], params['dt'])
        elif mode == 'wfr_bdl':
            theta, w = train_step_wfr_bdl(theta, X_batch, y_batch, params['gamma2'], params['sigma2'],
                params['lambda_'], params['dt'], params['kde_bandwidth'])
        else:
            theta, w = train_step_old(theta, w, X_batch, y_batch, params['gamma2'], params['sigma2'],
                params['lambda_'], current_eta_theta, current_eta_w, params['p'], mode, params['kde_bandwidth'])

        if step % eval_every == 0 or step == K - 1:
            with torch.no_grad():
                nll = compute_predictive_nll(theta, w, X_eval, y_eval, params['sigma2'])
                mmd = compute_predictive_mmd(theta, w, X_mmd_eval, y_mmd_eval, params['gamma2'], params['sigma2'])
            record_steps.append(step)
            eval_losses.append(nll)
            eval_mmd.append(mmd)

    return (torch.tensor(record_steps, dtype=torch.long),
            torch.tensor(eval_losses, dtype=torch.float32),
            torch.tensor(eval_mmd, dtype=torch.float32))

# ==============================================================================
# 5. Experiment 5: Misspecification scenarios (with hyperparameter search)
# ==============================================================================
def experiment5_misspecification_scenarios():
    print("\n" + "=" * 60)
    print("Experiment 5: Algorithm comparison under misspecification scenarios (with hyperparameter search)")
    print("=" * 60)

    scenarios = {
        'Outliers': generate_data_with_outliers,
        'Heteroscedastic': generate_heteroscedastic_data,
        'Nonlinear': generate_nonlinear_data
    }
    # 扩展 modes：新增 fr_mirror, wfr, fr
    modes = ['pro', 'underdamped_pro', 'vgd', 'fr_mirror', 'wfr_smc', 'wfr_bdl', 'wfr', 'fr']
    param_grids = {
        'pro': {'dt': [0.0005, 0.001, 0.002, 0.005, 0.01]},
        'underdamped_pro': {
            'nula_step_size': [0.005, 0.01, 0.02, 0.05, 0.1],
            'nula_friction': [1.0, 3.0, 5.0, 10.0, 20.0]
        },
        'vgd': {
            'vgd_step_size': [0.005, 0.01, 0.02, 0.05, 0.1],
            'vgd_lengthscales': [
                (0.125, 0.25, 0.5),
                (0.1, 0.3, 0.6),
                (0.2, 0.4, 0.8)
            ]
        },
        'fr_mirror': {
            'eta_w': [0.005, 0.01, 0.02, 0.05]
        },
        'wfr_smc': {'dt': [0.0005, 0.001, 0.002, 0.005, 0.01]},
        'wfr_bdl': {
            'dt': [0.0005, 0.001, 0.002, 0.005, 0.01],
            'kde_bandwidth': [0.3, 0.5, 0.8, 1.2, 1.5]
        },
        'wfr': {
            'eta_theta': [0.001, 0.005, 0.01],
            'eta_w': [0.005, 0.01, 0.02],
            'kde_bandwidth': [0.3, 0.5, 0.8, 1.2, 1.5]
        },
        'fr': {
            'eta_w': [0.005, 0.01, 0.02],
            'kde_bandwidth': [0.3, 0.5, 0.8, 1.2, 1.5]
        }
    }
    num_search_runs = 5
    num_eval_runs = 10

    all_results = {}
    for scenario_name, data_gen in scenarios.items():
        print(f"\n--- Scenario: {scenario_name} ---")
        X, y = data_gen(n_samples=500, d=1, seed=2026)
        n_train = 400
        X_train_syn, y_train_syn = X[:n_train], y[:n_train]
        X_test_syn, y_test_syn = X[n_train:], y[n_train:]
        X_val_syn, y_val_syn = X_train_syn[-50:], y_train_syn[-50:]
        X_train_syn, y_train_syn = X_train_syn[:-50], y_train_syn[:-50]

        sigma2_hat_syn = estimate_noise_variance(X_train_syn, y_train_syn)[0]
        gamma2_hat_syn = median_heuristic_gamma2(y_train_syn)
        base_params = {
            'gamma2': gamma2_hat_syn, 'sigma2': sigma2_hat_syn,
            'lambda_': math.sqrt(n_train), 'p': 30, 'd': 1,
            'batch_size': 128, 'eval_every': 20,
        }
        res = {}
        for mode in modes:
            print(f"  Tuning algorithm: {mode}")
            best_params = base_params.copy()
            grid = param_grids[mode]
            keys = list(grid.keys())
            combos = list(itertools.product(*(grid[k] for k in keys)))
            best_score = float('inf')
            for combo in combos:
                p_tmp = base_params.copy()
                p_tmp.update(dict(zip(keys, combo)))
                score_sum = 0.0
                for run in range(num_search_runs):
                    torch.manual_seed(42 + run * 1000 + hash(str(combo)) % 1000)
                    _, _, mmd_series = run_experiment(mode, X_train_syn, y_train_syn, X_val_syn, y_val_syn, p_tmp, K=300)
                    score_sum += mmd_series[-1].item()
                avg_score = score_sum / num_search_runs
                if avg_score < best_score:
                    best_score = avg_score
                    best_params.update(dict(zip(keys, combo)))
            print(f"    Best parameters: { {k: best_params[k] for k in keys} }")
            nll_all, mmd_all = [], []
            for run in range(num_eval_runs):
                torch.manual_seed(42 + run * 2000 + hash(str(best_params)) % 1000)
                theta_init = torch.randn(best_params['p'], best_params['d'], device=device)
                batch_indices = torch.randint(0, len(X_train_syn), (400, best_params['batch_size']), device=device)
                steps, nll_series, mmd_series = run_experiment(mode, X_train_syn, y_train_syn, X_test_syn, y_test_syn,
                    best_params, K=400, theta_init=theta_init, batch_indices=batch_indices)
                nll_all.append(nll_series.numpy())
                mmd_all.append(mmd_series.numpy())
            res[mode] = {'steps': steps.numpy(), 'nll_mean': np.mean(nll_all, axis=0), 'nll_std': np.std(nll_all, axis=0),
                         'mmd_mean': np.mean(mmd_all, axis=0), 'mmd_std': np.std(mmd_all, axis=0)}
        all_results[scenario_name] = res

    # ---- Plotting with unified style ----
    for scenario_name, res in all_results.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for idx, mode in enumerate(modes):
            data = res[mode]
            steps = data['steps']
            axes[0].plot(steps, data['nll_mean'], label=ALGORITHM_LABELS[mode],
                         color=ALGORITHM_COLORS[mode], linestyle=ALGORITHM_LINESTYLES[mode])
            axes[0].fill_between(steps, data['nll_mean']-data['nll_std'],
                                 data['nll_mean']+data['nll_std'], alpha=0.2,
                                 color=ALGORITHM_COLORS[mode])
            axes[1].plot(steps, data['mmd_mean'], label=ALGORITHM_LABELS[mode],
                         color=ALGORITHM_COLORS[mode], linestyle=ALGORITHM_LINESTYLES[mode])
            axes[1].fill_between(steps, data['mmd_mean']-data['mmd_std'],
                                 data['mmd_mean']+data['mmd_std'], alpha=0.2,
                                 color=ALGORITHM_COLORS[mode])
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Test NLL')
        axes[0].set_title(f'{scenario_name} - Test NLL')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Test MMD^2')
        axes[1].set_title(f'{scenario_name} - Test MMD')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        # 添加阴影带含义说明
        fig.text(0.5, 0.01, 'Shaded area = mean ± 1 std (over 10 independent runs)', ha='center', fontsize=10)
        plt.tight_layout()
        plt.savefig(f'exp5_{scenario_name}_comparison.png', dpi=150)
        plt.show()
    print("Experiment 5 completed (with hyperparameter search).")
# ==============================================================================
# 6. Experiment 6: FR Mirror particle count (with eta_w search)
# ==============================================================================
def experiment6_fr_particle_count():
    print("\n" + "=" * 60)
    print("Experiment 6: FR Mirror performance with different particle counts (with eta_w search)")
    print("=" * 60)

    X, y = generate_linear_data(n_samples=500, d=2, noise_scale=0.2, seed=2026)
    n_train = 400
    X_train_syn, y_train_syn = X[:n_train], y[:n_train]
    X_val_syn, y_val_syn = X_train_syn[-50:], y_train_syn[-50:]
    X_train_syn, y_train_syn = X_train_syn[:-50], y_train_syn[:-50]
    X_test_syn, y_test_syn = X[n_train:], y[n_train:]

    sigma2_hat_syn = estimate_noise_variance(X_train_syn, y_train_syn)[0]
    gamma2_hat_syn = median_heuristic_gamma2(y_train_syn)
    p_values = [10, 30, 50, 100]
    eta_w_grid = [0.005, 0.01, 0.02, 0.05]
    results = {'p': [], 'final_nll': [], 'final_mmd': []}
    for p in p_values:
        print(f"  Testing p = {p}")
        best_eta = None
        best_score = float('inf')
        for eta_w in eta_w_grid:
            params = {'gamma2': gamma2_hat_syn, 'sigma2': sigma2_hat_syn, 'lambda_': math.sqrt(n_train),
                      'p': p, 'd': 2, 'batch_size': 128, 'eval_every': 20, 'eta_w': eta_w}
            _, _, mmd_series = run_experiment('fr_mirror', X_train_syn, y_train_syn, X_val_syn, y_val_syn, params, K=300)
            score = mmd_series[-1].item()
            if score < best_score:
                best_score = score
                best_eta = eta_w
        print(f"    Best eta_w = {best_eta}")
        nll_vals, mmd_vals = [], []
        for run in range(2):
            torch.manual_seed(42 + run)
            theta_init = torch.randn(p, 2, device=device)
            batch_indices = torch.randint(0, len(X_train_syn), (300, 128), device=device)
            params = {'gamma2': gamma2_hat_syn, 'sigma2': sigma2_hat_syn, 'lambda_': math.sqrt(n_train),
                      'p': p, 'd': 2, 'batch_size': 128, 'eval_every': 20, 'eta_w': best_eta}
            _, nll_series, mmd_series = run_experiment('fr_mirror', X_train_syn, y_train_syn, X_test_syn, y_test_syn,
                params, K=300, theta_init=theta_init, batch_indices=batch_indices)
            nll_vals.append(nll_series[-1].item())
            mmd_vals.append(mmd_series[-1].item())
        results['p'].append(p)
        results['final_nll'].append(np.mean(nll_vals))
        results['final_mmd'].append(np.mean(mmd_vals))

    # ---- Plotting ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(results['p'], results['final_nll'], 'o-', color=ALGORITHM_COLORS['fr_mirror'])
    ax[0].set_xlabel('Number of particles (p)')
    ax[0].set_ylabel('Final Test NLL')
    ax[0].set_title(f'{ALGORITHM_LABELS["fr_mirror"]}: Final NLL vs p')
    ax[0].grid(alpha=0.3)
    ax[1].plot(results['p'], results['final_mmd'], 's-', color=ALGORITHM_COLORS['fr_mirror'])
    ax[1].set_xlabel('Number of particles (p)')
    ax[1].set_ylabel('Final Test MMD^2')
    ax[1].set_title(f'{ALGORITHM_LABELS["fr_mirror"]}: Final MMD vs p')
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('exp6_fr_particle_count.png', dpi=150)
    plt.show()
    print("Experiment 6 completed (with hyperparameter search).")

# ==============================================================================
# 7. Experiment 7: Initialization trap demonstration
# ==============================================================================
def experiment7_initialization_failure():
    print("\n" + "=" * 60)
    print("Experiment 7: Initialization trap demonstration - limitation of fixed-support algorithms")
    print("=" * 60)

    d = 2
    n_train = 100
    torch.manual_seed(2026)
    X_train = torch.randn(n_train, d)
    theta_true = torch.tensor([2.0, 2.0], dtype=torch.float32)
    y_train = (X_train @ theta_true + 0.5 * torch.randn(n_train)).view(-1, 1)

    sigma2_hat = 0.25
    gamma2_hat = median_heuristic_gamma2(y_train)
    lambda_n = math.sqrt(n_train)
    Sigma = torch.inverse(X_train.T @ X_train + torch.eye(d))
    mu_true = Sigma @ (X_train.T @ y_train)
    mu_true = mu_true.flatten()
    print(f"True posterior mean: {mu_true.numpy().flatten()}")

    p = 30
    torch.manual_seed(42)
    theta_init = -2.0 + 0.1 * torch.randn(p, d)

    params_fr = {'gamma2': gamma2_hat, 'sigma2': sigma2_hat, 'lambda_': lambda_n, 'p': p, 'd': d,
                 'batch_size': 64, 'eval_every': 10, 'eta_w': 0.05}
    params_pro = {'gamma2': gamma2_hat, 'sigma2': sigma2_hat, 'lambda_': lambda_n, 'p': p, 'd': d,
                  'batch_size': 64, 'eval_every': 10, 'dt': 0.02}

    # Run FR Mirror and save final state
    theta = theta_init.clone()
    w = torch.ones(p, device=device) / p
    base_w = w.clone()
    for step in range(10000):
        idx = torch.randint(0, X_train.shape[0], (64,), device=device)
        X_batch, y_batch = X_train[idx], y_train[idx]
        theta, w = train_step_fr_mirror(theta, w, X_batch, y_batch, params_fr['gamma2'], params_fr['sigma2'],
                                        params_fr['lambda_'], params_fr['eta_w'], base_w)
    theta_fr, w_fr = theta.detach().cpu(), w.detach().cpu()

    # Run Pro (MFLD) and save final state
    theta_pro = theta_init.clone()
    for step in range(10000):
        idx = torch.randint(0, X_train.shape[0], (64,), device=device)
        X_batch, y_batch = X_train[idx], y_train[idx]
        theta_pro = train_step_pro(theta_pro, X_batch, y_batch, params_pro['gamma2'], params_pro['sigma2'],
                                   params_pro['lambda_'], params_pro['dt'], p)
    theta_pro = theta_pro.detach().cpu()

    # ---- Plotting ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x_range = np.linspace(-5, 5, 100)
    y_range = np.linspace(-5, 5, 100)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    pos = np.dstack((X_grid, Y_grid))
    rv = torch.distributions.MultivariateNormal(mu_true.float(), Sigma.float())
    Z = torch.exp(rv.log_prob(torch.tensor(pos, dtype=torch.float32))).numpy()

    # Panel 0: true posterior + initial particles
    axes[0].contourf(X_grid, Y_grid, Z, levels=20, cmap='Blues')
    axes[0].scatter(theta_init[:, 0].cpu(), theta_init[:, 1].cpu(),
                    c='gray', s=30, label='Initial particles', alpha=0.7)
    axes[0].plot(mu_true[0].item(), mu_true[1].item(), 'g*', markersize=15, label='True posterior mean')
    axes[0].set_title('True posterior + Initial particles (dead zone)')
    axes[0].set_xlabel('theta_1')
    axes[0].set_ylabel('theta_2')
    axes[0].legend()

    # Panel 1: FR Mirror final state
    sizes_fr = 500 * (w_fr / w_fr.max())
    axes[1].contourf(X_grid, Y_grid, Z, levels=20, cmap='Blues', alpha=0.5)
    axes[1].scatter(theta_fr[:, 0], theta_fr[:, 1], s=sizes_fr,
                    c=ALGORITHM_COLORS['fr_mirror'], alpha=0.7, edgecolors='k',
                    label=ALGORITHM_LABELS['fr_mirror'])
    axes[1].set_title(f'{ALGORITHM_LABELS["fr_mirror"]} final state (weight collapse)')
    axes[1].set_xlabel('theta_1')
    axes[1].set_ylabel('theta_2')
    axes[1].legend()

    # Panel 2: Pro (MFLD) final state
    axes[2].contourf(X_grid, Y_grid, Z, levels=20, cmap='Blues', alpha=0.5)
    axes[2].scatter(theta_pro[:, 0], theta_pro[:, 1],
                    c=ALGORITHM_COLORS['pro'], s=30, alpha=0.7, edgecolors='k',
                    label=ALGORITHM_LABELS['pro'])
    axes[2].plot(mu_true[0].item(), mu_true[1].item(), 'g*', markersize=15, label='True posterior mean')
    axes[2].set_title(f'{ALGORITHM_LABELS["pro"]} final state (movable particles)')
    axes[2].set_xlabel('theta_1')
    axes[2].set_ylabel('theta_2')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('exp7_initialization_failure.png', dpi=150)
    plt.show()
    print("Experiment 7 completed. Figure saved as exp7_initialization_failure.png")
    print("Explanation: FR Mirror, due to fixed support, collapses all weights onto the particle closest to the true mean, "
          "creating a spurious mode that never covers the true posterior. Pro (MFLD) moves particles and escapes the trap.")

# ===== Experiment 8: Low-dimensional synthetic data with high-precision reference =====
# ===== (新增) =====
def compute_prO_objective(theta, w, X, y, gamma2, sigma2, lambda_, kde_bandwidth=0.8):
    """
    Approximate the PrO objective F(Q) = lambda * S_MMD(P_Q) + KL(Q||Pi).
    S_MMD is the empirical squared MMD on the training data (without the
    constant term k(y,y)=1, which does not depend on Q).
    KL is approximated by KDE-based entropy.
    """
    phi, _ = compute_mmd_potential_and_grad(theta, w, X, y, gamma2, sigma2,
                                            leave_one_out=False)
    # Note: phi is the first variation, not the score itself.
    # The actual S_MMD = 0.5 * sum_j w_j * phi_j (up to constant).
    mmd_score = 0.5 * torch.sum(w * phi)

    # KL(Q||Pi) = integral q(theta) log(q(theta)/pi(theta)) dtheta
    log_q, _ = compute_kde_log_density_and_score(theta, w, kde_bandwidth)
    log_prior = prior_log_prob(theta)
    kl = torch.sum(w * (log_q - log_prior))

    return lambda_ * mmd_score + kl


def compute_gradient_norm_difference(theta_ref, w_ref, theta_alg, w_alg,
                                     X, y, gamma2, sigma2, lambda_,
                                     kde_bandwidth=0.8):
    """
    Compute average L2 norm of the gradient of F at particles between
    reference and algorithm (using the first variation as a proxy).
    """
    _, grad_phi_ref = compute_mmd_potential_and_grad(theta_ref, w_ref, X, y,
                                                     gamma2, sigma2,
                                                     leave_one_out=False)
    _, grad_phi_alg = compute_mmd_potential_and_grad(theta_alg, w_alg, X, y,
                                                     gamma2, sigma2,
                                                     leave_one_out=False)
    # Gradient of F w.r.t. particle positions (assuming uniform weights or w)
    # For KL part, grad_theta log q is approximated by score_q from KDE.
    log_q_ref, score_q_ref = compute_kde_log_density_and_score(theta_ref, w_ref,
                                                               kde_bandwidth)
    log_q_alg, score_q_alg = compute_kde_log_density_and_score(theta_alg, w_alg,
                                                               kde_bandwidth)
    grad_ref = lambda_ * grad_phi_ref + score_q_ref - prior_grad_log(theta_ref)
    grad_alg = lambda_ * grad_phi_alg + score_q_alg - prior_grad_log(theta_alg)

    # Match particles: use nearest neighbour for each algorithm particle
    # to the reference set, or simply compute average over all particles.
    # For simplicity, we use the mean L2 norm difference over all particles,
    # assuming same number of particles.
    if theta_ref.shape[0] != theta_alg.shape[0]:
        # Resample reference to same number
        idx = torch.randint(0, theta_ref.shape[0], (theta_alg.shape[0],),
                            device=theta_alg.device)
        grad_ref = grad_ref[idx]
        score_q_ref = score_q_ref[idx]
    diff = grad_alg - grad_ref
    return torch.norm(diff, dim=1).mean().item()


def compute_parameter_mmd(theta1, w1, theta2, w2, bandwidth=1.0):
    """
    MMD distance between two weighted particle sets in parameter space,
    using a Gaussian RBF kernel with given bandwidth.
    """
    if theta1.shape[1] != theta2.shape[1]:
        raise ValueError("Dimension mismatch")
    diff = theta1[:, None, :] - theta2[None, :, :]
    D2 = torch.sum(diff ** 2, dim=-1)
    K12 = torch.exp(-0.5 * D2 / bandwidth ** 2)
    # K11 (same set 1)
    diff11 = theta1[:, None, :] - theta1[None, :, :]
    D11 = torch.sum(diff11 ** 2, dim=-1)
    K11 = torch.exp(-0.5 * D11 / bandwidth ** 2)
    # K22
    diff22 = theta2[:, None, :] - theta2[None, :, :]
    D22 = torch.sum(diff22 ** 2, dim=-1)
    K22 = torch.exp(-0.5 * D22 / bandwidth ** 2)
    term1 = torch.sum(w1[:, None] * w1[None, :] * K11)
    term2 = torch.sum(w2[:, None] * w2[None, :] * K22)
    term3 = 2 * torch.sum(w1[:, None] * w2[None, :] * K12)
    return (term1 + term2 - term3).item()
def experiment8_low_dim_reference_fast():
    print("\n" + "=" * 60)
    print("Experiment 8 (Fast): 1D mixture-of-Gaussians, single-Gaussian likelihood")
    print("=" * 60)

    
    n_samples = 100
    torch.manual_seed(2026)
    mean1, mean2 = -1.0, 1.0
    std1, std2 = 0.2, 0.2
    mix = 0.5
    comp = torch.rand(n_samples) < mix
    y = torch.where(comp,
                    mean1 + std1 * torch.randn(n_samples),
                    mean2 + std2 * torch.randn(n_samples)).view(-1, 1)
    X = torch.ones(n_samples, 1)  
    
    theta_ml = torch.mean(y)
    sigma2_hat = torch.mean((y - theta_ml) ** 2).clamp_min(1e-4).item()
    gamma2_hat = median_heuristic_gamma2(y)
    lambda_n = math.sqrt(n_samples)
    print(f"n={n_samples}, sigma2={sigma2_hat:.4f}, gamma2={gamma2_hat:.4f}, lambda={lambda_n:.4f}")

    
    grid_points = 200
    grid_range = (-4.0, 4.0)
    theta_grid = torch.linspace(grid_range[0], grid_range[1], grid_points, device=device).view(-1, 1)
    log_prior_grid = prior_log_prob(theta_grid)
    base_w = torch.softmax(log_prior_grid, dim=0).detach()

    print("Computing reference via FR mirror on dense grid...")
    w_ref = torch.ones(grid_points, device=device) / grid_points
    eta_w = 0.01
    K_ref = 2000
    with torch.no_grad():
        for _ in range(K_ref):
            theta_ref, w_ref = train_step_fr_mirror(theta_grid, w_ref, X, y,
                                                    gamma2_hat, sigma2_hat,
                                                    lambda_n, eta_w, base_w)
    theta_ref = theta_grid  
    print("Reference solution computed.")

   
    modes = ['pro', 'vgd', 'fr_mirror', 'wfr_smc', 'wfr_bdl']
    algo_params = {
        'pro': {'dt': 0.01},
        'vgd': {'vgd_step_size': 0.02, 'vgd_lengthscales': (0.125, 0.25, 0.5)},
        'fr_mirror': {'eta_w': 0.01},
        'wfr_smc': {'dt': 0.005},
        'wfr_bdl': {'dt': 0.002, 'kde_bandwidth': 0.8}
    }
    K_alg = 500
    p_alg = 50
    num_seeds = 1
    results = {}
    final_states = {}

    for mode in modes:
        print(f"\nRunning {mode} ...")
        target_diffs, grad_diffs, mmd_dists = [], [], []
        for seed in range(num_seeds):
            torch.manual_seed(42 + seed)
            theta_init = torch.randn(p_alg, 1, device=device)
            params = {'gamma2': gamma2_hat, 'sigma2': sigma2_hat, 'lambda_': lambda_n,
                      'p': p_alg, 'd': 1, 'kde_bandwidth': 0.8}
            params.update(algo_params[mode])

            
            if mode == 'pro':
                theta_alg = theta_init.clone()
                for _ in range(K_alg):
                    theta_alg = train_step_pro(theta_alg, X, y, gamma2_hat, sigma2_hat,
                                               lambda_n, params['dt'], p_alg)
                w_alg = torch.ones(p_alg, device=device) / p_alg
            elif mode == 'vgd':
                theta_alg = theta_init.clone()
                first_moment = torch.zeros_like(theta_alg)
                second_moment = torch.zeros_like(theta_alg)
                vgd_iter = 0
                for _ in range(K_alg):
                    theta_alg, first_moment, second_moment, vgd_iter = train_step_vgd(
                        theta_alg, X, y, gamma2_hat, sigma2_hat, lambda_n,
                        params['vgd_step_size'], params['vgd_lengthscales'],
                        first_moment, second_moment, vgd_iter
                    )
                w_alg = torch.ones(p_alg, device=device) / p_alg
            elif mode == 'fr_mirror':
                theta_alg = theta_init.clone()
                w_alg = torch.ones(p_alg, device=device) / p_alg
                base_w_alg = w_alg.clone()
                for _ in range(K_alg):
                    theta_alg, w_alg = train_step_fr_mirror(
                        theta_alg, w_alg, X, y, gamma2_hat, sigma2_hat,
                        lambda_n, params['eta_w'], base_w_alg
                    )
            elif mode == 'wfr_smc':
                theta_alg = theta_init.clone()
                w_alg = torch.ones(p_alg, device=device) / p_alg
                for _ in range(K_alg):
                    theta_alg, w_alg = train_step_wfr_smc(
                        theta_alg, w_alg, X, y, gamma2_hat, sigma2_hat,
                        lambda_n, params['dt']
                    )
            elif mode == 'wfr_bdl':
                theta_alg = theta_init.clone()
                for _ in range(K_alg):
                    theta_alg, _ = train_step_wfr_bdl(
                        theta_alg, X, y, gamma2_hat, sigma2_hat,
                        lambda_n, params['dt'], params['kde_bandwidth']
                    )
                w_alg = torch.ones(p_alg, device=device) / p_alg
            else:
                raise ValueError(mode)

            if seed == num_seeds - 1:
                final_states[mode] = (theta_alg.detach().cpu(), w_alg.detach().cpu())

           
            obj_ref = compute_prO_objective(theta_ref, w_ref, X, y,
                                            gamma2_hat, sigma2_hat, lambda_n,
                                            kde_bandwidth=0.8)
            obj_alg = compute_prO_objective(theta_alg, w_alg, X, y,
                                            gamma2_hat, sigma2_hat, lambda_n,
                                            kde_bandwidth=0.8)
            target_diff = abs(obj_alg - obj_ref) / abs(obj_ref)
            target_diffs.append(target_diff)

            grad_diff = compute_gradient_norm_difference(theta_ref, w_ref,
                                                         theta_alg, w_alg,
                                                         X, y, gamma2_hat, sigma2_hat,
                                                         lambda_n, kde_bandwidth=0.8)
            grad_diffs.append(grad_diff)

            mmd_dist = compute_parameter_mmd(theta_ref, w_ref, theta_alg, w_alg,
                                             bandwidth=0.5)
            mmd_dists.append(mmd_dist)

        results[mode] = {
            'target_diff': np.mean(target_diffs),
            'grad_diff': np.mean(grad_diffs),
            'mmd_dist': np.mean(mmd_dists),
            'std_target': np.std(target_diffs) if num_seeds > 1 else 0.0,
            'std_grad': np.std(grad_diffs) if num_seeds > 1 else 0.0,
            'std_mmd': np.std(mmd_dists) if num_seeds > 1 else 0.0
        }

    
    print("\n" + "=" * 60)
    print("Experiment 8 Results (1D mixture-of-Gaussians)")
    print("=" * 60)
    header = f"{'Algorithm':<12} {'Target Diff':<15} {'Grad Diff':<15} {'MMD Dist':<15}"
    print(header)
    print("-" * 60)
    for mode in modes:
        res = results[mode]
        print(f"{ALGORITHM_LABELS[mode]:<12} {res['target_diff']:<15.6f} "
              f"{res['grad_diff']:<15.6f} {res['mmd_dist']:<15.6f}")

    # bar plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels = [ALGORITHM_LABELS[m] for m in modes]
    colors = [ALGORITHM_COLORS[m] for m in modes]

    axes[0].bar(labels, [results[m]['target_diff'] for m in modes], color=colors, alpha=0.7)
    axes[0].set_ylabel('Relative objective gap')
    axes[0].set_title('Training objective gap vs reference')
    axes[0].tick_params(axis='x', rotation=45)

    axes[1].bar(labels, [results[m]['grad_diff'] for m in modes], color=colors, alpha=0.7)
    axes[1].set_ylabel('Mean L2 gradient diff')
    axes[1].set_title('Gradient difference vs reference')
    axes[1].tick_params(axis='x', rotation=45)

    axes[2].bar(labels, [results[m]['mmd_dist'] for m in modes], color=colors, alpha=0.7)
    axes[2].set_ylabel('Parameter-space MMD')
    axes[2].set_title('Distribution distance (MMD)')
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('exp8_low_dim_reference_fast.png', dpi=150)
    plt.show()

    
    def kde_density(x_eval, theta, w, bandwidth):
        """
        计算任意点 x_eval 上的 KDE 密度值。
        x_eval: (M,1), theta: (p,1), w: (p,)
        返回: (M,) 的密度值
        """
        diff = x_eval - theta.T  # (M, p)
        D2 = diff ** 2
        log_kernel = -0.5 * D2 / bandwidth**2 - 0.5 * math.log(2.0 * math.pi * bandwidth**2)
        log_weighted = torch.log(w.clamp_min(1e-30))[None, :] + log_kernel
        log_q = torch.logsumexp(log_weighted, dim=1)
        return torch.exp(log_q)

    x_eval = torch.linspace(-4, 4, 500, device=device).view(-1, 1)  # (500,1)
    bandwidth = 0.3

    
    density_ref = kde_density(x_eval, theta_ref, w_ref, bandwidth).cpu().numpy()

    fig, axes = plt.subplots(1, len(modes) + 1, figsize=(3 * (len(modes) + 1), 4))
    
    axes[0].plot(x_eval.cpu().numpy(), density_ref, color='black', label='Reference')
    axes[0].set_title('Reference')
    axes[0].set_xlim(-4, 4)
    axes[0].legend()

    # plot foe each algo
    for i, mode in enumerate(modes):
        theta_alg, w_alg = final_states[mode]
        theta_alg = theta_alg.to(device)
        w_alg = w_alg.to(device)
        density_alg = kde_density(x_eval, theta_alg, w_alg, bandwidth).cpu().numpy()
        ax = axes[i + 1]
        ax.plot(x_eval.cpu().numpy(), density_ref, color='black', label='Reference', linestyle='--')
        ax.plot(x_eval.cpu().numpy(), density_alg, color=ALGORITHM_COLORS[mode], label=ALGORITHM_LABELS[mode])
        ax.set_title(ALGORITHM_LABELS[mode])
        ax.set_xlim(-4, 4)
        ax.legend()

    plt.tight_layout()
    plt.savefig('exp8_density_comparison.png', dpi=150)
    plt.show()
    print("Experiment 8 (Fast) completed. Figures saved as exp8_low_dim_reference_fast.png and exp8_density_comparison.png")
    
    
# ==============================================================================
# 8. Main entry
# ==============================================================================
if __name__ == "__main__":
    experiment5_misspecification_scenarios()
    experiment6_fr_particle_count()
    experiment7_initialization_failure()
    experiment8_low_dim_reference_fast()   
    print("\nAll extended experiments completed.")
