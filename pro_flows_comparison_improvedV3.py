"""
pro_flows_comparison_improved_V3_final.py
==========================================
Extended experiments: additional comparative studies of PrO posterior (final improved version, addressing reviewer comment 20).

This script adds three experiments to the original:
1. Bayes vs PrO (MFLD) on synthetic data
2. Comparison of all PrO algorithms on synthetic data (trajectories, weights)
3. PrO vs SMC-WFR vs Bayes exact on real dataset (UCI Concrete or diabetes),
   with separate hyperparameter search (dt, lambda, gamma2), larger particle counts,
   more iterations, and progress bars.

Improvements (Experiment 3, addressing reviewer comment 20):
- Separate grid search on validation set for PrO and SMC-WFR (dt, lambda, gamma2)
- Separate hyperparameter search for Bayes MFLD (dt, prior_precision)
- Run Bayes MFLD and record particle norms and gradient norms
- Explicitly state that mini-batch likelihood is used
- Compare Bayes MFLD to exact Bayesian posterior
"""

import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import itertools

# Attempt to import tqdm; fallback to a simple alternative
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, **kwargs):
        print(f"Progress: {desc}")
        return iterable

# ==================== Unified algorithm style mappings ====================
ALGORITHM_LABELS = {
    'pro': 'PrO (MFLD)',
    'underdamped_pro': 'PrO (NULA)',
    'vgd': 'VGD',
    'fr_mirror': 'FR Mirror',
    'wfr_smc': 'SMC-WFR',
    'wfr_bdl': 'BDL-WFR',
    'wfr': 'KDE-WFR',
    'fr': 'KDE-FR',
    'bayes': 'Bayes (MFLD)',
}
ALGORITHM_COLORS = {
    'pro': '#e41a1c',
    'underdamped_pro': '#ff7f00',
    'vgd': '#4daf4a',
    'fr_mirror': '#984ea3',
    'wfr_smc': '#377eb8',
    'wfr_bdl': '#f781bf',
    'wfr': '#a65628',
    'fr': '#1f78b4',
    'bayes': '#999999'
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

# ==============================================================================
# Core utility functions (copied from pro_flows_comparison to avoid external dependencies)
# ==============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prior_log_prob(theta):
    return -0.5 * torch.sum(theta ** 2, dim=-1)

def prior_grad_log(theta):
    return -theta

def prior_sample(key, p, d):
    return torch.randn(p, d, device=device)

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

# ---- MMD potential and gradient ----
def compute_mmd_potential_and_grad(theta, w, X_batch, y_batch, gamma2, sigma2, leave_one_out=False):
    B = X_batch.shape[0]
    p = theta.shape[0]
    if gamma2 <= 0 or sigma2 <= 0:
        raise ValueError("gamma2 and sigma2 must both be strictly positive.")
    if leave_one_out and p < 2:
        raise ValueError("leave_one_out=True requires at least two particles.")
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

def compute_mmd_potential_and_grad_at(theta_eval, theta_measure, w_measure, X_batch, y_batch, gamma2, sigma2):
    B = X_batch.shape[0]
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

# ---- KDE ----
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

# ---- Resampling ----
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

# ---- Algorithm step functions ----
def train_step_pro(theta, X_batch, y_batch, gamma2, sigma2, lambda_, dt, p):
    w = torch.ones(p, device=device) / p
    _, grad_score = compute_mmd_potential_and_grad(theta, w, X_batch, y_batch, gamma2, sigma2, leave_one_out=True)
    grad_prior = prior_grad_log(theta)
    noise = torch.randn_like(theta) * math.sqrt(2.0 * dt)
    theta_new = theta - dt * (lambda_ * grad_score - grad_prior) + noise
    return theta_new.detach()

def train_step_underdamped_pro(theta, velocity, X_batch, y_batch, gamma2, sigma2, lambda_, step_size, friction, p):
    w = torch.ones(p, device=theta.device) / p
    _, grad_phi = compute_mmd_potential_and_grad(theta, w, X_batch, y_batch, gamma2, sigma2, leave_one_out=False)
    force = lambda_ * grad_phi - prior_grad_log(theta)
    h = float(step_size)
    gamma = float(friction)
    phi2 = math.exp(-gamma * h)
    phi0 = (1.0 - phi2) / gamma
    phi1 = (h - phi0) / gamma
    sigma11 = (2.0 / gamma) * (h - 2.0 * phi0 + (1.0 - phi2 ** 2) / (2.0 * gamma))
    sigma12 = (1.0 - phi2) ** 2 / gamma
    sigma22 = 1.0 - phi2 ** 2
    sigma11 = max(sigma11, 0.0)
    sigma22 = max(sigma22, 0.0)
    z_position = torch.randn_like(theta)
    z_velocity = torch.randn_like(theta)
    if sigma11 <= 1e-30:
        noise_position = torch.zeros_like(theta)
        noise_velocity = math.sqrt(sigma22) * z_velocity
    else:
        sqrt_sigma11 = math.sqrt(sigma11)
        conditional_coefficient = sigma12 / sqrt_sigma11
        conditional_variance = max(sigma22 - sigma12**2 / sigma11, 0.0)
        noise_position = sqrt_sigma11 * z_position
        noise_velocity = conditional_coefficient * z_position + math.sqrt(conditional_variance) * z_velocity
    theta_new = theta + phi0 * velocity - phi1 * force + noise_position
    velocity_new = phi2 * velocity - phi0 * force + noise_velocity
    return theta_new.detach(), velocity_new.detach()

def compute_vgd_velocity(theta, X_batch, y_batch, gamma2, sigma2, lambda_, lengthscales):
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
    velocity = compute_vgd_velocity(theta, X_batch, y_batch, gamma2, sigma2, lambda_, lengthscales)
    iteration += 1
    first_moment = beta1 * first_moment + (1.0 - beta1) * velocity
    second_moment = beta2 * second_moment + (1.0 - beta2) * velocity.square()
    first_hat = first_moment / (1.0 - beta1 ** iteration)
    second_hat = second_moment / (1.0 - beta2 ** iteration)
    theta_new = theta + step_size * first_hat / (torch.sqrt(second_hat) + adam_epsilon)
    return theta_new.detach(), first_moment.detach(), second_moment.detach(), iteration

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

def update_weights_fisher_rao(w, eta, eta_w):
    eta_bar = torch.sum(w * eta)
    log_w_new = torch.log(w.clamp_min(1e-30)) - eta_w * (eta - eta_bar)
    return torch.softmax(log_w_new, dim=0)

# ---- Bayes MFLD specific ----
def train_step_bayes(theta, X_batch, y_batch, sigma2, prior_precision, dt, p):
    """
    Bayes MFLD step. Uses mini-batch likelihood gradient.
    """
    pred = X_batch @ theta.T
    residuals = pred - y_batch
    grad_likelihood = (X_batch.T @ residuals) / X_batch.shape[0] / sigma2
    grad_likelihood = grad_likelihood.T
    grad_prior = prior_precision * theta
    force = grad_likelihood - grad_prior
    noise = torch.randn_like(theta) * math.sqrt(2.0 * dt)
    theta_new = theta + dt * force + noise
    return theta_new.detach()

def compute_bayes_force_norm(theta, X_batch, y_batch, sigma2, prior_precision):
    """Compute average L2 norm of the force for Bayes MFLD."""
    pred = X_batch @ theta.T
    residuals = pred - y_batch
    grad_likelihood = (X_batch.T @ residuals) / X_batch.shape[0] / sigma2
    grad_likelihood = grad_likelihood.T
    grad_prior = prior_precision * theta
    force = grad_likelihood - grad_prior
    norm = torch.mean(torch.norm(force, dim=1))
    return norm.item()

# ---- Evaluation functions ----
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
# Extended run_experiment (supports recording particle norms, gradient norms)
# ==============================================================================
def run_experiment_extended(
    mode,
    X_train,
    y_train,
    X_eval,
    y_eval,
    params,
    K,
    target_type='pro',
    theta_init=None,
    batch_indices=None,
    record_particles=False,
    record_every=10,
    record_particle_norm=False,
    record_grad_norm=False,
):
    d = X_train.shape[1]
    if theta_init is None:
        theta = torch.randn(params['p'], d, device=device)
    else:
        theta = theta_init.clone().to(device)

    w = torch.ones(params['p'], device=device) / params['p']
    weight_history = [] if record_particles else None
    particle_history = [] if record_particles else None
    particle_norm_history = [] if record_particle_norm else None
    grad_norm_history = [] if record_grad_norm else None

    momentum_buffer = torch.zeros_like(theta)
    vgd_first_moment = torch.zeros_like(theta)
    vgd_second_moment = torch.zeros_like(theta)
    vgd_iteration = 0
    underdamped_velocity = torch.randn_like(theta) if mode == 'underdamped_pro' else None
    base_w = w.clone()

    record_steps = []
    eval_losses = []
    eval_mmd = []

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

        if target_type == 'pro':
            if mode == 'pro':
                theta = train_step_pro(theta, X_batch, y_batch, params['gamma2'], params['sigma2'],
                                       params['lambda_'], params['dt'], params['p'])
                w = torch.ones(params['p'], device=device) / params['p']
            elif mode == 'underdamped_pro':
                theta, underdamped_velocity = train_step_underdamped_pro(
                    theta, underdamped_velocity, X_batch, y_batch,
                    params['gamma2'], params['sigma2'], params['lambda_'],
                    params['nula_step_size'], params['nula_friction'], params['p'])
                w = torch.ones(params['p'], device=device) / params['p']
            elif mode == 'vgd':
                theta, vgd_first_moment, vgd_second_moment, vgd_iteration = train_step_vgd(
                    theta, X_batch, y_batch, params['gamma2'], params['sigma2'], params['lambda_'],
                    params['vgd_step_size'], params['vgd_lengthscales'],
                    vgd_first_moment, vgd_second_moment, vgd_iteration)
                w = torch.ones(params['p'], device=device) / params['p']
            elif mode == 'fr_mirror':
                theta, w = train_step_fr_mirror(theta, w, X_batch, y_batch,
                                                params['gamma2'], params['sigma2'], params['lambda_'],
                                                params['eta_w'], base_w)
            elif mode == 'wfr_smc':
                theta, w = train_step_wfr_smc(theta, w, X_batch, y_batch,
                                              params['gamma2'], params['sigma2'], params['lambda_'],
                                              params['dt'])
            elif mode == 'wfr_bdl':
                theta, w = train_step_wfr_bdl(theta, X_batch, y_batch,
                                              params['gamma2'], params['sigma2'], params['lambda_'],
                                              params['dt'], params['kde_bandwidth'])
            elif mode == 'wfr_imp':
                theta, w, momentum_buffer = train_step_wfr_imp(
                    theta, w, X_batch, y_batch, params['gamma2'], params['sigma2'], params['lambda_'],
                    params['eta_theta'], params['eta_w'], params['p'],
                    momentum_buffer, params['kde_bandwidth'])
            else:
                theta, w = train_step_old(theta, w, X_batch, y_batch,
                                          params['gamma2'], params['sigma2'], params['lambda_'],
                                          params['eta_theta'], params['eta_w'], params['p'],
                                          mode, params['kde_bandwidth'])
        elif target_type == 'bayes':
            if mode != 'pro':
                raise ValueError("Bayes posterior currently only supports 'pro' mode (MFLD)")
            theta = train_step_bayes(theta, X_batch, y_batch, params['sigma2'],
                                     params['prior_precision'], params['dt'], params['p'])
            w = torch.ones(params['p'], device=device) / params['p']
        else:
            raise ValueError(f"Unknown target_type: {target_type}")

        # ---------- Record particle trajectories, weights, particle norms and gradient norms ----------
        if (record_particles or record_particle_norm or record_grad_norm) and (step % record_every == 0 or step == K-1):
            if record_particles:
                particle_history.append(theta.detach().cpu().numpy())
                weight_history.append(w.detach().cpu().numpy())
            if record_particle_norm:
                theta_norm = torch.mean(torch.norm(theta, dim=1)).item()
                particle_norm_history.append(theta_norm)
            if record_grad_norm:
                if target_type == 'bayes':
                    grad_norm = compute_bayes_force_norm(theta, X_batch, y_batch,
                                                         params['sigma2'], params['prior_precision'])
                else:
                    grad_norm = compute_force_norm(theta, X_batch, y_batch,
                                                   params['sigma2'], params['lambda_'])
                grad_norm_history.append(grad_norm)

        # ---------- Evaluation ----------
        if step % eval_every == 0 or step == K - 1:
            with torch.no_grad():
                nll = compute_predictive_nll(theta, w, X_eval, y_eval, params['sigma2'])
                mmd = compute_predictive_mmd(theta, w, X_mmd_eval, y_mmd_eval,
                                             params['gamma2'], params['sigma2'])
            record_steps.append(step)
            eval_losses.append(nll)
            eval_mmd.append(mmd)

    if record_particles:
        particle_history = np.array(particle_history)
        weight_history = np.array(weight_history)
    if record_particle_norm and particle_norm_history is not None:
        particle_norm_history = np.array(particle_norm_history)
    if record_grad_norm and grad_norm_history is not None:
        grad_norm_history = np.array(grad_norm_history)

    return (
        torch.tensor(record_steps, dtype=torch.long),
        torch.tensor(eval_losses, dtype=torch.float32),
        torch.tensor(eval_mmd, dtype=torch.float32),
        particle_history,
        weight_history,
        particle_norm_history,
        grad_norm_history,
    )

def compute_force_norm(theta, X_batch, y_batch, sigma2, lambda_):
    """Compute average L2 norm of the force for PrO."""
    pred = X_batch @ theta.T
    residuals = pred - y_batch
    grad_likelihood = (X_batch.T @ residuals) / X_batch.shape[0] / sigma2
    grad_likelihood = grad_likelihood.T
    grad_prior = lambda_ * theta
    force = grad_likelihood - grad_prior
    norm = torch.mean(torch.norm(force, dim=1))
    return norm.item()

# ==============================================================================
# Experiment 1: PrO vs. Bayes (synthetic data)
# ==============================================================================
def generate_nonlinear_data(n_samples=1000, d=1, noise_scale=0.1, seed=42):
    torch.manual_seed(seed)
    X = torch.randn(n_samples, d)
    if d == 1:
        y = X[:, 0] ** 2 + noise_scale * torch.randn(n_samples)
    else:
        y = torch.sum(X ** 2, dim=1) + noise_scale * torch.randn(n_samples)
    return X, y.view(-1, 1)

def generate_linear_data(n_samples=1000, d=1, noise_scale=0.1, seed=42):
    torch.manual_seed(seed)
    X = torch.randn(n_samples, d)
    theta_true = torch.randn(d)
    y = X @ theta_true + noise_scale * torch.randn(n_samples)
    return X, y.view(-1, 1)

def experiment1_prO_vs_bayes():
    print("\n" + "=" * 60)
    print("Experiment 1: PrO posterior vs. Bayes posterior (MFLD)")
    print("=" * 60)

    X, y = generate_nonlinear_data(n_samples=500, d=1, noise_scale=0.2, seed=2026)
    n_train = 400
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    theta_ols = torch.linalg.lstsq(X_train, y_train, rcond=None)[0]
    residuals = y_train - X_train @ theta_ols
    sigma2_hat = torch.mean(residuals ** 2).clamp_min(1e-4).item()

    p = 50
    lambda_n = math.sqrt(n_train)
    gamma2_hat = median_heuristic_gamma2(y_train)

    # Added prior_precision key
    params = {
        'gamma2': gamma2_hat,
        'lambda_': lambda_n,
        'sigma2': sigma2_hat,
        'dt': 0.002,
        'p': p,
        'batch_size': 128,
        'eval_every': 20,
        'prior_precision': 1.0,   # for Bayes mode prior precision
    }

    K = 500
    num_runs = 3
    seed = 42

    bayes_nll_list = []
    prO_nll_list = []
    bayes_particles_list = []
    prO_particles_list = []

    for run in range(num_runs):
        torch.manual_seed(seed + run)
        theta_init = torch.randn(p, X_train.shape[1], device=device)
        batch_indices = torch.randint(0, n_train, (K, params['batch_size']), device=device)

        steps_b, nll_b, mmd_b, hist_b, _, _, _ = run_experiment_extended(
            mode='pro', X_train=X_train, y_train=y_train,
            X_eval=X_test, y_eval=y_test,
            params=params, K=K, target_type='bayes',
            theta_init=theta_init, batch_indices=batch_indices,
            record_particles=True, record_every=50
        )
        bayes_nll_list.append(nll_b)
        bayes_particles_list.append(hist_b)

        steps_p, nll_p, mmd_p, hist_p, _, _, _ = run_experiment_extended(
            mode='pro', X_train=X_train, y_train=y_train,
            X_eval=X_test, y_eval=y_test,
            params=params, K=K, target_type='pro',
            theta_init=theta_init, batch_indices=batch_indices,
            record_particles=True, record_every=50
        )
        prO_nll_list.append(nll_p)
        prO_particles_list.append(hist_p)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    steps = steps_b.numpy()
    bayes_mean = torch.stack(bayes_nll_list, dim=0).mean(dim=0).numpy()
    bayes_std = torch.stack(bayes_nll_list, dim=0).std(dim=0).numpy()
    pro_mean = torch.stack(prO_nll_list, dim=0).mean(dim=0).numpy()
    pro_std = torch.stack(prO_nll_list, dim=0).std(dim=0).numpy()

    axes[0].plot(steps, bayes_mean, label=ALGORITHM_LABELS['bayes'],
                 color=ALGORITHM_COLORS['bayes'], linestyle=ALGORITHM_LINESTYLES['bayes'])
    axes[0].fill_between(steps, bayes_mean - bayes_std, bayes_mean + bayes_std,
                         alpha=0.2, color=ALGORITHM_COLORS['bayes'])
    axes[0].plot(steps, pro_mean, label=ALGORITHM_LABELS['pro'],
                 color=ALGORITHM_COLORS['pro'], linestyle=ALGORITHM_LINESTYLES['pro'])
    axes[0].fill_between(steps, pro_mean - pro_std, pro_mean + pro_std,
                         alpha=0.2, color=ALGORITHM_COLORS['pro'])
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Test negative log-likelihood')
    axes[0].set_title('Predictive NLL: Bayes vs PrO (misspecified)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    final_bayes = bayes_particles_list[-1][-1]
    final_pro = prO_particles_list[-1][-1]
    X_test_np = X_test.cpu().numpy().flatten()
    y_test_np = y_test.cpu().numpy().flatten()
    axes[1].scatter(X_test_np, y_test_np, alpha=0.3, label='Test data', s=10, color='gray')
    x_range = np.linspace(X_test_np.min(), X_test_np.max(), 100)
    for i in range(min(20, p)):
        theta_b = final_bayes[i, 0]
        y_b = theta_b * x_range
        axes[1].plot(x_range, y_b, color=ALGORITHM_COLORS['bayes'], alpha=0.1, linewidth=0.5)
        theta_p = final_pro[i, 0]
        y_p = theta_p * x_range
        axes[1].plot(x_range, y_p, color=ALGORITHM_COLORS['pro'], alpha=0.1, linewidth=0.5)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_title('Particle lines (last iteration)')
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=ALGORITHM_COLORS['bayes'], lw=1),
                    Line2D([0], [0], color=ALGORITHM_COLORS['pro'], lw=1),
                    Line2D([0], [0], color='gray', marker='o', linestyle='none')]
    axes[1].legend(custom_lines, [ALGORITHM_LABELS['bayes'], ALGORITHM_LABELS['pro'], 'Data'])
    plt.tight_layout()
    plt.savefig('exp1_bayes_vs_pro.png', dpi=150)
    plt.show()
    print("Experiment 1 finished, figure saved as exp1_bayes_vs_pro.png")

# ==============================================================================
# Experiment 2: Comparison of all PrO algorithms on synthetic data
# ==============================================================================
def experiment2_algorithm_comparison_synthetic():
    print("\n" + "=" * 60)
    print("Experiment 2: Comparison of all PrO algorithms on synthetic data")
    print("=" * 60)

    X, y = generate_linear_data(n_samples=500, d=1, noise_scale=0.2, seed=2026)
    n_train = 400
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    theta_ols = torch.linalg.lstsq(X_train, y_train, rcond=None)[0]
    residuals = y_train - X_train @ theta_ols
    sigma2_hat = torch.mean(residuals ** 2).clamp_min(1e-4).item()
    gamma2_hat = median_heuristic_gamma2(y_train)
    lambda_n = math.sqrt(n_train)

    base_params = {
        'gamma2': gamma2_hat,
        'lambda_': lambda_n,
        'sigma2': sigma2_hat,
        'p': 30,
        'batch_size': 128,
        'eval_every': 20,
        'kde_bandwidth': 0.8,
        'eta_theta': 0.0005,
        'eta_w': 0.01,
    }

    algo_params = {
        'pro': {'dt': 0.002},
        'underdamped_pro': {'nula_step_size': 0.02, 'nula_friction': 5.0},
        'vgd': {'vgd_step_size': 0.02, 'vgd_lengthscales': (0.125, 0.25, 0.5)},
        'fr_mirror': {'eta_w': 0.01},
        'wfr_smc': {'dt': 0.002},
        'wfr_bdl': {'dt': 0.001, 'kde_bandwidth': 0.8},
        'wfr': {'eta_theta': 0.0005, 'eta_w': 0.005},
        'fr': {'eta_w': 0.005},
    }
    all_params = {}
    for mode, extra in algo_params.items():
        p = base_params.copy()
        p.update(extra)
        all_params[mode] = p

    modes = list(algo_params.keys())
    K = 500
    num_runs = 2
    seed = 42

    results = {mode: {'nll': [], 'mmd': [], 'particles': [], 'weights': []} for mode in modes}

    for run in range(num_runs):
        torch.manual_seed(seed + run)
        theta_init = torch.randn(base_params['p'], X_train.shape[1], device=device)
        batch_indices = torch.randint(0, n_train, (K, base_params['batch_size']), device=device)

        for mode in modes:
            print(f"  Run {run+1}, mode {mode}")
            steps, nll_series, mmd_series, part_hist, w_hist, _, _ = run_experiment_extended(
                mode=mode, X_train=X_train, y_train=y_train,
                X_eval=X_test, y_eval=y_test,
                params=all_params[mode], K=K, target_type='pro',
                theta_init=theta_init, batch_indices=batch_indices,
                record_particles=True, record_every=50
            )
            results[mode]['nll'].append(nll_series)
            results[mode]['mmd'].append(mmd_series)
            results[mode]['particles'].append(part_hist)
            results[mode]['weights'].append(w_hist)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for mode in modes:
        nll_mat = torch.stack(results[mode]['nll'], dim=0).numpy()
        mean_nll = nll_mat.mean(axis=0)
        std_nll = nll_mat.std(axis=0)
        axes[0].plot(steps.numpy(), mean_nll, label=ALGORITHM_LABELS[mode],
                     color=ALGORITHM_COLORS[mode], linestyle=ALGORITHM_LINESTYLES[mode])
        axes[0].fill_between(steps.numpy(), mean_nll - std_nll, mean_nll + std_nll,
                             alpha=0.1, color=ALGORITHM_COLORS[mode])
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Test NLL')
    axes[0].set_title('NLL curves (synthetic data)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for mode in modes:
        mmd_mat = torch.stack(results[mode]['mmd'], dim=0).numpy()
        mean_mmd = mmd_mat.mean(axis=0)
        std_mmd = mmd_mat.std(axis=0)
        axes[1].plot(steps.numpy(), mean_mmd, label=ALGORITHM_LABELS[mode],
                     color=ALGORITHM_COLORS[mode], linestyle=ALGORITHM_LINESTYLES[mode])
        axes[1].fill_between(steps.numpy(), mean_mmd - std_mmd, mean_mmd + std_mmd,
                             alpha=0.1, color=ALGORITHM_COLORS[mode])
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Test MMD^2')
    axes[1].set_title('MMD curves (synthetic data)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('exp2_algo_comparison_synthetic.png', dpi=150)
    plt.show()

    for mode in modes:
        part_hist = results[mode]['particles'][-1]
        w_hist = results[mode]['weights'][-1]
        T, p, d = part_hist.shape
        fig, ax = plt.subplots(figsize=(8, 5))
        for i in range(p):
            ax.plot(range(T), part_hist[:, i, 0], alpha=0.6, linewidth=0.8, color=ALGORITHM_COLORS[mode])
        ax.set_xlabel('Iteration (recorded)')
        ax.set_ylabel('Particle location (θ)')
        ax.set_title(f'Particle trajectories: {ALGORITHM_LABELS[mode]}')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'exp2_{mode}_trajectory.png', dpi=150)
        plt.close()

        if mode in ['fr_mirror', 'wfr_smc', 'wfr_bdl', 'wfr', 'fr']:
            fig, ax = plt.subplots(figsize=(8, 5))
            for i in range(p):
                ax.plot(range(T), w_hist[:, i], alpha=0.6, linewidth=0.8, color=ALGORITHM_COLORS[mode])
            ax.set_xlabel('Iteration (recorded)')
            ax.set_ylabel('Weight')
            ax.set_title(f'Particle weights: {ALGORITHM_LABELS[mode]}')
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'exp2_{mode}_weights.png', dpi=150)
            plt.close()

    print("Experiment 2 finished, figures saved as exp2_*.png")

# ==============================================================================
# Experiment 3: Real dataset + separate hyperparameter search + Bayes MFLD comparison (addresses comment 20)
# ==============================================================================
def load_real_dataset():
    try:
        print("Attempting to load Concrete Compressive Strength dataset from OpenML by name...")
        concrete = fetch_openml(name='Concrete_Compressive_Strength', version=1, as_frame=True)
        X = concrete.data.to_numpy().astype(np.float32)
        y = concrete.target.to_numpy().astype(np.float32).reshape(-1, 1)
        print("Successfully loaded Concrete dataset!")
        return X, y
    except Exception as e:
        print(f"Loading by name failed: {e}")

    try:
        print("Attempting to load by data_id=165...")
        concrete = fetch_openml(data_id=165, as_frame=True)
        X = concrete.data.to_numpy().astype(np.float32)
        y = concrete.target.to_numpy().astype(np.float32).reshape(-1, 1)
        print("Successfully loaded Concrete dataset!")
        return X, y
    except Exception as e:
        print(f"Loading by data_id failed: {e}")

    print("Both UCI loading methods failed, falling back to sklearn diabetes dataset.")
    diabetes = load_diabetes()
    X = diabetes.data.astype(np.float32)
    y = diabetes.target.astype(np.float32).reshape(-1, 1)
    print("Successfully loaded diabetes dataset!")
    return X, y

def compute_bayes_posterior_linear(X, y, sigma2, prior_precision=1.0):
    XtX = X.T @ X
    Xty = X.T @ y
    Sigma_inv = XtX / sigma2 + prior_precision * torch.eye(X.shape[1], device=X.device)
    Sigma = torch.linalg.inv(Sigma_inv)
    mu = Sigma @ (Xty / sigma2)
    return mu, Sigma

def experiment3_real_dataset():
    print("\n" + "=" * 60)
    print("Experiment 3 (final improved version): Separate hyperparameter search + Bayes MFLD comparison")
    print("=" * 60)

    # ---------- Load and preprocess data ----------
    X_np, y_np = load_real_dataset()
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_np, y_np, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=43
    )
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device).view(-1, 1)
    X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val = torch.tensor(y_val, dtype=torch.float32, device=device).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.float32, device=device).view(-1, 1)

    n_train = X_train.shape[0]
    d = X_train.shape[1]

    theta_ols = torch.linalg.lstsq(X_train, y_train, rcond=None)[0]
    residuals = y_train - X_train @ theta_ols
    sigma2_init = torch.mean(residuals ** 2).clamp_min(1e-4).item()
    gamma2_init = median_heuristic_gamma2(y_train)
    base_lambda = math.sqrt(n_train)

    print(f"Dataset: d={d}, n_train={n_train}")
    print(f"Initial sigma2={sigma2_init:.4f}, gamma2={gamma2_init:.4f}, base_lambda={base_lambda:.4f}")

    # ---------- Helper: hyperparameter search with progress bar (PrO) ----------
    def search_hyperparams_prO(mode, dt_candidates, lambda_candidates, gamma2_candidates=None):
        best_val_nll = float('inf')
        best_params = None
        p_search = 20
        K_search = 3000
        batch_size = 256

        param_combos = []
        if gamma2_candidates is not None:
            for dt in dt_candidates:
                for lam in lambda_candidates:
                    for g2 in gamma2_candidates:
                        param_combos.append((dt, lam, g2))
        else:
            for dt in dt_candidates:
                for lam in lambda_candidates:
                    param_combos.append((dt, lam, None))

        for dt, lam, g2 in tqdm(param_combos, desc=f"Searching {mode}"):
            params = {
                'gamma2': g2 if g2 is not None else gamma2_init,
                'lambda_': lam,
                'sigma2': sigma2_init,
                'dt': dt,
                'p': p_search,
                'batch_size': batch_size,
                'eval_every': 50,
            }
            torch.manual_seed(0)
            theta_init = torch.randn(p_search, d, device=device)
            batch_indices = torch.randint(0, n_train, (K_search, batch_size), device=device)
            _, nll_series, _, _, _, _, _ = run_experiment_extended(
                mode=mode, X_train=X_train, y_train=y_train,
                X_eval=X_val, y_eval=y_val,
                params=params, K=K_search, target_type='pro',
                theta_init=theta_init, batch_indices=batch_indices,
                record_particles=False,
            )
            avg_nll = nll_series[-20:].mean().item()
            tqdm.write(f"  {mode}: dt={dt}, lam={lam:.3f}, gamma2={params['gamma2']:.3f} -> val NLL={avg_nll:.4f}")
            if avg_nll < best_val_nll:
                best_val_nll = avg_nll
                best_params = params.copy()
        return best_params, best_val_nll

    # ---------- Helper: search hyperparameters for Bayes MFLD ----------
    def search_hyperparams_bayes(dt_candidates, prior_precision_candidates):
        best_val_nll = float('inf')
        best_params = None
        p_search = 20
        K_search = 3000
        batch_size = 256

        param_combos = list(itertools.product(dt_candidates, prior_precision_candidates))

        for dt, prior_prec in tqdm(param_combos, desc="Searching Bayes MFLD"):
            params = {
                'sigma2': sigma2_init,
                'prior_precision': prior_prec,
                'dt': dt,
                'p': p_search,
                'batch_size': batch_size,
                'eval_every': 50,
                'gamma2': gamma2_init,  # needed for MMD evaluation
                'lambda_': prior_prec,   # placeholder, not actually used for Bayes
            }
            torch.manual_seed(0)
            theta_init = torch.randn(p_search, d, device=device)
            batch_indices = torch.randint(0, n_train, (K_search, batch_size), device=device)
            _, nll_series, _, _, _, _, _ = run_experiment_extended(
                mode='pro', X_train=X_train, y_train=y_train,
                X_eval=X_val, y_eval=y_val,
                params=params, K=K_search, target_type='bayes',
                theta_init=theta_init, batch_indices=batch_indices,
                record_particles=False,
            )
            avg_nll = nll_series[-20:].mean().item()
            tqdm.write(f"  Bayes MFLD: dt={dt}, prior_prec={prior_prec:.3f} -> val NLL={avg_nll:.4f}")
            if avg_nll < best_val_nll:
                best_val_nll = avg_nll
                best_params = params.copy()
        return best_params, best_val_nll

    # ---------- Define search grids ----------
    dt_range = [0.0003, 0.0005, 0.001, 0.002, 0.004]
    lambda_range = [0.3 * base_lambda, 0.5 * base_lambda, base_lambda, 1.5 * base_lambda, 2.0 * base_lambda]
    gamma2_range = [0.5 * gamma2_init, gamma2_init, 2.0 * gamma2_init]
    prior_prec_range = [0.1, 0.5, 1.0, 5.0, 10.0]  # separate prior precision search

    print("\n--- Searching hyperparameters for PrO (MFLD) ---")
    pro_best_params, pro_best_nll = search_hyperparams_prO('pro', dt_range, lambda_range, gamma2_range)
    print(f"PrO best validation NLL: {pro_best_nll:.4f}, parameters: dt={pro_best_params['dt']}, lambda={pro_best_params['lambda_']:.3f}, gamma2={pro_best_params['gamma2']:.3f}")

    print("\n--- Searching hyperparameters for SMC-WFR ---")
    wfr_best_params, wfr_best_nll = search_hyperparams_prO('wfr_smc', dt_range, lambda_range, gamma2_range)
    print(f"SMC-WFR best validation NLL: {wfr_best_nll:.4f}, parameters: dt={wfr_best_params['dt']}, lambda={wfr_best_params['lambda_']:.3f}, gamma2={wfr_best_params['gamma2']:.3f}")

    print("\n--- Searching hyperparameters for Bayes MFLD ---")
    bayes_best_params, bayes_best_nll = search_hyperparams_bayes(dt_range, prior_prec_range)
    print(f"Bayes MFLD best validation NLL: {bayes_best_nll:.4f}, parameters: dt={bayes_best_params['dt']}, prior_prec={bayes_best_params['prior_precision']:.3f}")

    # ---------- Final experiment with optimal parameters ----------
    p_final = 20
    K_final = 1000
    num_runs = 5
    seed = 42

    pro_params = pro_best_params.copy()
    pro_params['p'] = p_final
    pro_params['batch_size'] = 256

    wfr_params = wfr_best_params.copy()
    wfr_params['p'] = p_final
    wfr_params['batch_size'] = 256

    bayes_params = bayes_best_params.copy()
    bayes_params['p'] = p_final
    bayes_params['batch_size'] = 256
    bayes_params['gamma2'] = gamma2_init  # for MMD evaluation
    bayes_params['lambda_'] = bayes_params['prior_precision']  # placeholder

    # Exact Bayesian posterior (using Bayes MFLD's optimal prior_precision)
    mu_post, Sigma_post = compute_bayes_posterior_linear(
        X_train, y_train, sigma2_init, prior_precision=bayes_best_params['prior_precision']
    )
    try:
        mvn = torch.distributions.MultivariateNormal(mu_post.squeeze(), Sigma_post)
        theta_exact = mvn.sample((p_final,))
    except:
        U, S, V = torch.linalg.svd(Sigma_post)
        S_safe = S.clamp_min(1e-6)
        L = U @ torch.diag(torch.sqrt(S_safe))
        z = torch.randn(p_final, d, device=device)
        theta_exact = mu_post.squeeze() + (L @ z.T).T
    w_exact = torch.ones(p_final, device=device) / p_final
    nll_exact = compute_predictive_nll(theta_exact, w_exact, X_test, y_test, sigma2_init)
    mmd_exact = compute_predictive_mmd(theta_exact, w_exact, X_test, y_test, gamma2_init, sigma2_init)
    print(f"Exact Bayes posterior predictive NLL = {nll_exact:.4f}, MMD = {mmd_exact:.4f}")

    pro_nll_list, pro_mmd_list, pro_norm_list, pro_grad_norm_list = [], [], [], []
    wfr_nll_list, wfr_mmd_list, wfr_norm_list, wfr_grad_norm_list = [], [], [], []
    bayes_nll_list, bayes_mmd_list, bayes_norm_list, bayes_grad_norm_list = [], [], [], []

    for run in tqdm(range(num_runs), desc="Running experiments"):
        torch.manual_seed(seed + run)
        theta_init = torch.randn(p_final, d, device=device)
        batch_indices = torch.randint(0, n_train, (K_final, pro_params['batch_size']), device=device)

        steps, nll_p, mmd_p, _, _, norm_p, grad_norm_p = run_experiment_extended(
            mode='pro', X_train=X_train, y_train=y_train,
            X_eval=X_test, y_eval=y_test,
            params=pro_params, K=K_final, target_type='pro',
            theta_init=theta_init, batch_indices=batch_indices,
            record_particle_norm=True, record_grad_norm=True, record_every=50,
        )
        pro_nll_list.append(nll_p); pro_mmd_list.append(mmd_p)
        pro_norm_list.append(norm_p); pro_grad_norm_list.append(grad_norm_p)

        steps, nll_w, mmd_w, _, _, norm_w, grad_norm_w = run_experiment_extended(
            mode='wfr_smc', X_train=X_train, y_train=y_train,
            X_eval=X_test, y_eval=y_test,
            params=wfr_params, K=K_final, target_type='pro',
            theta_init=theta_init, batch_indices=batch_indices,
            record_particle_norm=True, record_grad_norm=True, record_every=50,
        )
        wfr_nll_list.append(nll_w); wfr_mmd_list.append(mmd_w)
        wfr_norm_list.append(norm_w); wfr_grad_norm_list.append(grad_norm_w)

        steps, nll_b, mmd_b, _, _, norm_b, grad_norm_b = run_experiment_extended(
            mode='pro', X_train=X_train, y_train=y_train,
            X_eval=X_test, y_eval=y_test,
            params=bayes_params, K=K_final, target_type='bayes',
            theta_init=theta_init, batch_indices=batch_indices,
            record_particle_norm=True, record_grad_norm=True, record_every=50,
        )
        bayes_nll_list.append(nll_b); bayes_mmd_list.append(mmd_b)
        bayes_norm_list.append(norm_b); bayes_grad_norm_list.append(grad_norm_b)

    # ---------- Plotting (show only first 1500 steps, 4 subplots) ----------
    steps_np = steps.numpy()
    mask = steps_np <= 1500
    x_plot = steps_np[mask]

    pro_nll_mean = torch.stack(pro_nll_list, dim=0).mean(dim=0).numpy()[mask]
    pro_nll_std  = torch.stack(pro_nll_list, dim=0).std(dim=0).numpy()[mask]
    wfr_nll_mean = torch.stack(wfr_nll_list, dim=0).mean(dim=0).numpy()[mask]
    wfr_nll_std  = torch.stack(wfr_nll_list, dim=0).std(dim=0).numpy()[mask]
    bayes_nll_mean = torch.stack(bayes_nll_list, dim=0).mean(dim=0).numpy()[mask]
    bayes_nll_std  = torch.stack(bayes_nll_list, dim=0).std(dim=0).numpy()[mask]

    pro_mmd_mean = torch.stack(pro_mmd_list, dim=0).mean(dim=0).numpy()[mask]
    pro_mmd_std  = torch.stack(pro_mmd_list, dim=0).std(dim=0).numpy()[mask]
    wfr_mmd_mean = torch.stack(wfr_mmd_list, dim=0).mean(dim=0).numpy()[mask]
    wfr_mmd_std  = torch.stack(wfr_mmd_list, dim=0).std(dim=0).numpy()[mask]
    bayes_mmd_mean = torch.stack(bayes_mmd_list, dim=0).mean(dim=0).numpy()[mask]
    bayes_mmd_std  = torch.stack(bayes_mmd_list, dim=0).std(dim=0).numpy()[mask]

    # Take particle norm and gradient norm from the last run
    pro_norm = pro_norm_list[-1]
    wfr_norm = wfr_norm_list[-1]
    bayes_norm = bayes_norm_list[-1]
    pro_grad_norm = pro_grad_norm_list[-1]
    wfr_grad_norm = wfr_grad_norm_list[-1]
    bayes_grad_norm = bayes_grad_norm_list[-1]

    record_x = np.arange(len(pro_norm))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Subplot 1: NLL
    axes[0,0].plot(x_plot, bayes_nll_mean, label=ALGORITHM_LABELS['bayes'],
                 color=ALGORITHM_COLORS['bayes'], linestyle=ALGORITHM_LINESTYLES['bayes'])
    axes[0,0].fill_between(x_plot, bayes_nll_mean - bayes_nll_std, bayes_nll_mean + bayes_nll_std,
                         alpha=0.2, color=ALGORITHM_COLORS['bayes'])
    axes[0,0].plot(x_plot, pro_nll_mean, label=ALGORITHM_LABELS['pro'],
                 color=ALGORITHM_COLORS['pro'], linestyle=ALGORITHM_LINESTYLES['pro'])
    axes[0,0].fill_between(x_plot, pro_nll_mean - pro_nll_std, pro_nll_mean + pro_nll_std,
                         alpha=0.2, color=ALGORITHM_COLORS['pro'])
    axes[0,0].plot(x_plot, wfr_nll_mean, label=ALGORITHM_LABELS['wfr_smc'],
                 color=ALGORITHM_COLORS['wfr_smc'], linestyle=ALGORITHM_LINESTYLES['wfr_smc'])
    axes[0,0].fill_between(x_plot, wfr_nll_mean - wfr_nll_std, wfr_nll_mean + wfr_nll_std,
                         alpha=0.2, color=ALGORITHM_COLORS['wfr_smc'])
    axes[0,0].axhline(y=nll_exact, color='k', linestyle='--', label='Bayes exact')
    axes[0,0].set_xlabel('Iteration')
    axes[0,0].set_ylabel('Test NLL')
    axes[0,0].set_title('Predictive NLL (first 1500 iters)')
    axes[0,0].legend()
    axes[0,0].grid(alpha=0.3)

    # Subplot 2: MMD
    axes[0,1].plot(x_plot, bayes_mmd_mean, label=ALGORITHM_LABELS['bayes'],
                 color=ALGORITHM_COLORS['bayes'], linestyle=ALGORITHM_LINESTYLES['bayes'])
    axes[0,1].fill_between(x_plot, bayes_mmd_mean - bayes_mmd_std, bayes_mmd_mean + bayes_mmd_std,
                         alpha=0.2, color=ALGORITHM_COLORS['bayes'])
    axes[0,1].plot(x_plot, pro_mmd_mean, label=ALGORITHM_LABELS['pro'],
                 color=ALGORITHM_COLORS['pro'], linestyle=ALGORITHM_LINESTYLES['pro'])
    axes[0,1].fill_between(x_plot, pro_mmd_mean - pro_mmd_std, pro_mmd_mean + pro_mmd_std,
                         alpha=0.2, color=ALGORITHM_COLORS['pro'])
    axes[0,1].plot(x_plot, wfr_mmd_mean, label=ALGORITHM_LABELS['wfr_smc'],
                 color=ALGORITHM_COLORS['wfr_smc'], linestyle=ALGORITHM_LINESTYLES['wfr_smc'])
    axes[0,1].fill_between(x_plot, wfr_mmd_mean - wfr_mmd_std, wfr_mmd_mean + wfr_mmd_std,
                         alpha=0.2, color=ALGORITHM_COLORS['wfr_smc'])
    axes[0,1].axhline(y=mmd_exact, color='k', linestyle='--', label='Bayes exact')
    axes[0,1].set_xlabel('Iteration')
    axes[0,1].set_ylabel('Test MMD^2')
    axes[0,1].set_title('MMD (first 1500 iters)')
    axes[0,1].legend()
    axes[0,1].grid(alpha=0.3)

    # Subplot 3: Particle norm (stability)
    axes[1,0].plot(record_x, bayes_norm, label=ALGORITHM_LABELS['bayes'],
                 color=ALGORITHM_COLORS['bayes'], linestyle=ALGORITHM_LINESTYLES['bayes'])
    axes[1,0].plot(record_x, pro_norm, label=ALGORITHM_LABELS['pro'],
                 color=ALGORITHM_COLORS['pro'], linestyle=ALGORITHM_LINESTYLES['pro'])
    axes[1,0].plot(record_x, wfr_norm, label=ALGORITHM_LABELS['wfr_smc'],
                 color=ALGORITHM_COLORS['wfr_smc'], linestyle=ALGORITHM_LINESTYLES['wfr_smc'])
    axes[1,0].set_xlabel('Recorded iteration')
    axes[1,0].set_ylabel('Average particle norm')
    axes[1,0].set_title('Particle norm evolution (stability)')
    axes[1,0].legend()
    axes[1,0].grid(alpha=0.3)

    # Subplot 4: Gradient norm (convergence)
    axes[1,1].plot(record_x, bayes_grad_norm, label=ALGORITHM_LABELS['bayes'],
                 color=ALGORITHM_COLORS['bayes'], linestyle=ALGORITHM_LINESTYLES['bayes'])
    axes[1,1].plot(record_x, pro_grad_norm, label=ALGORITHM_LABELS['pro'],
                 color=ALGORITHM_COLORS['pro'], linestyle=ALGORITHM_LINESTYLES['pro'])
    axes[1,1].plot(record_x, wfr_grad_norm, label=ALGORITHM_LABELS['wfr_smc'],
                 color=ALGORITHM_COLORS['wfr_smc'], linestyle=ALGORITHM_LINESTYLES['wfr_smc'])
    axes[1,1].set_xlabel('Recorded iteration')
    axes[1,1].set_ylabel('Average gradient norm')
    axes[1,1].set_title('Gradient norm evolution (convergence)')
    axes[1,1].legend()
    axes[1,1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('exp3_real_dataset_improved_final.png', dpi=150)
    plt.show()
    print("Experiment 3 (final improved version) finished, figure saved as exp3_real_dataset_improved_final.png")

# ==============================================================================
# Main entry point
# ==============================================================================
if __name__ == "__main__":
    experiment1_prO_vs_bayes()
    experiment2_algorithm_comparison_synthetic()
    experiment3_real_dataset()
    print("\nAll extended experiments completed.")