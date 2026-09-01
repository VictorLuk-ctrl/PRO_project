# import torch
# import math
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import fetch_california_housing
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import itertools  # for generating full grid


# # ==============================================================================
# # SUMMARY
# # ==============================================================================
# # This script compares particle algorithms for the entropy-regularised nonlinear
# # variational problem
# #
# #     F(Q) = lambda_n * S_MMD(P_Q) + KL(Q || Pi),
# #
# # where S_MMD(P_Q) is the empirical conditional squared-MMD score, Pi is the
# # N(0, I) parameter prior, and P_Q is the posterior predictive mixture. If
# # phi_Q(theta) denotes the first variation of the MMD score, then, up to
# # an additive constant,
# #
# #     delta F / delta Q(theta)
# #       = lambda_n * phi_Q(theta) + log q(theta) - log pi(theta).
# #
# # The continuum Wasserstein, Fisher--Rao and Wasserstein--Fisher--Rao equations
# # are standard for a sufficiently regular functional F.  
# #
# # PRIMARY REFERENCES
# #
# # [PRO]
# #   McLatchie, Y.; Cherief-Abdellatif, B.-E.; Frazier, D. T.;
# #   Knoblauch, J. (2025), "Predictively Oriented Posteriors",
# #   arXiv:2510.01915, especially Section 6 and Appendix E. The paper computes
# #   PrO posteriors using mean-field Langevin dynamics and gives the MMD
# #   particle force used here (Appendix E.1).
# #   https://arxiv.org/abs/2510.01915
# #
# # [MFLD-HRSS]
# #   Hu, K.; Ren, Z.; Siska, D.; Szpruch, L. (2021),
# #   "Mean-field Langevin dynamics and energy landscape of neural networks",
# #   Annales de l'Institut Henri Poincare, Probabilites et Statistiques 57(4).
# #   General MFLD for nonlinear entropy-regularised functionals on measures.
# #   Preprint: arXiv:1905.07769.
# #   https://arxiv.org/abs/1905.07769
# #
# # [MFLD-CHIZAT]
# #   Chizat, L. (2022), "Mean-Field Langevin Dynamics: Exponential Convergence
# #   and Annealing", Transactions on Machine Learning Research; arXiv:2202.01009.
# #   Noisy particle gradient descent for convex objectives over measures with an
# #   entropy term, and its mean-field limit.
# #   https://arxiv.org/abs/2202.01009
# #
# # [VGD]
# #   Chazal, C.; Kanagawa, H.; Shen, Z.; Korba, A.; Oates, C. J. (2026),
# #   "A Computable Measure of Suboptimality for Entropy-Regularised
# #   Variational Objectives", arXiv:2509.10393, Section 3.2.2.2, equation (19).
# #   Introduces variational gradient descent (VGD), a nonlinear extension of
# #   Stein variational gradient descent for objectives L(Q)+KL(Q||Q0). VGD
# #   transports an equally weighted particle set using an RKHS-smoothed version
# #   of the variational gradient and a kernel repulsion term.
# #   https://arxiv.org/abs/2509.10393
# #
# # [NULA]
# #   Fu, Q.; Wilson, A. C. (2024), "Mean-field Underdamped Langevin Dynamics
# #   and its Spacetime Discretization", Proceedings of the 41st International
# #   Conference on Machine Learning, PMLR 235:14175--14206; arXiv:2312.16360.
# #   Algorithm 1 gives the N-particle underdamped Langevin algorithm (NULA),
# #   obtained by exactly integrating the frozen-force kinetic Ornstein--Uhlenbeck
# #   dynamics over each time step, including the correlated position/velocity
# #   Gaussian increment used below.
# #   https://proceedings.mlr.press/v235/fu24g.html
# #
# # [FR-MIRROR]
# #   Yao, R.; Huang, L.; Yang, Y. (2024), "Minimizing Convex Functionals over
# #   Space of Probability Measures via KL Divergence Gradient Flow",
# #   Proceedings of AISTATS 2024, PMLR 238; arXiv:2311.00894.
# #   Treats the Fisher--Rao/KL gradient flow and mirror/proximal discretisations
# #   for general convex functionals on probability measures.
# #   https://arxiv.org/abs/2311.00894
# #
# # [FR-TEMPERING]
# #   Chopin, N.; Crucinio, F. R.; Korba, A. (2024), "A connection between
# #   Tempering and Entropic Mirror Descent", ICML 2024, PMLR 235.
# #   For a fixed target pi, the exact FR trajectory is the geometric interpolation
# #   q_t proportional to q_0^{exp(-t)} pi^{1-exp(-t)}.
# #   https://proceedings.mlr.press/v235/chopin24a.html
# #
# # [FR-KERNEL-MEANFIELD]
# #   Lazic, P.; Liu, L.; Majka, M. B. (2026), "On propagation of chaos for the
# #   Fisher-Rao gradient flow in entropic mean-field optimization",
# #   arXiv:2602.15094. Gives a rigorous kernelised interacting-particle
# #   approximation of FR flow for nonlinear entropic mean-field objectives.
# #   https://arxiv.org/abs/2602.15094
# #
# # [BDL-2019]
# #   Lu, Y.; Lu, J.; Nolen, J. (2019), "Accelerating Langevin Sampling with
# #   Birth-death", arXiv:1905.09863. Introduces birth--death Langevin as a
# #   particle approximation of the WFR gradient flow of reverse KL.
# #   https://arxiv.org/abs/1905.09863
# #
# # [BDL-2023]
# #   Lu, Y.; Slepcev, D.; Wang, L. (2023), "Birth-death dynamics for sampling:
# #   Global convergence, approximations and their asymptotics", Nonlinearity 36,
# #   5731--5772; arXiv:2211.00450. Refines the continuum and particle analysis.
# #   https://arxiv.org/abs/2211.00450
# #
# # [SMC-WFR]
# #   Crucinio, F. R.; Pathiraja, S. (2025/2026), "Sequential Monte Carlo
# #   approximations of Wasserstein--Fisher--Rao gradient flows",
# #   arXiv:2506.05905, especially Algorithm 1 and equation (3.5).
# #   Gives the ULA + exact-FR reweighting + resampling construction used below
# #   for a fixed reverse-KL target.
# #   https://arxiv.org/abs/2506.05905
# #
# # [WFR-NONLINEAR]
# #   Yan, Y.; Wang, K.; Rigollet, P. (2024), "Learning Gaussian Mixtures Using
# #   the Wasserstein--Fisher--Rao Gradient Flow", Annals of Statistics 52(4),
# #   1774--1795; arXiv:2301.01766. A standard weighted-particle WFR method for a
# #   nonlinear functional (the mixture negative log-likelihood), but without the
# #   continuous KL-to-prior term used in the PrO objective here.
# #   https://arxiv.org/abs/2301.01766
# #
# # NOTES
# # ----------------------------
# # * train_step_pro is a standard MFLD discretisation of the *nonlinear PrO
# #   functional itself*.
# # * train_step_vgd implements equation (19) of [VGD] with a multiscale
# #   inverse-multiquadric parameter-space kernel. It is deterministic, keeps
# #   equal particle weights, and uses kernel repulsion instead of Brownian noise
# #   to represent the entropy contribution.
# # * train_step_underdamped_pro implements Algorithm 1 of [NULA] for the same
# #   smooth PrO energy used by MFLD. It augments every parameter particle with a
# #   velocity and uses the paper's correlated Gaussian spacetime increment.
# # * train_step_fr_mirror is a standard explicit first-variation / entropic
# #   mirror step on a fixed particle support. It is a legitimate nonlinear
# #   mean-field discretisation, but it is only a quadrature approximation of the
# #   continuous problem and cannot move its support.
# # * train_step_wfr_smc and train_step_wfr_bdl use standard fixed-target WFR
# #   subroutines after replacing the fixed target by the instantaneous
# #   frozen Gibbs target pi_Q proportional to Pi exp(-lambda phi_Q).
# #   This is a natural extension to the PrO setting, though note the fixed-target
# #   convergence theorems in [SMC-WFR], [BDL-2019] and [BDL-2023] do not
# #   automatically apply to this self-consistent PrO implementation.
# # * the KDE-WF/FR/WFR routines are direct KDE discretisations of the
# #   formal continuum equations. 
# # ==============================================================================

# # ==================== Unified algorithm style mapping ====================
# # Used across all plots in this script (and compatible with other scripts)
# ALGORITHM_LABELS = {
#     'pro': 'PrO (MFLD)',
#     'underdamped_pro': 'PrO (NULA)',
#     'vgd': 'VGD',
#     'fr_mirror': 'FR Mirror',
#     'wfr_smc': 'SMC-WFR',
#     'wfr_bdl': 'BDL-WFR',
#     'wfr': 'KDE-WFR',
#     'fr': 'KDE-FR',
#     'bayes': 'Bayes',
# }
# ALGORITHM_COLORS = {
#     'pro': '#e41a1c',          # red
#     'underdamped_pro': '#ff7f00',  # orange
#     'vgd': '#4daf4a',          # green
#     'fr_mirror': '#984ea3',    # purple
#     'wfr_smc': '#377eb8',      # blue
#     'wfr_bdl': '#f781bf',      # pink
#     'wfr': '#a65628',          # brown
#     'fr': '#999999',           # grey
#     'bayes': '#1f78b4',        # dark blue
# }
# ALGORITHM_LINESTYLES = {
#     'pro': '-',
#     'underdamped_pro': '--',
#     'vgd': '-.',
#     'fr_mirror': ':',
#     'wfr_smc': '-',
#     'wfr_bdl': '--',
#     'wfr': '-.',
#     'fr': ':',
#     'bayes': '-',
# }
# # ========================================================================

# # ==================== 1. Device selection and configuration ====================
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

# # ==================== 2. Load and preprocess real dataset ====================
# def load_real_data():
#     """Load California housing data and create leakage-free data splits.

#     Returns six tensors: X_train, y_train, X_val, y_val, X_test and y_test.
#     Feature tensors have shape [number of observations, d]; response tensors
#     have shape [number of observations, 1]. All tensors are placed directly on
#     the selected CPU/GPU device.

#     StandardScaler parameters are estimated using training observations only.
#     """
#     data = fetch_california_housing()
#     X, y = data.data, data.target

#     X_train_full, X_test, y_train_full, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_train_full, y_train_full, test_size=0.2, random_state=43
#     )

#     scaler_X = StandardScaler()
#     scaler_y = StandardScaler()

#     X_train = scaler_X.fit_transform(X_train)
#     X_val = scaler_X.transform(X_val)
#     X_test = scaler_X.transform(X_test)

#     y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
#     y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
#     y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

#     return (
#         torch.tensor(X_train, dtype=torch.float32, device=device),
#         torch.tensor(y_train, dtype=torch.float32, device=device).view(-1, 1),
#         torch.tensor(X_val, dtype=torch.float32, device=device),
#         torch.tensor(y_val, dtype=torch.float32, device=device).view(-1, 1),
#         torch.tensor(X_test, dtype=torch.float32, device=device),
#         torch.tensor(y_test, dtype=torch.float32, device=device).view(-1, 1),
#     )


# X_train, y_train, X_val, y_val, X_test, y_test = load_real_data()
# n_features = X_train.shape[1]  # California housing has 8 features
# d = n_features
# print(f"Data dimension (features) d = {d}")


# # ==================== 3. Linear regression model and prior definitions ====================
# def prior_log_prob(theta):
#     """Evaluate log pi(theta) for the N(0, I_d) prior, up to a constant.

#     Parameters
#     ----------
#     theta : tensor, shape [p, d] or [m, d]
#         Parameter vectors at which the prior is evaluated.

#     Returns
#     -------
#     tensor, shape [p] or [m]
#         The value -||theta||^2/2 for each row.

#     The omitted normalising constant is -d*log(2*pi)/2. It is irrelevant here:
#     gradients remove constants, and softmax-normalised weights are unchanged if
#     the same constant is added to every log weight.
#     """
#     return -0.5 * torch.sum(theta ** 2, dim=-1)


# def prior_grad_log(theta):
#     """Return grad_theta log pi(theta) for a standard Gaussian prior.

#     Since log pi(theta) = constant - ||theta||^2/2, its gradient is -theta.
#     This term pulls transported particles back toward the prior centre.
#     """
#     return -theta


# def prior_sample(key, p):
#     """Draw p independent initial particles from Pi=N(0,I_d).

#     PyTorch randomness is controlled globally with torch.manual_seed 
#     before this function is called.
#     """
#     return torch.randn(p, d, device=device)


# def estimate_noise_variance(X, y):
#     """Estimate the fixed Gaussian observation variance from OLS residuals.

#     The algorithms require a known component variance sigma^2 in

#         Y | X=x, theta ~ N(x^T theta, sigma^2).

#     Rather than optimizing sigma^2 jointly with Q, this example obtains a simple
#     plug-in estimate from the mean squared training residual of ordinary least
#     squares. The clamp prevents a degenerate zero variance, which would make log
#     densities and score gradients undefined.

#     """
#     theta_ols = torch.linalg.lstsq(X, y, rcond=None)[0]
#     residuals = y - X @ theta_ols
#     sigma2 = torch.mean(residuals ** 2).clamp_min(1e-4)
#     return float(sigma2.item()), theta_ols.flatten()


# def median_heuristic_gamma2(y, max_points=2000):
#     """Choose the response-space RBF bandwidth by the median heuristic.

#     The MMD kernel is k(a,b)=exp(-(a-b)^2/(2*gamma^2)). This function sets
#     gamma^2 to the median non-zero squared distance between observed responses.
#     It subsamples at most ``max_points`` responses because forming every pair is
#     quadratic in the number of observations.

#     """
#     y_flat = y.flatten()
#     if y_flat.numel() > max_points:
#         generator = torch.Generator(device=device)
#         generator.manual_seed(20260729)
#         idx = torch.randperm(
#             y_flat.numel(), generator=generator, device=device
#         )[:max_points]
#         y_flat = y_flat[idx]

#     D2 = (y_flat[:, None] - y_flat[None, :]) ** 2
#     upper = D2[torch.triu_indices(D2.shape[0], D2.shape[1], offset=1, device=device).unbind()]
#     upper = upper[upper > 0]
#     if upper.numel() == 0:
#         return 1.0
#     return float(torch.median(upper).item())


# sigma2_hat, theta_ols = estimate_noise_variance(X_train, y_train)
# gamma2_hat = median_heuristic_gamma2(y_train)
# print(f"Estimated observation variance sigma^2 = {sigma2_hat:.4f}")
# print(f"MMD kernel bandwidth gamma^2 = {gamma2_hat:.4f}")


# # ==================== 4. Squared-MMD first variation ====================
# def compute_mmd_potential_and_grad(
#     theta,
#     w,
#     X_batch,
#     y_batch,
#     gamma2,
#     sigma2,
#     leave_one_out=False,
# ):
#     """Compute the MMD score first variation at all current particles.

#     Mathematical target
#     -------------------
#     On a minibatch of B input-response pairs, the score is

#         S_MMD(Q) = (1/B) sum_b MMD^2(P_Q(.|x_b), delta_{y_b}),

#     with Gaussian RBF kernel k(u,v)=exp(-(u-v)^2/(2*gamma^2)). Expanding MMD,

#         MMD^2(P_Q,delta_y)
#           = E[k(Y,Y')] - 2 E[k(Y,y)] + k(y,y),

#     where Y and Y' are independent draws from P_Q. The final term equals one
#     and has zero derivative, so it is absent from phi and grad_phi.

#     Why the Gaussian expectations are analytic
#     -------------------------------------------
#     If U~N(mu_1,s_1^2) and V~N(mu_2,s_2^2), then

#         E exp(-(U-V)^2/(2*gamma^2))
#           = sqrt(gamma^2/(gamma^2+s_1^2+s_2^2))
#             * exp(-(mu_1-mu_2)^2
#                   /(2*(gamma^2+s_1^2+s_2^2))).

#     Therefore pairwise predictive components use variance gamma^2+2*sigma^2,
#     while a predictive component compared with fixed y uses gamma^2+sigma^2.

#     Particle first variation
#     ------------------------
#     For Q=sum_l w_l delta_{theta_l}, the first variation at theta_j is

#         phi_j = 2 * average_b [sum_l w_l K_{j,l}(x_b) - K_{j,y_b}(x_b)].

#     ``grad_phi[j]`` differentiates only with respect to the evaluation particle
#     theta_j. This is the velocity information needed by WF/MFLD algorithms.

#     Leave-one-out option
#     --------------------
#     Mean-field particle systems commonly exclude particle j from the empirical
#     approximation used to construct its own interaction field. With
#     ``leave_one_out=True``, row j of the interaction matrix has zero mass at j
#     and is renormalized over the other p-1 particles. This removes a finite-p
#     self-interaction term; it becomes immaterial as p tends to infinity.

#     Shape guide
#     -----------
#     theta       [p,d]
#     pred_mean   [B,p]
#     diff_pair   [B,p,p]
#     K_pair      [B,p,p]
#     K_data      [B,p]
#     phi         [p]
#     grad_phi    [p,d]

#     Returns
#     -------
#     phi : tensor [p]
#         First variation values at the particles.
#     grad_phi : tensor [p,d]
#         Spatial gradient with respect to each corresponding particle location.
#     """
#     B = X_batch.shape[0]
#     p = theta.shape[0]

#     if gamma2 <= 0 or sigma2 <= 0:
#         raise ValueError("gamma2 and sigma2 must both be strictly positive.")
#     if leave_one_out and p < 2:
#         raise ValueError("leave_one_out=True requires at least two particles.")

#     # Matrix multiplication evaluates every particle on every input:
#     # entry (b,j) is x_b^T theta_j.
#     pred_mean = X_batch @ theta.T  # [B, p]

#     # E[k(Y_j, Y_l)] for two independent Gaussian predictive draws. The
#     # observation noise variances add, hence gamma^2 + 2 sigma^2.
#     pair_variance = gamma2 + 2.0 * sigma2
#     pair_scale = math.sqrt(gamma2 / pair_variance)
#     diff_pair = pred_mean[:, :, None] - pred_mean[:, None, :]  # [B, p, p]
#     K_pair = pair_scale * torch.exp(-(diff_pair ** 2) / (2.0 * pair_variance))

#     # E[k(Y_j, y_b)] for one Gaussian predictive draw and a fixed observation.
#     # Only one predictive variance is present, hence gamma^2 + sigma^2.
#     data_variance = gamma2 + sigma2
#     data_scale = math.sqrt(gamma2 / data_variance)
#     diff_data = pred_mean - y_batch  # [B, p]
#     K_data = data_scale * torch.exp(-(diff_data ** 2) / (2.0 * data_variance))

#     # Interaction measure used to approximate the integral over Q.
#     # Row j contains the weights used when integrating the interaction field
#     # seen by evaluation particle j. Normally all rows equal w.
#     interaction_weights = w[None, :].expand(p, p).clone()
#     if leave_one_out:
#         interaction_weights.fill_diagonal_(0.0)
#         interaction_weights = interaction_weights / interaction_weights.sum(dim=1, keepdim=True)

#     interaction_term = torch.sum(
#         K_pair * interaction_weights[None, :, :], dim=2
#     )  # [B, p]

#     # Exact first variation of the displayed MMD^2 objective.
#     phi = 2.0 * (interaction_term.mean(dim=0) - K_data.mean(dim=0))

#     # Derivatives with respect to the first predictive mean in each pair.
#     dK_pair_dmean = -(diff_pair / pair_variance) * K_pair
#     dK_data_dmean = -(diff_data / data_variance) * K_data

#     grad_mean = 2.0 * (
#         torch.sum(
#             dK_pair_dmean * interaction_weights[None, :, :], dim=2
#         )
#         - dK_data_dmean
#     )  # [B, p]

#     grad_phi = (grad_mean.T @ X_batch) / B  # [p, d]
#     return phi, grad_phi


# # ==================== 5. Continuous-density approximation for WF/FR/WFR ====================
# def compute_kde_log_density_and_score(theta, w, bandwidth):
#     """Approximate log q and grad log q at the particle locations by KDE.

#     The smoothed density is

#         q_hat(x) = sum_j w_j N(x; theta_j, h^2 I_d),

#     where ``bandwidth`` is h. For each evaluation location theta_i, the code
#     first computes the log contribution of every kernel centre theta_j. A
#     logsumexp gives log q_hat(theta_i). Softmax-normalized kernel contributions
#     are the responsibilities

#         r_{ij} = w_j K_h(theta_i-theta_j) / q_hat(theta_i).

#     Differentiating the KDE gives

#         grad log q_hat(theta_i)
#           = -sum_j r_{ij}(theta_i-theta_j)/h^2.

#     Why this is needed
#     ------------------
#     The entropy part of KL(Q||Pi) contributes log q to the FR potential and
#     grad log q to the WF velocity. Replacing log q by log w_j would describe a
#     different discrete optimization problem and would not approximate the
#     continuous KL to a Gaussian prior.

#     Limitations
#     -----------
#     KDE quality deteriorates in high dimension and depends materially on h. A
#     small h creates sharp, high-variance forces; a large h over-smooths Q. The
#     implementation includes self-kernels, which stabilize the density at each
#     particle but also affect the finite-particle bias.
#     """
#     if bandwidth <= 0:
#         raise ValueError("KDE bandwidth must be strictly positive.")

#     h2 = bandwidth ** 2
#     parameter_dim = theta.shape[1]
#     diff = theta[:, None, :] - theta[None, :, :]  # evaluation i minus centre j
#     D2 = torch.sum(diff ** 2, dim=-1)

#     log_kernel = (
#         -0.5 * D2 / h2
#         -0.5 * parameter_dim * math.log(2.0 * math.pi * h2)
#     )
#     log_weighted_kernel = torch.log(w.clamp_min(1e-30))[None, :] + log_kernel

#     # logsumexp computes log of the weighted kernel sum stably. The same
#     # normalized log contributions are posterior responsibilities of KDE centres.
#     log_q = torch.logsumexp(log_weighted_kernel, dim=1)
#     responsibilities = torch.softmax(log_weighted_kernel, dim=1)
#     score_q = -torch.sum(responsibilities[:, :, None] * diff, dim=1) / h2

#     return log_q, score_q


# def update_weights_fisher_rao(w, eta, eta_w):
#     """Advance a discrete centred Fisher--Rao/replicator equation.

#     Suppose eta_j approximates delta F/delta Q(theta_j). The probability-mass
#     ODE on fixed support is

#         d w_j/dt = -w_j (eta_j - eta_bar),
#         eta_bar   = sum_l w_l eta_l.

#     Centering is essential: summing the right-hand side over j gives zero, so
#     total mass is conserved. Particles with eta_j below the weighted mean gain
#     mass because placing probability there lowers F; particles above the mean
#     lose mass.

#     Freezing eta during one step and integrating multiplicatively gives

#         w_j^+ proportional to w_j exp[-eta_w(eta_j-eta_bar)].

#     Taking a softmax of log weights guarantees positivity and normalization.
#     When eta depends on Q, as it does here, this remains an explicit one-step
#     discretisation rather than the exact nonlinear flow map.
#     """
#     eta_bar = torch.sum(w * eta)
#     log_w_new = torch.log(w.clamp_min(1e-30)) - eta_w * (eta - eta_bar)
#     return torch.softmax(log_w_new, dim=0)


# # ==================== 5b. Standard FR/WFR particle approximations ====================
# def compute_mmd_potential_and_grad_at(
#     theta_eval,
#     theta_measure,
#     w_measure,
#     X_batch,
#     y_batch,
#     gamma2,
#     sigma2,
# ):
#     """
#     Evaluate the squared-MMD first variation phi_Q and its gradient at arbitrary
#     parameter values theta_eval, for Q represented by weighted particles
#     (theta_measure, w_measure).

#     This is needed by the SMC-WFR splitting because its Fisher--Rao correction
#     evaluates the frozen self-consistent target at newly proposed particles.
#     """
#     B = X_batch.shape[0]

#     pred_eval = X_batch @ theta_eval.T          # [B, m]
#     pred_measure = X_batch @ theta_measure.T    # [B, p]

#     pair_variance = gamma2 + 2.0 * sigma2
#     pair_scale = math.sqrt(gamma2 / pair_variance)
#     diff_pair = pred_eval[:, :, None] - pred_measure[:, None, :]
#     K_pair = pair_scale * torch.exp(-(diff_pair ** 2) / (2.0 * pair_variance))

#     data_variance = gamma2 + sigma2
#     data_scale = math.sqrt(gamma2 / data_variance)
#     diff_data = pred_eval - y_batch
#     K_data = data_scale * torch.exp(-(diff_data ** 2) / (2.0 * data_variance))

#     interaction_term = torch.sum(K_pair * w_measure[None, None, :], dim=2)
#     phi = 2.0 * (interaction_term.mean(dim=0) - K_data.mean(dim=0))

#     dK_pair_dmean = -(diff_pair / pair_variance) * K_pair
#     dK_data_dmean = -(diff_data / data_variance) * K_data
#     grad_mean = 2.0 * (
#         torch.sum(dK_pair_dmean * w_measure[None, None, :], dim=2)
#         - dK_data_dmean
#     )
#     grad_phi = (grad_mean.T @ X_batch) / B

#     return phi, grad_phi


# def compute_gaussian_mixture_log_density(x, means, w, variance):
#     """Log density of sum_j w_j N(x; means_j, variance I)."""
#     if variance <= 0:
#         raise ValueError("variance must be strictly positive.")

#     parameter_dim = x.shape[1]
#     diff = x[:, None, :] - means[None, :, :]
#     D2 = torch.sum(diff ** 2, dim=-1)
#     log_kernel = (
#         -0.5 * D2 / variance
#         -0.5 * parameter_dim * math.log(2.0 * math.pi * variance)
#     )
#     return torch.logsumexp(
#         torch.log(w.clamp_min(1e-30))[None, :] + log_kernel,
#         dim=1,
#     )


# def multinomial_resample(theta, w):
#     """Standard multinomial resampling."""
#     p = theta.shape[0]
#     ancestors = torch.multinomial(w, p, replacement=True)
#     theta_new = theta[ancestors]
#     w_new = torch.ones(p, device=theta.device) / p
#     return theta_new, w_new


# def systematic_resample(theta, w):
#     """Standard lower-variance systematic SMC resampling."""
#     p = theta.shape[0]
#     start = torch.rand(1, device=theta.device) / p
#     positions = start + torch.arange(p, device=theta.device) / p
#     cumulative = torch.cumsum(w, dim=0)
#     cumulative[-1] = 1.0
#     ancestors = torch.searchsorted(cumulative, positions, right=False)
#     theta_new = theta[ancestors]
#     w_new = torch.ones(p, device=theta.device) / p
#     return theta_new, w_new


# def train_step_fr_mirror(
#     theta,
#     w,
#     X_batch,
#     y_batch,
#     gamma2,
#     sigma2,
#     lambda_,
#     eta_w,
#     base_w,
# ):
#     """
#     Fixed-support Fisher--Rao / entropic-mirror step for the PrO functional.

#     CONTINUUM EQUATION
#     -----------------
#     For F(Q) = lambda*S_MMD(Q) + KL(Q || Pi), the mass-preserving FR flow is

#         partial_t q_t(theta)
#           = -q_t(theta) * [delta F/delta q_t(theta)
#                            - E_{Q_t}[delta F/delta q_t]],

#     with delta F/delta q = lambda*phi_Q + log(q/pi) + constant.
#     This is the standard FR/KL-gradient-flow equation for a general functional;
#     see [FR-MIRROR].

#     DISCRETISATION USED HERE
#     ------------------------
#     The support theta_j is drawn once from the prior and never moves. base_w is
#     its quadrature mass under the prior. At the current Q, freeze phi_Q and form

#         pi_Q(j) proportional to base_w(j) * exp{-lambda*phi_Q(theta_j)}.

#     Conditional on this frozen target, the reverse-KL FR equation has the exact
#     geometric-interpolation solution

#         w_j^+ proportional to
#             w_j^{exp(-eta_w)} * pi_Q(j)^{1-exp(-eta_w)};

#     see [FR-TEMPERING]. Recomputing phi_Q after each step makes this an explicit
#     first-variation / mirror discretisation of the nonlinear PrO problem.

#     NOTES
#     ---------------------
#     This is standard as an entropic mirror/FR step for a general functional, but
#     the finite-support quadrature is only an approximation of the continuous
#     PrO posterior. Pure FR cannot create support where no initial particle was
#     placed. More elaborate kernel interacting-particle FR approximations for
#     nonlinear entropic mean-field problems are studied in
#     [FR-KERNEL-MEANFIELD].
#     """
#     phi, _ = compute_mmd_potential_and_grad(
#         theta,
#         w,
#         X_batch,
#         y_batch,
#         gamma2,
#         sigma2,
#         leave_one_out=False,
#     )

#     decay = math.exp(-eta_w)
#     log_target = torch.log(base_w.clamp_min(1e-30)) - lambda_ * phi
#     log_w_new = decay * torch.log(w.clamp_min(1e-30)) + (1.0 - decay) * log_target
#     return theta.detach(), torch.softmax(log_w_new, dim=0).detach()


# def train_step_wfr_smc(
#     theta,
#     w,
#     X_batch,
#     y_batch,
#     gamma2,
#     sigma2,
#     lambda_,
#     gamma,
# ):
#     """
#     Sequential Monte Carlo (SMC) approximation of a self-consistent WFR splitting for PrO.

#     FIXED-TARGET REFERENCE ALGORITHM
#     --------------------------------
#     For F(mu)=KL(mu || pi), [SMC-WFR, Algorithm 1 and equation (3.5)]
#     alternates:

#       1. a Wasserstein/ULA proposal

#            X^+ = X + gamma * grad log pi(X) + sqrt(2 gamma) * noise;

#       2. an exact time-gamma Fisher--Rao correction with weights

#            w(x) proportional to
#              [pi(x) / mu_{n+1/2}(x)]^{1-exp(-gamma)},

#          where mu_{n+1/2} is the Gaussian-mixture law generated by the ULA
#          proposal;

#       3. resampling to prevent repeated importance-weight degeneracy.

#     NONLINEAR PrO EXTENSION USED HERE
#     ---------------------------------
#     At the current empirical Q, freeze the predictive first variation and form
#     the instantaneous self-consistent Gibbs target

#         log pi_Q(theta)
#           = log pi_0(theta) - lambda*phi_Q(theta) + constant.

#     The MFLD drift is therefore

#         grad log pi_Q = grad log pi_0 - lambda*grad phi_Q.

#     We then use the same Gaussian-mixture proposal denominator and FR exponent
#     1-exp(-gamma) as [SMC-WFR]. Systematic resampling is used instead of
#     multinomial resampling to reduce Monte Carlo variance; both are standard
#     SMC choices.

#     NOTES
#     ------
#     The fixed-target SMC-WFR substep is standard. Replacing pi by pi_Q and
#     refreshing Q after each step is a natural explicit mean-field extension,
#     but it is not Algorithm 1 of [SMC-WFR] verbatim and its fixed-target
#     convergence theorem does not automatically cover this nonlinear PrO case.
#     """
#     theta_old, w_old = systematic_resample(theta, w)

#     _, grad_phi_old = compute_mmd_potential_and_grad(
#         theta_old,
#         w_old,
#         X_batch,
#         y_batch,
#         gamma2,
#         sigma2,
#         leave_one_out=True,
#     )

#     proposal_means = theta_old + gamma * (
#         prior_grad_log(theta_old) - lambda_ * grad_phi_old
#     )
#     theta_new = proposal_means + math.sqrt(2.0 * gamma) * torch.randn_like(theta_old)

#     log_proposal = compute_gaussian_mixture_log_density(
#         theta_new,
#         proposal_means,
#         w_old,
#         variance=2.0 * gamma,
#     )

#     phi_new, _ = compute_mmd_potential_and_grad_at(
#         theta_new,
#         theta_old,
#         w_old,
#         X_batch,
#         y_batch,
#         gamma2,
#         sigma2,
#     )
#     log_target = prior_log_prob(theta_new) - lambda_ * phi_new

#     fr_exponent = 1.0 - math.exp(-gamma)
#     log_w_new = fr_exponent * (log_target - log_proposal)
#     w_new = torch.softmax(log_w_new, dim=0)

#     return theta_new.detach(), w_new.detach()


# def birth_death_population_step(theta, centred_rate, gamma):
#     """
#     Stochastic birth/death discretisation used by birth--death Langevin.

#     If beta_j is the centred FR rate, [BDL-2019] kills particle j when beta_j>0
#     with probability 1-exp(-gamma*beta_j), and duplicates it when beta_j<0
#     with probability 1-exp(gamma*beta_j). A final unbiased random correction
#     returns the population to its original size. See also [BDL-2023] and
#     [SMC-WFR, Appendix E, Algorithm 3].

#     This routine implements only the population reaction step. The Langevin
#     transport and construction of beta_j are in train_step_wfr_bdl.
#     """
#     p = theta.shape[0]
#     particles = []

#     event_probability = 1.0 - torch.exp(-gamma * torch.abs(centred_rate))
#     uniforms = torch.rand(p, device=theta.device)

#     for j in range(p):
#         event = bool((uniforms[j] < event_probability[j]).item())
#         rate_j = float(centred_rate[j].item())

#         if rate_j > 0.0 and event:
#             # Death: omit this particle.
#             continue

#         particles.append(theta[j])

#         if rate_j < 0.0 and event:
#             # Birth: add one duplicate.
#             particles.append(theta[j].clone())

#     # Extremely unlikely safeguard if every particle was killed.
#     if len(particles) == 0:
#         particles = [theta[torch.argmin(centred_rate)].clone()]

#     population = torch.stack(particles, dim=0)
#     n_population = population.shape[0]

#     if n_population > p:
#         keep = torch.randperm(n_population, device=theta.device)[:p]
#         population = population[keep]
#     elif n_population < p:
#         add = torch.randint(0, n_population, (p - n_population,), device=theta.device)
#         population = torch.cat([population, population[add]], dim=0)

#     return population


# def train_step_wfr_bdl(
#     theta,
#     X_batch,
#     y_batch,
#     gamma2,
#     sigma2,
#     lambda_,
#     gamma,
#     kde_bandwidth,
# ):
#     """
#     Birth--death Langevin particle approximation of the PrO WFR equation.

#     FIXED-TARGET REFERENCE ALGORITHM
#     --------------------------------
#     [BDL-2019] and [BDL-2023] approximate the WFR gradient flow of
#     KL(Q || pi) by alternating:

#       1. ULA transport:

#            theta_j^+ = theta_j + gamma*grad log pi(theta_j)
#                        + sqrt(2 gamma)*noise_j;

#       2. a birth/death reaction with centred rate

#            beta_j = log q_hat(theta_j^+) - log pi(theta_j^+),
#            beta_bar_j = beta_j - average_l beta_l,

#          where q_hat is a KDE because an empirical measure has no ordinary
#          density. Positive beta_bar causes death and negative beta_bar causes
#          duplication. See [SMC-WFR, Appendix E, Algorithm 3] for a compact
#          statement of the original algorithm.

#     NONLINEAR PrO EXTENSION USED HERE
#     ---------------------------------
#     The formal first variation of the PrO free energy is

#         lambda*phi_Q + log q - log pi_0.

#     Accordingly, the transport drift is

#         grad log pi_0 - lambda*grad phi_Q,

#     and, after the transport step, the reaction rate is estimated as

#         beta_j = lambda*phi_Q(theta_j) + log q_hat(theta_j)
#                  - log pi_0(theta_j).

#     The rate is centred and passed to birth_death_population_step.

#     NOTES
#     ------
#     The ULA + KDE birth/death mechanism is the standard BDL construction for a
#     fixed reverse-KL target. Adding the nonlinear first variation is the formal
#     WFR discretisation for the PrO functional and is closely related in spirit
#     to birth--death algorithms for nonlinear mean-field optimisation, but this
#     exact score-plus-KL implementation is not a published canonical algorithm.
#     The fixed-target convergence results in [BDL-2019] and [BDL-2023] therefore
#     should not be quoted as convergence results for this code without a new
#     analysis.
#     """
#     p = theta.shape[0]
#     w = torch.ones(p, device=theta.device) / p

#     _, grad_phi = compute_mmd_potential_and_grad(
#         theta,
#         w,
#         X_batch,
#         y_batch,
#         gamma2,
#         sigma2,
#         leave_one_out=True,
#     )

#     theta_new = theta + gamma * (
#         prior_grad_log(theta) - lambda_ * grad_phi
#     ) + math.sqrt(2.0 * gamma) * torch.randn_like(theta)

#     w_uniform = torch.ones(p, device=theta.device) / p
#     phi_new, _ = compute_mmd_potential_and_grad(
#         theta_new,
#         w_uniform,
#         X_batch,
#         y_batch,
#         gamma2,
#         sigma2,
#         leave_one_out=False,
#     )
#     log_q, _ = compute_kde_log_density_and_score(
#         theta_new,
#         w_uniform,
#         kde_bandwidth,
#     )

#     rate = lambda_ * phi_new + log_q - prior_log_prob(theta_new)
#     centred_rate = rate - rate.mean()
#     theta_new = birth_death_population_step(theta_new, centred_rate, gamma)

#     w_new = torch.ones(p, device=theta.device) / p
#     return theta_new.detach(), w_new


# # ==================== 6. Algorithm update steps ====================
# def compute_vgd_velocity(
#     theta,
#     X_batch,
#     y_batch,
#     gamma2,
#     sigma2,
#     lambda_,
#     lengthscales,
# ):
#     """Compute the variational-gradient-descent particle velocity.

#     VGD is the nonlinear entropy-regularised analogue of Stein variational
#     gradient descent introduced in [VGD, Section 3.2.2.2]. For

#         J(Q) = L(Q) + KL(Q || Q0),

#     equation (19) of [VGD] evolves equally weighted particles according to

#         d theta_i / dt
#           = (1/p) sum_j [
#                 k(theta_i, theta_j)
#                 {grad log q0(theta_j) - grad_V L(Q)(theta_j)}
#                 + grad_1 k(theta_j, theta_i)
#             ].

#     In the present PrO problem,

#         L(Q) = lambda_n * S_MMD(P_Q),

#     so grad_V L(Q)(theta_j) is ``lambda_ * grad_phi[j]``. The first term
#     smooths the current PrO drift through a parameter-space kernel. The second
#     term is repulsive: without it, a deterministic particle system would tend
#     to collapse and would not represent the entropy in KL(Q || Q0).

#     Kernel used here
#     ----------------
#     Following the multiscale inverse-multiquadric construction used in the VGD
#     experiments of [VGD, Appendix B.2.5], this implementation averages

#         k_l(x,y) = (1 + ||x-y||^2 / l^2)^(-1/2)

#     over the positive values in ``lengthscales``. The paper's numerical length
#     scales were specific to its low-dimensional example; this script exposes
#     them as numerical parameters because parameter-space distances depend on
#     dimension and standardisation.

#     For diff[i,j] = theta_i - theta_j,

#         grad_1 k_l(theta_j, theta_i)
#           = diff[i,j] / l^2
#             * (1 + ||diff[i,j]||^2/l^2)^(-3/2).

#     Returns
#     -------
#     velocity : tensor [p,d]
#         Direction in which the particle locations should move to decrease the
#         nonlinear entropy-regularised objective.
#     """
#     if len(lengthscales) == 0:
#         raise ValueError("VGD requires at least one positive kernel lengthscale.")
#     if any(float(ell) <= 0.0 for ell in lengthscales):
#         raise ValueError("All VGD kernel lengthscales must be strictly positive.")

#     p = theta.shape[0]
#     w = torch.ones(p, device=theta.device) / p

#     # Equation (19) uses the current empirical measure Q_p itself, rather than
#     # a leave-one-out approximation. For the MMD interaction, the derivative of
#     # a particle's self-kernel is zero, so the self term causes no singularity.
#     _, grad_phi = compute_mmd_potential_and_grad(
#         theta,
#         w,
#         X_batch,
#         y_batch,
#         gamma2,
#         sigma2,
#         leave_one_out=False,
#     )

#     # This is grad log q0 - grad_V L(Q) in equation (19). For the standard
#     # Gaussian prior, prior_grad_log(theta)=-theta.
#     variational_drift = prior_grad_log(theta) - lambda_ * grad_phi  # [p,d]

#     # diff[i,j] = theta_i - theta_j. The kernel is symmetric, but retaining the
#     # orientation is important for the sign of grad_1 k(theta_j, theta_i).
#     diff = theta[:, None, :] - theta[None, :, :]  # [p,p,d]
#     squared_distance = torch.sum(diff ** 2, dim=-1)  # [p,p]

#     kernel_sum = torch.zeros(p, p, device=theta.device, dtype=theta.dtype)
#     repulsion_sum = torch.zeros(p, p, theta.shape[1], device=theta.device, dtype=theta.dtype)

#     for ell in lengthscales:
#         ell2 = float(ell) ** 2
#         base = 1.0 + squared_distance / ell2
#         kernel_sum = kernel_sum + torch.rsqrt(base)
#         repulsion_sum = repulsion_sum + (diff / ell2) * base.pow(-1.5)[:, :, None]

#     n_scales = float(len(lengthscales))
#     kernel = kernel_sum / n_scales
#     repulsion = repulsion_sum / n_scales

#     # For each evaluation particle i, average source-particle contributions j.
#     smoothed_drift = torch.sum(
#         kernel[:, :, None] * variational_drift[None, :, :], dim=1
#     ) / p
#     repulsive_drift = torch.sum(repulsion, dim=1) / p

#     return smoothed_drift + repulsive_drift


# def train_step_vgd(
#     theta,
#     X_batch,
#     y_batch,
#     gamma2,
#     sigma2,
#     lambda_,
#     step_size,
#     lengthscales,
#     first_moment,
#     second_moment,
#     iteration,
#     beta1=0.9,
#     beta2=0.999,
#     adam_epsilon=1e-8,
# ):
#     """One Adam-discretised variational gradient descent step.

#     The underlying continuous-time method is equation (19) of [VGD]. A forward
#     Euler discretisation would use ``theta + step_size * velocity``. The VGD
#     experiments in [VGD, Appendix B.2.5] integrate the particle ODE using Adam,
#     so this implementation follows that numerical choice while keeping the
#     exact VGD vector field separate in ``compute_vgd_velocity``.

#     Equation (19) is stated using the exact variational gradient. To keep the
#     cost and data access comparable with the other methods in this script, our
#     implementation replaces the empirical-data average by a minibatch
#     estimate.

#     ``velocity`` is already a descent direction for the objective; therefore the
#     Adam increment is *added* to theta. The first- and second-moment arrays are
#     optimizer states, not physical particle velocities.
#     """
#     if step_size <= 0:
#         raise ValueError("VGD step_size must be strictly positive.")

#     velocity = compute_vgd_velocity(
#         theta,
#         X_batch,
#         y_batch,
#         gamma2,
#         sigma2,
#         lambda_,
#         lengthscales,
#     )

#     iteration = iteration + 1
#     first_moment = beta1 * first_moment + (1.0 - beta1) * velocity
#     second_moment = beta2 * second_moment + (1.0 - beta2) * velocity.square()

#     first_hat = first_moment / (1.0 - beta1 ** iteration)
#     second_hat = second_moment / (1.0 - beta2 ** iteration)
#     theta_new = theta + step_size * first_hat / (torch.sqrt(second_hat) + adam_epsilon)

#     return (
#         theta_new.detach(),
#         first_moment.detach(),
#         second_moment.detach(),
#         iteration,
#     )


# def compute_nula_coefficients(step_size, friction):
#     """Return the exact frozen-force NULA coefficients from [NULA, Algorithm 1].

#     During one interval of length h, NULA freezes the mean-field force and
#     exactly integrates

#         dX_t = V_t dt,
#         dV_t = -gamma V_t dt - force dt + sqrt(2 gamma) dB_t.

#     Writing ``friction=gamma`` and ``step_size=h``, Algorithm 1 defines

#         phi2 = exp(-gamma h),
#         phi0 = (1-phi2)/gamma,
#         phi1 = (h-phi0)/gamma,

#     together with a jointly Gaussian increment (B_x,B_v) with scalar block
#     covariances Sigma11, Sigma12 and Sigma22. The position and velocity noises
#     must be correlated; drawing them independently would not implement the
#     spacetime discretisation analysed in [NULA].
#     """
#     h = float(step_size)
#     gamma = float(friction)
#     if h <= 0.0:
#         raise ValueError("NULA step_size must be strictly positive.")
#     if gamma <= 0.0:
#         raise ValueError("NULA friction must be strictly positive.")

#     # expm1 avoids losing precision when gamma*h is small.
#     one_minus_phi2 = -math.expm1(-gamma * h)
#     phi2 = 1.0 - one_minus_phi2
#     phi0 = one_minus_phi2 / gamma
#     phi1 = (h - phi0) / gamma

#     sigma11 = (2.0 / gamma) * (
#         h - 2.0 * phi0 + (1.0 - phi2 ** 2) / (2.0 * gamma)
#     )
#     sigma12 = (one_minus_phi2 ** 2) / gamma
#     sigma22 = 1.0 - phi2 ** 2

#     # Floating-point cancellation can produce tiny negative values when h is
#     # extremely small. Clamping at zero preserves the intended covariance in
#     # that numerical limit.
#     sigma11 = max(sigma11, 0.0)
#     sigma22 = max(sigma22, 0.0)

#     return phi0, phi1, phi2, sigma11, sigma12, sigma22


# def sample_nula_noise(theta, sigma11, sigma12, sigma22):
#     """Sample the correlated Gaussian position/velocity increment in NULA.

#     For every particle coordinate, (B_x,B_v) has covariance

#         [[Sigma11, Sigma12],
#          [Sigma12, Sigma22]].

#     A two-normal conditional construction is used instead of materialising a
#     2d-by-2d covariance matrix. This is exact because each coordinate has the
#     same independent 2-by-2 covariance block.
#     """
#     z_position = torch.randn_like(theta)
#     z_velocity = torch.randn_like(theta)

#     if sigma11 <= 1e-30:
#         # This branch is relevant only at a vanishingly small time step.
#         noise_position = torch.zeros_like(theta)
#         noise_velocity = math.sqrt(max(sigma22, 0.0)) * z_velocity
#         return noise_position, noise_velocity

#     sqrt_sigma11 = math.sqrt(sigma11)
#     conditional_coefficient = sigma12 / sqrt_sigma11
#     conditional_variance = sigma22 - (sigma12 ** 2) / sigma11
#     conditional_variance = max(conditional_variance, 0.0)

#     noise_position = sqrt_sigma11 * z_position
#     noise_velocity = (
#         conditional_coefficient * z_position
#         + math.sqrt(conditional_variance) * z_velocity
#     )
#     return noise_position, noise_velocity


# def train_step_underdamped_pro(
#     theta,
#     velocity,
#     X_batch,
#     y_batch,
#     gamma2,
#     sigma2,
#     lambda_,
#     step_size,
#     friction,
#     p,
# ):
#     """One N-particle underdamped Langevin step for the squared-MMD PrO target.

#     The PrO free energy can be written as

#         lambda_n S_MMD(P_Q) - integral log pi(theta) Q(d theta) + Ent(Q).

#     In the notation of [NULA], the smooth intrinsic force is therefore

#         D_Q F(Q,theta)
#           = lambda_n grad phi_Q(theta) - grad log pi(theta).

#     The entropy coefficient is one, so the kinetic noise amplitude is
#     sqrt(2*friction), exactly as in Algorithm 1 of [NULA]. At each iteration the
#     current empirical PrO force is frozen, and the kinetic Ornstein--Uhlenbeck
#     system is integrated over one time step:

#         theta^+ = theta + phi0 velocity - phi1 force + B_x,
#         velocity^+ = phi2 velocity - phi0 force + B_v.

#     Algorithm 1 of [NULA] is formulated with the exact intrinsic derivative of
#     the population loss. This script uses a fresh data minibatch at each step,
#     as does the overdamped comparison. The update is therefore an exact NULA
#     spacetime step for a *frozen stochastic-gradient force*, but the convergence
#     theorem in [NULA] does not automatically apply to this minibatch extension.

#     The force uses the full empirical measure, matching the particle measure in
#     [NULA]. Unlike ordinary MFLD, the method retains velocity across iterations;
#     discarding it would remove the underdamped dynamics and its acceleration
#     mechanism.
#     """
#     if velocity.shape != theta.shape:
#         raise ValueError("NULA velocity and theta must have the same shape.")

#     w = torch.ones(p, device=theta.device) / p
#     _, grad_phi = compute_mmd_potential_and_grad(
#         theta,
#         w,
#         X_batch,
#         y_batch,
#         gamma2,
#         sigma2,
#         leave_one_out=False,
#     )
#     force = lambda_ * grad_phi - prior_grad_log(theta)

#     phi0, phi1, phi2, sigma11, sigma12, sigma22 = compute_nula_coefficients(
#         step_size,
#         friction,
#     )
#     noise_position, noise_velocity = sample_nula_noise(
#         theta,
#         sigma11,
#         sigma12,
#         sigma22,
#     )

#     theta_new = theta + phi0 * velocity - phi1 * force + noise_position
#     velocity_new = phi2 * velocity - phi0 * force + noise_velocity

#     return theta_new.detach(), velocity_new.detach()


# def train_step_pro(
#     theta, X_batch, y_batch, gamma2, sigma2, lambda_, dt, p,
# ):
#     """
#     Euler--Maruyama mean-field Langevin dynamics (MFLD) for the PrO posterior.

#     The Wasserstein gradient flow of

#         F(Q) = lambda*S_MMD(Q) + KL(Q || Pi)

#     has nonlinear Fokker--Planck equation

#         partial_t q
#           = div[q * (lambda*grad phi_Q - grad log pi)] + Delta q.

#     Its McKean--Vlasov representation is

#         dTheta_t = -[lambda*grad phi_{Q_t}(Theta_t)
#                      - grad log pi(Theta_t)] dt + sqrt(2) dB_t.

#     This function applies Euler--Maruyama to an equally weighted leave-one-out
#     particle approximation of that SDE. This is the algorithm advocated for PrO
#     posteriors in [PRO, Section 6 and Appendix E], and is a standard MFLD/noisy
#     particle-gradient method for nonlinear entropy-regularised objectives; see
#     [MFLD-HRSS] and [MFLD-CHIZAT].

#     Unlike the WFR adaptations below, this is directly a standard discretisation
#     of the nonlinear squared-MMD functional itself rather than a fixed-target
#     sampler with a self-consistent target substituted into it.
#     """
#     w = torch.ones(p, device=device) / p
#     _, grad_score = compute_mmd_potential_and_grad(
#         theta,
#         w,
#         X_batch,
#         y_batch,
#         gamma2,
#         sigma2,
#         leave_one_out=True,
#     )

#     grad_prior = prior_grad_log(theta)
#     # Euler--Maruyama has a deterministic drift of order dt and Brownian
#     # increment with standard deviation sqrt(2 dt). Omitting the noise would
#     # remove the entropy-generating part of the Wasserstein KL flow.
#     noise = torch.randn_like(theta) * math.sqrt(2.0 * dt)
#     theta_new = theta - dt * (lambda_ * grad_score - grad_prior) + noise
#     return theta_new.detach()


# def compute_deterministic_flow_terms(
#     theta,
#     w,
#     X_batch,
#     y_batch,
#     gamma2,
#     sigma2,
#     lambda_,
#     kde_bandwidth,
# ):
#     """
#     KDE approximation of the formal WF/FR/WFR first-variation fields.

#     For F(Q)=lambda*S_MMD(Q)+KL(Q||Pi), the formal fields are

#         Wasserstein velocity gradient:
#             lambda*grad phi_Q + grad log q - grad log pi;

#         Fisher--Rao reaction potential:
#             lambda*phi_Q + log q - log pi.

#     We replace q and grad log q by a Gaussian KDE of the weighted particles.
#     This yields a direct deterministic particle discretisation of the continuum
#     equations. Kernel approximations of Wasserstein and Fisher--Rao flows are a
#     recognised methodology, and rigorous kernel FR particle schemes for
#     nonlinear entropic mean-field problems are developed in
#     [FR-KERNEL-MEANFIELD]. However, this particular shared-bandwidth KDE scheme
#     is a baseline chosen for simplicity, not a uniquely standard algorithm.
#     """
#     phi, grad_score = compute_mmd_potential_and_grad(
#         theta,
#         w,
#         X_batch,
#         y_batch,
#         gamma2,
#         sigma2,
#         leave_one_out=False,
#     )
#     log_q, score_q = compute_kde_log_density_and_score(theta, w, kde_bandwidth)

#     # Gradient of lambda * predictive score + KL(Q || Pi). Here score_q means
#     # grad log q_hat, while -prior_grad_log is -grad log pi.
#     transport_grad = lambda_ * grad_score + score_q - prior_grad_log(theta)

#     # First variation, up to an additive constant which disappears after
#     # centring in the Fisher-Rao flow.
#     reaction_potential = lambda_ * phi + log_q - prior_log_prob(theta)

#     return transport_grad, reaction_potential


# def train_step_wfr_imp(
#     theta,
#     w,
#     X_batch,
#     y_batch,
#     gamma2,
#     sigma2,
#     lambda_,
#     eta_theta,
#     eta_w,
#     p,
#     momentum_buffer,
#     kde_bandwidth,
# ):
#     """
#     Heuristically damped version of the legacy deterministic KDE-WFR baseline.

#     It combines the KDE transport/reaction fields from
#     compute_deterministic_flow_terms with:

#       * an exponential moving average of the transport gradient; and
#       * convex damping of the new FR weights with the previous weights.

#     The coefficients 0.9/0.1 and 0.6/0.4 are numerical heuristics.
#     """
#     grad, eta = compute_deterministic_flow_terms(
#         theta,
#         w,
#         X_batch,
#         y_batch,
#         gamma2,
#         sigma2,
#         lambda_,
#         kde_bandwidth,
#     )

#     momentum_buffer = 0.9 * momentum_buffer + 0.1 * grad
#     theta_new = theta - eta_theta * momentum_buffer

#     w_raw = update_weights_fisher_rao(w, eta, eta_w)
#     w_new = 0.6 * w + 0.4 * w_raw
#     w_new = w_new / torch.sum(w_new)

#     return theta_new.detach(), w_new.detach(), momentum_buffer.detach()


# def train_step_old(
#     theta,
#     w,
#     X_batch,
#     y_batch,
#     gamma2,
#     sigma2,
#     lambda_,
#     eta_theta,
#     eta_w,
#     p,
#     mode,
#     kde_bandwidth,
# ):
#     """
#     Deterministic KDE approximations of WF, FR and WFR.

#     mode='wf':
#         explicit Euler transport of particle locations using the KDE estimate
#         of grad log q; weights remain fixed.

#     mode='fr':
#         support remains fixed and weights use the centred exponential FR update.

#     mode='wfr':
#         combines both updates in one explicit step.

#     The continuum equations are standard. Weighted location-and-mass particle
#     WFR discretisations for nonlinear functionals appear, for example, in
#     [WFR-NONLINEAR]. The present code differs because its KL(Q||Pi) term requires
#     a continuous-density/score approximation, supplied here by a Gaussian KDE.
#     """
#     grad, eta = compute_deterministic_flow_terms(
#         theta,
#         w,
#         X_batch,
#         y_batch,
#         gamma2,
#         sigma2,
#         lambda_,
#         kde_bandwidth,
#     )

#     if mode in ['wfr', 'wf']:
#         theta_new = theta - eta_theta * grad
#     else:
#         theta_new = theta

#     if mode in ['wfr', 'fr']:
#         w_new = update_weights_fisher_rao(w, eta, eta_w)
#     else:
#         w_new = w

#     return theta_new.detach(), w_new.detach()


# def get_lr_cosine(step, base_lr, total_steps):
#     floor = 0.01 * base_lr
#     return max(base_lr * 0.5 * (1.0 + math.cos(math.pi * step / total_steps)), floor)


# def get_lr_exp(step, base_lr, decay_rate):
#     floor = 0.01 * base_lr
#     return max(base_lr * (decay_rate ** step), floor)


# # ==================== 7. Predictive evaluation ====================
# def compute_predictive_nll(theta, w, X_eval, y_eval, sigma2, chunk_size=2048):
#     """Compute exact average negative log posterior-predictive density.

#     For observation b, the predictive density is

#         p_Q(y_b | x_b) = sum_j w_j Normal(y_b; x_b^T theta_j, sigma^2).

#     The result is the arithmetic mean of -log p_Q(y_b|x_b). ``logsumexp``
#     evaluates the mixture stably even when individual component densities are
#     extremely small. Smaller values indicate better predictive density fit.
#     """
#     log_w = torch.log(w.clamp_min(1e-30))
#     log_normalizer = -0.5 * math.log(2.0 * math.pi * sigma2)
#     total_nll = 0.0
#     total_count = 0

#     # Chunking changes memory use only; totals are accumulated before dividing
#     # by the complete number of evaluation observations.
#     for start in range(0, X_eval.shape[0], chunk_size):
#         stop = min(start + chunk_size, X_eval.shape[0])
#         X_chunk = X_eval[start:stop]
#         y_chunk = y_eval[start:stop]

#         pred_mean = X_chunk @ theta.T
#         log_components = (
#             log_w[None, :]
#             + log_normalizer
#             - 0.5 * (y_chunk - pred_mean) ** 2 / sigma2
#         )
#         nll = -torch.logsumexp(log_components, dim=1)
#         total_nll += float(nll.sum().item())
#         total_count += nll.numel()

#     return total_nll / total_count


# def compute_predictive_mmd(theta, w, X_eval, y_eval, gamma2, sigma2, chunk_size=512):
#     """Compute exact average conditional squared MMD on evaluation data.

#     For every x_b, this compares the mixture P_Q(.|x_b) with delta_{y_b}. The
#     Gaussian predictive components and Gaussian RBF kernel allow all three MMD
#     terms to be evaluated analytically:

#         E k(Y,Y'),  E k(Y,y_b),  and  k(y_b,y_b)=1.

#     The returned value averages these conditional discrepancies over b. Smaller
#     values indicate closer predictive distributions under the chosen kernel.
#     """
#     total_score = 0.0
#     total_count = 0

#     pair_variance = gamma2 + 2.0 * sigma2
#     pair_scale = math.sqrt(gamma2 / pair_variance)
#     data_variance = gamma2 + sigma2
#     data_scale = math.sqrt(gamma2 / data_variance)

#     for start in range(0, X_eval.shape[0], chunk_size):
#         stop = min(start + chunk_size, X_eval.shape[0])
#         X_chunk = X_eval[start:stop]
#         y_chunk = y_eval[start:stop]

#         pred_mean = X_chunk @ theta.T
#         diff_pair = pred_mean[:, :, None] - pred_mean[:, None, :]
#         K_pair = pair_scale * torch.exp(-(diff_pair ** 2) / (2.0 * pair_variance))

#         diff_data = pred_mean - y_chunk
#         K_data = data_scale * torch.exp(-(diff_data ** 2) / (2.0 * data_variance))

#         pair_term = torch.sum(
#             K_pair * w[None, :, None] * w[None, None, :], dim=(1, 2)
#         )
#         data_term = torch.sum(K_data * w[None, :], dim=1)
#         score = pair_term + 1.0 - 2.0 * data_term

#         total_score += float(score.sum().item())
#         total_count += score.numel()

#     return total_score / total_count


# # ==================== 8. Single independent experiment (run_experiment) ====================
# def run_experiment(
#     mode,
#     X_train,
#     y_train,
#     X_eval,
#     y_eval,
#     params,
#     K,
#     theta_init=None,
#     batch_indices=None,
# ):
#     """Run one complete optimization/sampling trajectory for one method.

#     Parameters
#     ----------
#     mode : str
#         Selects the update rule. Supported names are handled in the branch below.
#     X_train, y_train : tensors
#         Data used in stochastic minibatch first-variation estimates.
#     X_eval, y_eval : tensors
#         Held-out data used only for recorded diagnostics, never for updates.
#     params : dict
#         Contains both common target parameters and method-specific numerical
#         parameters. See ``base_params`` in main for a commented list.
#     K : int
#         Number of particle updates.
#     theta_init : optional tensor [p,d]
#         Common initial particles. Supplying this enables paired comparisons.
#     batch_indices : optional integer tensor [K,batch_size]
#         Pre-generated minibatches. Supplying this ensures every method sees the
#         same stochastic sequence of training observations.

#     Returns
#     -------
#     record_steps : integer tensor
#         Iterations at which diagnostics were evaluated.
#     eval_losses : float tensor
#         Exact predictive NLL at those iterations.
#     eval_mmd : float tensor
#         Exact conditional MMD^2 at those iterations.

#     NOTES
#     -----
#     Evaluation is intentionally less frequent than updating because evaluating
#     every held-out observation can cost as much as several minibatch steps. The
#     state ``theta,w`` is the evolving approximation Q; no averaging of past
#     states is performed.
#     """
#     if theta_init is None:
#         theta = prior_sample(None, params['p'])
#     else:
#         theta = theta_init.clone().to(device)


#     w = torch.ones(params['p'], device=device) / params['p']
#     base_w = w.clone()
#     momentum_buffer = torch.zeros_like(theta)

#     # VGD uses Adam only as an ODE integrator. These arrays are optimizer
#     # moments and should not be confused with physical velocity.
#     vgd_first_moment = torch.zeros_like(theta)
#     vgd_second_moment = torch.zeros_like(theta)
#     vgd_iteration = 0

#     # NULA augments each parameter particle with a physical momentum/velocity.
#     # Initialising from N(0,I) matches the kinetic Gaussian used in [NULA].
#     underdamped_velocity = (
#         torch.randn_like(theta) if mode == 'underdamped_pro' else None
#     )

#     record_steps = []
#     eval_losses = []
#     eval_mmd = []

#     batch_size = params.get('batch_size', 128)
#     eval_every = params.get('eval_every', 10)
#     max_mmd_eval_points = params.get('max_mmd_eval_points', 1024)

#     if X_eval.shape[0] > max_mmd_eval_points:
#         X_mmd_eval = X_eval[:max_mmd_eval_points]
#         y_mmd_eval = y_eval[:max_mmd_eval_points]
#     else:
#         X_mmd_eval = X_eval
#         y_mmd_eval = y_eval

#     for step in range(K):
#         if batch_indices is None:
#             idx = torch.randint(0, X_train.shape[0], (batch_size,), device=device)
#         else:
#             idx = batch_indices[step]
#         X_batch, y_batch = X_train[idx], y_train[idx]

#         current_eta_theta = get_lr_cosine(step, params['eta_theta'], K)
#         current_eta_w = get_lr_exp(step, params['eta_w'], 0.995)

#         if mode == 'pro':
#             # Standard nonlinear MFLD for the PrO functional: [PRO],
#             # [MFLD-HRSS], [MFLD-CHIZAT].
#             # A fixed small step size is appropriate for sampling from the
#             # stationary PrO law; annealing it to zero changes the dynamics.
#             theta = train_step_pro(
#                 theta,
#                 X_batch,
#                 y_batch,
#                 params['gamma2'],
#                 params['sigma2'],
#                 params['lambda_'],
#                 params['dt'],
#                 params['p'],
#             )
#             w = torch.ones(params['p'], device=device) / params['p']
#         elif mode == 'underdamped_pro':
#             # N-particle underdamped Langevin spacetime discretisation from
#             # [NULA, Algorithm 1], applied to the nonlinear PrO force.
#             theta, underdamped_velocity = train_step_underdamped_pro(
#                 theta,
#                 underdamped_velocity,
#                 X_batch,
#                 y_batch,
#                 params['gamma2'],
#                 params['sigma2'],
#                 params['lambda_'],
#                 params['nula_step_size'],
#                 params['nula_friction'],
#                 params['p'],
#             )
#             w = torch.ones(params['p'], device=device) / params['p']
#         elif mode == 'vgd':
#             # Variational gradient descent, equation (19) of [VGD]. The
#             # particles remain equally weighted and evolve deterministically.
#             (
#                 theta,
#                 vgd_first_moment,
#                 vgd_second_moment,
#                 vgd_iteration,
#             ) = train_step_vgd(
#                 theta,
#                 X_batch,
#                 y_batch,
#                 params['gamma2'],
#                 params['sigma2'],
#                 params['lambda_'],
#                 params['vgd_step_size'],
#                 params['vgd_lengthscales'],
#                 vgd_first_moment,
#                 vgd_second_moment,
#                 vgd_iteration,
#                 beta1=params.get('vgd_beta1', 0.9),
#                 beta2=params.get('vgd_beta2', 0.999),
#                 adam_epsilon=params.get('vgd_adam_epsilon', 1e-8),
#             )
#             w = torch.ones(params['p'], device=device) / params['p']
#         elif mode == 'fr_mirror':
#             # Standard entropic-mirror/FR first-variation step on fixed support:
#             # [FR-MIRROR], with the exact frozen-target interpolation from
#             # [FR-TEMPERING].
#             theta, w = train_step_fr_mirror(
#                 theta,
#                 w,
#                 X_batch,
#                 y_batch,
#                 params['gamma2'],
#                 params['sigma2'],
#                 params['lambda_'],
#                 params['eta_w'],
#                 base_w,
#             )
#         elif mode == 'wfr_smc':
#             # Standard fixed-target SMC-WFR substep [SMC-WFR], adapted here by
#             # substituting the current self-consistent PrO target pi_Q.
#             theta, w = train_step_wfr_smc(
#                 theta,
#                 w,
#                 X_batch,
#                 y_batch,
#                 params['gamma2'],
#                 params['sigma2'],
#                 params['lambda_'],
#                 params['dt'],
#             )
#         elif mode == 'wfr_bdl':
#             # Standard fixed-target BDL mechanism [BDL-2019, BDL-2023], with
#             # the nonlinear PrO first variation inserted into drift and rate.
#             theta, w = train_step_wfr_bdl(
#                 theta,
#                 X_batch,
#                 y_batch,
#                 params['gamma2'],
#                 params['sigma2'],
#                 params['lambda_'],
#                 params['dt'],
#                 params['kde_bandwidth'],
#             )
#         elif mode == 'wfr_imp':
#             # Nonstandard heuristic damping of the direct KDE-WFR baseline.
#             theta, w, momentum_buffer = train_step_wfr_imp(
#                 theta,
#                 w,
#                 X_batch,
#                 y_batch,
#                 params['gamma2'],
#                 params['sigma2'],
#                 params['lambda_'],
#                 current_eta_theta,
#                 current_eta_w,
#                 params['p'],
#                 momentum_buffer,
#                 params['kde_bandwidth'],
#             )
#         else:
#             # Direct KDE discretisations of the formal WF/FR/WFR equations;
#             # useful baselines, but not canonical single-paper implementations.
#             theta, w = train_step_old(
#                 theta,
#                 w,
#                 X_batch,
#                 y_batch,
#                 params['gamma2'],
#                 params['sigma2'],
#                 params['lambda_'],
#                 current_eta_theta,
#                 current_eta_w,
#                 params['p'],
#                 mode,
#                 params['kde_bandwidth'],
#             )

#         if step % eval_every == 0 or step == K - 1:
#             with torch.no_grad():
#                 nll = compute_predictive_nll(
#                     theta, w, X_eval, y_eval, params['sigma2']
#                 )
#                 mmd = compute_predictive_mmd(
#                     theta,
#                     w,
#                     X_mmd_eval,
#                     y_mmd_eval,
#                     params['gamma2'],
#                     params['sigma2'],
#                 )

#             record_steps.append(step)
#             eval_losses.append(nll)
#             eval_mmd.append(mmd)

#     return (
#         torch.tensor(record_steps, dtype=torch.long),
#         torch.tensor(eval_losses, dtype=torch.float32),
#         torch.tensor(eval_mmd, dtype=torch.float32),
#     )


# # ==================== 9. Full grid search module ====================
# def full_grid_search(
#     modes,
#     base_params,
#     param_grids,
#     num_grid_runs=3,
#     K_grid=1000,
# ):
#     """
#     Tune only algorithmic discretisation parameters on the validation set.

#     The target-defining quantities gamma2, lambda_, sigma2, the prior and the
#     number of particles are held fixed across modes so that every algorithm is
#     attempting to compute the same PrO posterior.
#     """
#     print("\n" + "=" * 60)
#     print("Starting validation-based numerical-parameter grid search...")
#     print("=" * 60)

#     best_params_by_mode = {}

#     for mode_index, mode in enumerate(modes):
#         grid = param_grids[mode]
#         keys = list(grid.keys())
#         combinations = list(itertools.product(*(grid[key] for key in keys)))

#         print(f"\nMode {mode.upper()}: {len(combinations)} combinations")
#         best_score = float('inf')
#         best_params = None

#         for i, combo in enumerate(combinations):
#             params = base_params.copy()
#             params.update(dict(zip(keys, combo)))
#             print(f"Progress: [{i + 1}/{len(combinations)}] Testing: "
#                   f"{dict(zip(keys, combo))}", end=" ")

#             total_loss = 0.0
#             for run in range(num_grid_runs):
#                 common_seed = run * 42 + 10
#                 torch.manual_seed(common_seed)
#                 theta_init = prior_sample(None, params['p'])
#                 batch_indices = torch.randint(
#                     0,
#                     X_train.shape[0],
#                     (K_grid, params['batch_size']),
#                     device=device,
#                 )

#                 # Separate stochastic-noise stream, while preserving the same
#                 # initialization and minibatches for every parameter choice.
#                 torch.manual_seed(common_seed + 100000 + mode_index)
#                 _, nll_series, mmd_series = run_experiment(
#                     mode,
#                     X_train,
#                     y_train,
#                     X_val,
#                     y_val,
#                     params,
#                     K_grid,
#                     theta_init=theta_init,
#                     batch_indices=batch_indices,
#                 )
#                 validation_score = mmd_series
#                 tail = max(1, validation_score.numel() // 5)
#                 total_loss += validation_score[-tail:].mean().item()

#             avg_loss = total_loss / num_grid_runs
#             print(f" -> Average tail validation target score: {avg_loss:.4f}")

#             if avg_loss < best_score:
#                 best_score = avg_loss
#                 best_params = params.copy()

#         print(
#             f"Best {mode.upper()} numerical parameters: "
#             f"{ {key: best_params[key] for key in keys} } "
#             f"(validation target score={best_score:.4f})"
#         )
#         best_params_by_mode[mode] = best_params

#     print("\nValidation grid search completed.")
#     return best_params_by_mode


# # ==================== 10. Experiment runner ====================
# def main():
#     # --- Control switches ---
#     ENABLE_GRID_SEARCH = False

#     # --- Common PrO target: fixed across every numerical algorithm ---
#     n_train = X_train.shape[0]
#     lambda_n = math.sqrt(n_train)

#     base_params = {

#         # Response-space RBF bandwidth used by both MMD training and evaluation.
#         'gamma2': gamma2_hat,

#         # Predictive-score multiplier in F(Q). This is part of the target, not a
#         # mere optimizer learning rate, so it is shared across algorithms.
#         'lambda_': lambda_n,

#         # Fixed variance of every Gaussian regression component.
#         'sigma2': sigma2_hat,

#         # Base deterministic transport and FR reaction step sizes. Individual
#         # methods may override these below or through validation grid search.
#         'eta_theta': 0.0002,
#         'eta_w': 0.01,

#         # Diffusion/ULA time step used by MFLD, SMC-WFR and BDL modes.
#         'dt': 0.0001,

#         # NULA time step and kinetic friction. These determine the exact
#         # frozen-force spacetime coefficients and correlated Gaussian noise.
#         'nula_step_size': 0.02,
#         'nula_friction': 5.0,

#         # VGD's Adam integration step and multiscale IMQ parameter kernel.
#         # Length scales are in standardized parameter-space units.
#         'vgd_step_size': 0.02,
#         'vgd_lengthscales': (0.125, 0.25, 0.5),
#         'vgd_beta1': 0.9,
#         'vgd_beta2': 0.999,
#         'vgd_adam_epsilon': 1e-8,

#         # Number of particles. Pairwise interactions generally cost O(p^2).
#         'p': 50,

#         # Parameter-space KDE bandwidth used only by algorithms that explicitly
#         # approximate log q or grad log q.
#         'kde_bandwidth': 0.8,

#         # Stochastic-gradient minibatch size and diagnostic frequency.
#         'batch_size': 128,
#         'eval_every': 25,

#         # MMD evaluation has O(p^2) work per observation, so it uses a fixed,
#         # reproducible subset of held-out points when the set is large.
#         'max_mmd_eval_points': 1024,
#     }

#     print("\nCommon PrO target parameters:")
#     print(
#         f"target_score=MMD, "
#         f"gamma^2={base_params['gamma2']:.4f}, "
#         f"lambda_n={base_params['lambda_']:.4f}, "
#         f"sigma^2={base_params['sigma2']:.4f}, "
#         f"p={base_params['p']}"
#     )

#     modes = [
#         'pro',
#         'underdamped_pro',
#         'vgd',
#         'fr_mirror',
#         'wfr_smc',
#         'wfr_bdl',
#         'wfr',
#         'fr',
#     ]

#     # The common target is fixed. Only numerical time steps are method-specific.
#     final_params_by_mode = {mode: base_params.copy() for mode in modes}
#     final_params_by_mode['pro']['dt'] = 0.002
#     final_params_by_mode['underdamped_pro'].update(
#         nula_step_size=0.02, nula_friction=5.0
#     )
#     final_params_by_mode['vgd'].update(
#         vgd_step_size=0.02, vgd_lengthscales=(0.125, 0.25, 0.5)
#     )
#     final_params_by_mode['fr_mirror']['eta_w'] = 0.01
#     final_params_by_mode['wfr_smc']['dt'] = 0.002
#     final_params_by_mode['wfr_bdl']['dt'] = 0.001
#     final_params_by_mode['wfr'].update(eta_theta=0.0005, eta_w=0.005)
#     final_params_by_mode['fr']['eta_w'] = 0.005

#     if ENABLE_GRID_SEARCH:
#         param_grids = {
#             'pro': {'dt': [0.00005, 0.0001, 0.0002, 0.001, 0.002]},
#             'underdamped_pro': {
#                 'nula_step_size': [0.005, 0.01, 0.02, 0.03],
#                 'nula_friction': [1.0, 2.0, 5.0, 10.0],
#             },
#             'vgd': {
#                 'vgd_step_size': [0.005, 0.01, 0.02, 0.03],
#                 'vgd_lengthscales': [
#                     (0.125, 0.25, 0.5),
#                     (0.25, 0.5, 1.0),
#                     (0.5, 1.0, 2.0),
#                 ],
#             },
#             'fr_mirror': {'eta_w': [0.002, 0.005, 0.01, 0.02]},
#             'wfr_smc': {'dt': [0.00005, 0.0001, 0.0002, 0.0005]},
#             'wfr_bdl': {
#                 'dt': [0.00005, 0.0001, 0.0002],
#                 'kde_bandwidth': [0.5, 0.8, 1.0],
#             },
#             'wfr': {
#                 'eta_theta': [0.0001, 0.0002, 0.0005],
#                 'eta_w': [0.002, 0.005, 0.01],
#             },
#             'fr': {'eta_w': [0.002, 0.005, 0.01]},
#         }
#         final_params_by_mode = full_grid_search(
#             modes=modes,
#             base_params=base_params,
#             param_grids=param_grids,
#             num_grid_runs=2,
#             K_grid=500,
#         )

#     # --- Run the final comparison on the untouched test set ---
#     K = 1000
#     num_runs = 20
#     results = {
#         mode: {'steps': [], 'test_losses': [], 'mmd_series': []}
#         for mode in modes
#     }

#     print(f"\nStarting {num_runs} paired runs (K={K}) ...")

#     for run in range(num_runs):
#         common_seed = run * 42 + 1
#         torch.manual_seed(common_seed)
#         theta_init = prior_sample(None, base_params['p'])
#         batch_indices = torch.randint(
#             0,
#             X_train.shape[0],
#             (K, base_params['batch_size']),
#             device=device,
#         )

#         print(f"Run {run + 1}/{num_runs} ...")
#         for mode_index, mode in enumerate(modes):
#             torch.manual_seed(common_seed + 100000 + mode_index)
#             steps, test_losses, mmd_series = run_experiment(
#                 mode,
#                 X_train,
#                 y_train,
#                 X_test,
#                 y_test,
#                 final_params_by_mode[mode],
#                 K,
#                 theta_init=theta_init,
#                 batch_indices=batch_indices,
#             )
#             results[mode]['steps'].append(steps)
#             results[mode]['test_losses'].append(test_losses)
#             results[mode]['mmd_series'].append(mmd_series)

#     # ========== Post-processing and visualisation ==========
#     fig, axes = plt.subplots(1, 2, figsize=(18, 7))

#     for mode in modes:
#         plot_steps = results[mode]['steps'][0].numpy()
#         losses_mat = torch.stack(results[mode]['test_losses'], dim=0).numpy()
#         mmd_mat = torch.stack(results[mode]['mmd_series'], dim=0).numpy()

#         mean_loss = losses_mat.mean(axis=0)
#         std_loss = losses_mat.std(axis=0)
#         mean_mmd = mmd_mat.mean(axis=0)
#         std_mmd = mmd_mat.std(axis=0)

#         # Use unified style mapping
#         axes[0].plot(plot_steps, mean_loss, label=ALGORITHM_LABELS[mode],
#                      color=ALGORITHM_COLORS[mode], linestyle=ALGORITHM_LINESTYLES[mode], linewidth=2)
#         axes[0].fill_between(
#             plot_steps,
#             mean_loss - std_loss,
#             mean_loss + std_loss,
#             alpha=0.12,
#             color=ALGORITHM_COLORS[mode]
#         )
#         axes[1].plot(plot_steps, mean_mmd, label=ALGORITHM_LABELS[mode],
#                      color=ALGORITHM_COLORS[mode], linestyle=ALGORITHM_LINESTYLES[mode], linewidth=2)
#         axes[1].fill_between(
#             plot_steps,
#             mean_mmd - std_mmd,
#             mean_mmd + std_mmd,
#             alpha=0.12,
#             color=ALGORITHM_COLORS[mode]
#         )

#     axes[0].set_xlabel('Iteration (Steps)')
#     axes[0].set_ylabel('Test negative log predictive density')
#     axes[0].set_title('Posterior-predictive negative log-likelihood')
#     axes[0].legend(fontsize=9)
#     axes[0].grid(alpha=0.3)

#     axes[1].set_xlabel('Iteration (Steps)')
#     axes[1].set_ylabel('Test conditional MMD squared')
#     axes[1].set_title('Posterior-predictive squared MMD')
#     axes[1].legend(fontsize=9)
#     axes[1].grid(alpha=0.3)

#     fig.suptitle(
#         "Algorithms targeting the squared-MMD PrO posterior",
#         fontsize=14,
#     )
#     plt.tight_layout()
#     output_name = "pro_standard_flows_mmd.png"
#     plt.savefig(output_name, dpi=180)
#     print(f"Plot saved as '{output_name}'")
#     plt.show()


# if __name__ == "__main__":
#     main()

import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import itertools
import time

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

# ==================== 1. Device selection and configuration ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== 2. Load and preprocess real dataset ====================
def load_real_data():
    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
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
    return (torch.tensor(X_train, dtype=torch.float32, device=device),
            torch.tensor(y_train, dtype=torch.float32, device=device).view(-1, 1),
            torch.tensor(X_val, dtype=torch.float32, device=device),
            torch.tensor(y_val, dtype=torch.float32, device=device).view(-1, 1),
            torch.tensor(X_test, dtype=torch.float32, device=device),
            torch.tensor(y_test, dtype=torch.float32, device=device).view(-1, 1))

X_train, y_train, X_val, y_val, X_test, y_test = load_real_data()
n_features = X_train.shape[1]
d = n_features
print(f"Data dimension (features) d = {d}")

# ==================== 3. Linear regression model and prior definitions ====================
def prior_log_prob(theta):
    return -0.5 * torch.sum(theta ** 2, dim=-1)

def prior_grad_log(theta):
    return -theta

def prior_sample(key, p):
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

sigma2_hat, theta_ols = estimate_noise_variance(X_train, y_train)
gamma2_hat = median_heuristic_gamma2(y_train)
print(f"Estimated observation variance sigma^2 = {sigma2_hat:.4f}")
print(f"MMD kernel bandwidth gamma^2 = {gamma2_hat:.4f}")

# ==================== 4. Squared-MMD first variation ====================
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

# ==================== 5. KDE and resampling ====================
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

def update_weights_fisher_rao(w, eta, eta_w):
    eta_bar = torch.sum(w * eta)
    log_w_new = torch.log(w.clamp_min(1e-30)) - eta_w * (eta - eta_bar)
    return torch.softmax(log_w_new, dim=0)

# ==================== 6. Algorithm step functions ====================
def train_step_pro(theta, X_batch, y_batch, gamma2, sigma2, lambda_, dt, p):
    w = torch.ones(p, device=device) / p
    _, grad_score = compute_mmd_potential_and_grad(theta, w, X_batch, y_batch, gamma2, sigma2, leave_one_out=True)
    grad_prior = prior_grad_log(theta)
    noise = torch.randn_like(theta) * math.sqrt(2.0 * dt)
    theta_new = theta - dt * (lambda_ * grad_score - grad_prior) + noise
    return theta_new.detach()

def train_step_underdamped_pro(theta, velocity, X_batch, y_batch, gamma2, sigma2, lambda_, step_size, friction, p):
    if velocity.shape != theta.shape:
        raise ValueError("NULA velocity and theta must have the same shape.")
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
    if step_size <= 0:
        raise ValueError("VGD step_size must be strictly positive.")
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

def get_lr_cosine(step, base_lr, total_steps):
    floor = 0.01 * base_lr
    return max(base_lr * 0.5 * (1.0 + math.cos(math.pi * step / total_steps)), floor)

def get_lr_exp(step, base_lr, decay_rate):
    floor = 0.01 * base_lr
    return max(base_lr * (decay_rate ** step), floor)

# ==================== 7. Predictive evaluation ====================
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

# ==================== 8. run_experiment ====================
def run_experiment(mode, X_train, y_train, X_eval, y_eval, params, K, theta_init=None, batch_indices=None):
    if theta_init is None:
        theta = prior_sample(None, params['p'])
    else:
        theta = theta_init.clone().to(device)
    w = torch.ones(params['p'], device=device) / params['p']
    base_w = w.clone()
    momentum_buffer = torch.zeros_like(theta)
    vgd_first_moment = torch.zeros_like(theta)
    vgd_second_moment = torch.zeros_like(theta)
    vgd_iteration = 0
    underdamped_velocity = torch.randn_like(theta) if mode == 'underdamped_pro' else None

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

        current_eta_theta = get_lr_cosine(step, params['eta_theta'], K)
        current_eta_w = get_lr_exp(step, params['eta_w'], 0.995)

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
        elif mode == 'wfr_imp':
            theta, w, momentum_buffer = train_step_wfr_imp(theta, w, X_batch, y_batch,
                params['gamma2'], params['sigma2'], params['lambda_'], current_eta_theta, current_eta_w,
                params['p'], momentum_buffer, params['kde_bandwidth'])
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

# ==================== 9. Main ====================
def main():
    # --- Control switches ---
    ENABLE_GRID_SEARCH = False

    n_train = X_train.shape[0]
    lambda_n = math.sqrt(n_train)

    base_params = {
        'gamma2': gamma2_hat,
        'lambda_': lambda_n,
        'sigma2': sigma2_hat,
        'eta_theta': 0.0002,
        'eta_w': 0.01,
        'dt': 0.0001,
        'nula_step_size': 0.02,
        'nula_friction': 5.0,
        'vgd_step_size': 0.02,
        'vgd_lengthscales': (0.125, 0.25, 0.5),
        'vgd_beta1': 0.9,
        'vgd_beta2': 0.999,
        'vgd_adam_epsilon': 1e-8,
        'p': 50,
        'kde_bandwidth': 0.8,
        'batch_size': 128,
        'eval_every': 25,
        'max_mmd_eval_points': 1024,
    }

    print("\nCommon PrO target parameters:")
    print(f"target_score=MMD, gamma^2={base_params['gamma2']:.4f}, lambda_n={base_params['lambda_']:.4f}, sigma^2={base_params['sigma2']:.4f}, p={base_params['p']}")

    modes = ['pro', 'underdamped_pro', 'vgd', 'fr_mirror', 'wfr_smc', 'wfr_bdl', 'wfr', 'fr']

    final_params_by_mode = {mode: base_params.copy() for mode in modes}
    final_params_by_mode['pro']['dt'] = 0.002
    final_params_by_mode['underdamped_pro'].update(nula_step_size=0.02, nula_friction=5.0)
    final_params_by_mode['vgd'].update(vgd_step_size=0.02, vgd_lengthscales=(0.125, 0.25, 0.5))
    final_params_by_mode['fr_mirror']['eta_w'] = 0.01
    final_params_by_mode['wfr_smc']['dt'] = 0.002
    final_params_by_mode['wfr_bdl']['dt'] = 0.001
    final_params_by_mode['wfr'].update(eta_theta=0.0005, eta_w=0.005)
    final_params_by_mode['fr']['eta_w'] = 0.005

    if ENABLE_GRID_SEARCH:
        param_grids = {
            'pro': {'dt': [0.00005, 0.0001, 0.0002, 0.001, 0.002]},
            'underdamped_pro': {'nula_step_size': [0.005, 0.01, 0.02, 0.03], 'nula_friction': [1.0, 2.0, 5.0, 10.0]},
            'vgd': {'vgd_step_size': [0.005, 0.01, 0.02, 0.03], 'vgd_lengthscales': [(0.125,0.25,0.5),(0.25,0.5,1.0),(0.5,1.0,2.0)]},
            'fr_mirror': {'eta_w': [0.002, 0.005, 0.01, 0.02]},
            'wfr_smc': {'dt': [0.00005, 0.0001, 0.0002, 0.0005]},
            'wfr_bdl': {'dt': [0.00005, 0.0001, 0.0002], 'kde_bandwidth': [0.5, 0.8, 1.0]},
            'wfr': {'eta_theta': [0.0001, 0.0002, 0.0005], 'eta_w': [0.002, 0.005, 0.01]},
            'fr': {'eta_w': [0.002, 0.005, 0.01]},
        }
        # 
        pass

    K = 1000
    num_runs = 20

    results = {mode: {'steps': [], 'test_losses': [], 'mmd_series': []} for mode in modes}

    print(f"\nStarting {num_runs} paired runs (K={K}) ...")

    wall_times = {mode: 0.0 for mode in modes}

    for run in range(num_runs):
        common_seed = run * 42 + 1
        torch.manual_seed(common_seed)
        theta_init = prior_sample(None, base_params['p'])
        batch_indices = torch.randint(0, X_train.shape[0], (K, base_params['batch_size']), device=device)

        print(f"Run {run + 1}/{num_runs} ...")
        for mode_index, mode in enumerate(modes):
            torch.manual_seed(common_seed + 100000 + mode_index)
            start_time = time.time()
            steps, test_losses, mmd_series = run_experiment(
                mode, X_train, y_train, X_test, y_test,
                final_params_by_mode[mode], K,
                theta_init=theta_init, batch_indices=batch_indices,
            )
            elapsed = time.time() - start_time
            wall_times[mode] += elapsed
            results[mode]['steps'].append(steps)
            results[mode]['test_losses'].append(test_losses)
            results[mode]['mmd_series'].append(mmd_series)

    avg_wall_times = {mode: wall_times[mode] / num_runs for mode in modes}

    # table 2
    print("\n--- Table 2: Final Performance (Mean ± Std over 20 runs) ---")
    print(f"{'Algorithm':<12} {'Final NLL':<20} {'Final MMD':<20} {'Wall Time (s)':<15} {'Convergence Iter':<20}")
    for mode in modes:
        mmd_mat = torch.stack(results[mode]['mmd_series'], dim=0).numpy()
        nll_mat = torch.stack(results[mode]['test_losses'], dim=0).numpy()
        final_nll_mean = nll_mat[:, -1].mean()
        final_nll_std = nll_mat[:, -1].std()
        final_mmd_mean = mmd_mat[:, -1].mean()
        final_mmd_std = mmd_mat[:, -1].std()

        # 
        conv_iters = []
        for i in range(num_runs):
            series = mmd_mat[i]  # length L
            L = len(series)
            final_val = series[-1]
            start_val = series[0]
            threshold = final_val + (start_val - final_val) * 0.01

            if L < 3:
                conv_iters.append(L - 1)
                continue

            # 
            d2 = np.diff(series, n=2)  # length L-2

            conv_idx = None
            for k in range(L - 2):
                #
                mid_idx = k + 1
                if abs(d2[k]) < 0.05 and series[mid_idx] < threshold:
                    conv_idx = mid_idx
                    break

            if conv_idx is None:
                
                idx = np.where(series < threshold)[0]
                if len(idx) > 0:
                    conv_idx = idx[0]
                else:
                    conv_idx = L - 1
            conv_iters.append(conv_idx)

        conv_avg = np.mean(conv_iters)
        print(f"{ALGORITHM_LABELS[mode]:<12} {final_nll_mean:.4f}±{final_nll_std:.4f}  "
              f"{final_mmd_mean:.4f}±{final_mmd_std:.4f}  "
              f"{avg_wall_times[mode]:<15.2f} {conv_avg:<20.1f}")

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for mode in modes:
        plot_steps = results[mode]['steps'][0].numpy()
        losses_mat = torch.stack(results[mode]['test_losses'], dim=0).numpy()
        mmd_mat = torch.stack(results[mode]['mmd_series'], dim=0).numpy()

        mean_loss = losses_mat.mean(axis=0)
        std_loss = losses_mat.std(axis=0)
        mean_mmd = mmd_mat.mean(axis=0)
        std_mmd = mmd_mat.std(axis=0)

        axes[0].plot(plot_steps, mean_loss, label=ALGORITHM_LABELS[mode],
                     color=ALGORITHM_COLORS[mode], linestyle=ALGORITHM_LINESTYLES[mode], linewidth=2)
        axes[0].fill_between(plot_steps, mean_loss - std_loss, mean_loss + std_loss,
                             alpha=0.12, color=ALGORITHM_COLORS[mode])
        axes[1].plot(plot_steps, mean_mmd, label=ALGORITHM_LABELS[mode],
                     color=ALGORITHM_COLORS[mode], linestyle=ALGORITHM_LINESTYLES[mode], linewidth=2)
        axes[1].fill_between(plot_steps, mean_mmd - std_mmd, mean_mmd + std_mmd,
                             alpha=0.12, color=ALGORITHM_COLORS[mode])

    axes[0].set_xlabel('Iteration (Steps)')
    axes[0].set_ylabel('Test negative log predictive density')
    axes[0].set_title('Posterior-predictive negative log-likelihood')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel('Iteration (Steps)')
    axes[1].set_ylabel('Test conditional MMD squared')
    axes[1].set_title('Posterior-predictive squared MMD')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Algorithms targeting the squared-MMD PrO posterior", fontsize=14)
    plt.tight_layout()
    output_name = "pro_standard_flows_mmd.png"
    plt.savefig(output_name, dpi=180)
    print(f"Plot saved as '{output_name}'")
    plt.show()

if __name__ == "__main__":
    main()
