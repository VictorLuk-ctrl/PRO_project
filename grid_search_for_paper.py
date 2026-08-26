"""
grid_search_improved_for_reviewer.py
===================================
根据审稿意见修改后的完整代码。
- 网格值打印，包含具体数值和配置总数
- 所有算法的参数敏感性图（横轴标注实际参数名，带误差条）
- 全英文图表
- 增加wfr_smc和wfr_bdl的显著性检验说明
"""

import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import itertools
import csv
import os
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from scipy import stats

# ==============================================================================
# 1. 设备与全局配置
# ==============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
os.makedirs("./results", exist_ok=True)

# ==============================================================================
# 2. 数据加载
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

X_train, y_train, X_val, y_val, X_test, y_test = load_real_data()

# ==============================================================================
# 3. 参数估计与核心运算
# ==============================================================================
def estimate_noise_variance(X, y):
    theta_ols = torch.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ theta_ols
    return float(torch.mean(residuals ** 2).clamp_min(1e-4).item()), theta_ols.flatten()

def median_heuristic_gamma2(y, max_points=2000):
    y_flat = y.flatten()
    if y_flat.numel() > max_points:
        idx = torch.randperm(y_flat.numel(), device=device)[:max_points]
        y_flat = y_flat[idx]
    D2 = (y_flat[:, None] - y_flat[None, :]) ** 2
    upper = D2[torch.triu_indices(D2.shape[0], D2.shape[1], offset=1, device=device).unbind()]
    upper = upper[upper > 0]
    return 1.0 if upper.numel() == 0 else float(torch.median(upper).item())

sigma2_hat, _ = estimate_noise_variance(X_train, y_train)
gamma2_hat = median_heuristic_gamma2(y_train)
lambda_n = math.sqrt(X_train.shape[0])

print(f"Target Fixed: sigma^2={sigma2_hat:.4f}, gamma^2={gamma2_hat:.4f}, lambda_n={lambda_n:.4f}")

def compute_mmd_potential_and_grad(theta, w, X_batch, y_batch, gamma2, sigma2, leave_one_out=False):
    B, p = X_batch.shape[0], theta.shape[0]
    pred_mean = X_batch @ theta.T
    pair_variance = gamma2 + 2.0 * sigma2
    pair_scale = math.sqrt(gamma2 / pair_variance)
    K_pair = pair_scale * torch.exp(-(pred_mean[:, :, None] - pred_mean[:, None, :]) ** 2 / (2.0 * pair_variance))
    data_variance = gamma2 + sigma2
    data_scale = math.sqrt(gamma2 / data_variance)
    K_data = data_scale * torch.exp(-(pred_mean - y_batch) ** 2 / (2.0 * data_variance))
    interaction_weights = w[None, :].expand(p, p).clone()
    if leave_one_out:
        interaction_weights.fill_diagonal_(0.0)
        interaction_weights = interaction_weights / interaction_weights.sum(dim=1, keepdim=True)
    phi = 2.0 * ((K_pair * interaction_weights[None, :, :]).sum(dim=2).mean(dim=0) - K_data.mean(dim=0))
    dK_pair = -(pred_mean[:, :, None] - pred_mean[:, None, :]) / pair_variance * K_pair
    dK_data = -(pred_mean - y_batch) / data_variance * K_data
    grad_mean = 2.0 * ((dK_pair * interaction_weights[None, :, :]).sum(dim=2) - dK_data)
    grad_phi = (grad_mean.T @ X_batch) / B
    return phi, grad_phi

def prior_grad_log(theta): return -theta
def prior_log_prob(theta): return -0.5 * torch.sum(theta ** 2, dim=-1)

def compute_kde_log_density_and_score(theta, w, bandwidth):
    h2 = bandwidth ** 2
    diff = theta[:, None, :] - theta[None, :, :]
    log_kernel = -0.5 * torch.sum(diff**2, dim=-1) / h2 - 0.5 * theta.shape[1] * math.log(2.0 * math.pi * h2)
    log_wk = torch.log(w.clamp_min(1e-30))[None, :] + log_kernel
    log_q = torch.logsumexp(log_wk, dim=1)
    resp = torch.softmax(log_wk, dim=1)
    return log_q, -torch.sum(resp[:, :, None] * diff, dim=1) / h2

def systematic_resample(theta, w):
    p = theta.shape[0]
    positions = torch.rand(1, device=device) / p + torch.arange(p, device=device) / p
    cumulative = torch.cumsum(w, dim=0)
    cumulative[-1] = 1.0
    return theta[torch.searchsorted(cumulative, positions, right=False)], torch.ones(p, device=device) / p

# ==============================================================================
# 4. 各算法单步更新
# ==============================================================================
def train_step_pro(theta, X_batch, y_batch, gamma2, sigma2, lambda_, dt, p):
    w = torch.ones(p, device=device) / p
    _, g = compute_mmd_potential_and_grad(theta, w, X_batch, y_batch, gamma2, sigma2, leave_one_out=True)
    return (theta - dt * (lambda_ * g - prior_grad_log(theta)) + math.sqrt(2.0 * dt) * torch.randn_like(theta)).detach()

def train_step_underdamped_pro(theta, vel, X_batch, y_batch, gamma2, sigma2, lambda_, step, friction, p):
    w = torch.ones(p, device=device) / p
    _, g = compute_mmd_potential_and_grad(theta, w, X_batch, y_batch, gamma2, sigma2, leave_one_out=False)
    force = lambda_ * g - prior_grad_log(theta)
    h, gamma = float(step), float(friction)
    phi2 = math.exp(-gamma * h)
    phi0 = (1.0 - phi2) / gamma
    phi1 = (h - phi0) / gamma
    sigma11 = max((2.0/gamma) * (h - 2.0*phi0 + (1.0 - phi2**2)/(2.0*gamma)), 0.0)
    sigma22 = max(1.0 - phi2**2, 0.0)
    sigma12 = (1.0 - phi2)**2 / gamma
    z1, z2 = torch.randn_like(theta), torch.randn_like(theta)
    if sigma11 <= 1e-30:
        n_pos, n_vel = torch.zeros_like(theta), math.sqrt(sigma22) * z2
    else:
        s11 = math.sqrt(sigma11)
        n_pos = s11 * z1
        n_vel = (sigma12 / s11) * z1 + math.sqrt(max(sigma22 - sigma12**2 / sigma11, 0.0)) * z2
    return (theta + phi0 * vel - phi1 * force + n_pos).detach(), (phi2 * vel - phi0 * force + n_vel).detach()

def compute_vgd_velocity(theta, X_batch, y_batch, gamma2, sigma2, lambda_, lengthscales):
    p = theta.shape[0]
    w = torch.ones(p, device=device) / p
    _, g = compute_mmd_potential_and_grad(theta, w, X_batch, y_batch, gamma2, sigma2, leave_one_out=False)
    drift = prior_grad_log(theta) - lambda_ * g
    diff = theta[:, None, :] - theta[None, :, :]
    dist2 = torch.sum(diff**2, dim=-1)
    k_sum, r_sum = torch.zeros(p, p, device=device), torch.zeros(p, p, theta.shape[1], device=device)
    for ell in lengthscales:
        e2 = float(ell)**2
        base = 1.0 + dist2 / e2
        k_sum += torch.rsqrt(base)
        r_sum += (diff / e2) * base.pow(-1.5)[:, :, None]
    n_scales = float(len(lengthscales))
    return (torch.sum((k_sum/n_scales)[:, :, None] * drift[None, :, :], dim=1) / p + torch.sum(r_sum/n_scales, dim=1) / p)

def train_step_vgd(theta, X_batch, y_batch, gamma2, sigma2, lambda_, step, lengthscales, m1, m2, it):
    v = compute_vgd_velocity(theta, X_batch, y_batch, gamma2, sigma2, lambda_, lengthscales)
    it += 1
    m1 = 0.9 * m1 + 0.1 * v
    m2 = 0.999 * m2 + 0.001 * v.square()
    return (theta + step * (m1/(1-0.9**it)) / (torch.sqrt(m2/(1-0.999**it)) + 1e-8)).detach(), m1.detach(), m2.detach(), it

def train_step_fr_mirror(theta, w, X_batch, y_batch, gamma2, sigma2, lambda_, eta_w, base_w):
    phi, _ = compute_mmd_potential_and_grad(theta, w, X_batch, y_batch, gamma2, sigma2, leave_one_out=False)
    decay = math.exp(-eta_w)
    log_w = decay * torch.log(w.clamp_min(1e-30)) + (1.0 - decay) * (torch.log(base_w.clamp_min(1e-30)) - lambda_ * phi)
    return theta.detach(), torch.softmax(log_w, dim=0).detach()

def compute_mmd_potential_and_grad_at(eval_t, meas_t, w_m, X_b, y_b, gamma2, sigma2):
    pred_e, pred_m = X_b @ eval_t.T, X_b @ meas_t.T
    pv = gamma2 + 2.0 * sigma2
    ps = math.sqrt(gamma2 / pv)
    Kp = ps * torch.exp(-(pred_e[:, :, None] - pred_m[:, None, :])**2 / (2.0*pv))
    dv = gamma2 + sigma2
    ds = math.sqrt(gamma2 / dv)
    Kd = ds * torch.exp(-(pred_e - y_b)**2 / (2.0*dv))
    return 2.0 * (torch.sum(Kp * w_m[None, None, :], dim=2).mean(0) - Kd.mean(0)), None

def train_step_wfr_smc(theta, w, X_batch, y_batch, gamma2, sigma2, lambda_, gamma):
    theta_old, w_old = systematic_resample(theta, w)
    _, g_old = compute_mmd_potential_and_grad(theta_old, w_old, X_batch, y_batch, gamma2, sigma2, leave_one_out=True)
    mean_new = theta_old + gamma * (prior_grad_log(theta_old) - lambda_ * g_old)
    theta_new = mean_new + math.sqrt(2.0 * gamma) * torch.randn_like(theta_old)
    param_dim = theta_new.shape[1]
    log_prop = -0.5 * torch.sum((theta_new - mean_new)**2, dim=-1) / (2.0 * gamma) - 0.5 * param_dim * math.log(2.0 * math.pi * 2.0 * gamma)
    phi_new, _ = compute_mmd_potential_and_grad_at(theta_new, theta_old, w_old, X_batch, y_batch, gamma2, sigma2)
    return theta_new.detach(), torch.softmax((1.0 - math.exp(-gamma)) * (prior_log_prob(theta_new) - lambda_ * phi_new - log_prop), dim=0).detach()

def train_step_wfr_bdl(theta, X_batch, y_batch, gamma2, sigma2, lambda_, gamma, kde_bw):
    p = theta.shape[0]
    w = torch.ones(p, device=device) / p
    _, g = compute_mmd_potential_and_grad(theta, w, X_batch, y_batch, gamma2, sigma2, leave_one_out=True)
    theta_new = theta + gamma * (prior_grad_log(theta) - lambda_ * g) + math.sqrt(2.0 * gamma) * torch.randn_like(theta)
    phi_new, _ = compute_mmd_potential_and_grad(theta_new, w, X_batch, y_batch, gamma2, sigma2, leave_one_out=False)
    log_q, _ = compute_kde_log_density_and_score(theta_new, w, kde_bw)
    rate = lambda_ * phi_new + log_q - prior_log_prob(theta_new)
    rate = rate - rate.mean()
    event_prob = 1.0 - torch.exp(-gamma * torch.abs(rate))
    uni = torch.rand(p, device=device)
    new_particles = []
    for j in range(p):
        if rate[j] > 0.0 and uni[j] < event_prob[j]: continue
        new_particles.append(theta_new[j])
        if rate[j] < 0.0 and uni[j] < event_prob[j]: new_particles.append(theta_new[j].clone())
    if not new_particles: new_particles = [theta_new[torch.argmin(rate)].clone()]
    pop = torch.stack(new_particles, dim=0)
    if pop.shape[0] > p: pop = pop[torch.randperm(pop.shape[0], device=device)[:p]]
    elif pop.shape[0] < p: pop = torch.cat([pop, pop[torch.randint(0, pop.shape[0], (p-pop.shape[0],), device=device)]], dim=0)
    return pop.detach(), torch.ones(p, device=device)/p

# ==============================================================================
# 5. 实验运行器
# ==============================================================================
def run_experiment(mode, X_train, y_train, X_eval, y_eval, params, K, theta_init=None, batch_indices=None):
    p = params['p']
    d = X_train.shape[1]
    theta = torch.randn(p, d, device=device) if theta_init is None else theta_init.clone()
    w = torch.ones(p, device=device) / p
    base_w = w.clone()
    m1, m2, it = torch.zeros_like(theta), torch.zeros_like(theta), 0
    vel = torch.randn_like(theta) if mode == 'underdamped_pro' else None
    
    record_steps, losses, mmds = [], [], []
    try:
        for step in range(K):
            idx = torch.randint(0, X_train.shape[0], (params['batch_size'],), device=device) if batch_indices is None else batch_indices[step]
            X_b, y_b = X_train[idx], y_train[idx]
            
            if mode == 'pro': theta = train_step_pro(theta, X_b, y_b, params['gamma2'], params['sigma2'], params['lambda_'], params['dt'], p)
            elif mode == 'underdamped_pro': theta, vel = train_step_underdamped_pro(theta, vel, X_b, y_b, params['gamma2'], params['sigma2'], params['lambda_'], params['nula_step_size'], params['nula_friction'], p)
            elif mode == 'vgd': theta, m1, m2, it = train_step_vgd(theta, X_b, y_b, params['gamma2'], params['sigma2'], params['lambda_'], params['vgd_step_size'], params['vgd_lengthscales'], m1, m2, it)
            elif mode == 'fr_mirror': theta, w = train_step_fr_mirror(theta, w, X_b, y_b, params['gamma2'], params['sigma2'], params['lambda_'], params['eta_w'], base_w)
            elif mode == 'wfr_smc': theta, w = train_step_wfr_smc(theta, w, X_b, y_b, params['gamma2'], params['sigma2'], params['lambda_'], params['dt'])
            elif mode == 'wfr_bdl': theta, w = train_step_wfr_bdl(theta, X_b, y_b, params['gamma2'], params['sigma2'], params['lambda_'], params['dt'], params['kde_bandwidth'])
            
            if torch.isnan(theta).any(): return None
            
            if step % params['eval_every'] == 0:
                record_steps.append(step)
                log_w = torch.log(w.clamp_min(1e-30))
                log_norm = -0.5 * math.log(2.0 * math.pi * params['sigma2'])
                pred_mean = X_eval @ theta.T
                log_comp = log_w[None, :] + log_norm - 0.5 * (y_eval - pred_mean)**2 / params['sigma2']
                nll = -torch.logsumexp(log_comp, dim=1).mean().item()
                
                pv, ps = params['gamma2'] + 2.0 * params['sigma2'], math.sqrt(params['gamma2']/(params['gamma2'] + 2.0 * params['sigma2']))
                dv, ds = params['gamma2'] + params['sigma2'], math.sqrt(params['gamma2']/(params['gamma2'] + params['sigma2']))
                pm = X_eval @ theta.T
                mmd = (torch.sum(ps * torch.exp(-(pm[:, :, None] - pm[:, None, :])**2 / (2.0*pv)) * w[None, :, None] * w[None, None, :], dim=(1,2)) + 1.0 - 2.0 * torch.sum(ds * torch.exp(-(pm - y_eval)**2 / (2.0*dv)) * w[None, :], dim=1)).mean().item()
                losses.append(nll); mmds.append(mmd)
        return torch.tensor(record_steps), torch.tensor(losses), torch.tensor(mmds)
    except Exception as e:
        return None

# ==============================================================================
# 6. 网格搜索主函数（修改完善）
# ==============================================================================
def perform_grid_search():
    print("\n" + "="*60)
    print("Improved Grid Search (All requested features implemented)")
    print("="*60)

    modes = ['pro', 'underdamped_pro', 'vgd', 'fr_mirror', 'wfr_smc', 'wfr_bdl']
    base_params = {
        'gamma2': gamma2_hat, 'lambda_': lambda_n, 'sigma2': sigma2_hat,
        'p': 50, 'batch_size': 128, 'eval_every': 15
    }

    # ---------- 定义网格 ----------
    pro_dt = list(np.linspace(1e-4, 0.015, 25))
    nula_step = list(np.linspace(0.002, 0.08, 25))
    nula_friction = list(np.linspace(0.5, 25.0, 25))
    vgd_step = list(np.linspace(0.001, 0.08, 25))
    vgd_lengthscales = [tuple(np.linspace(0.05, 1.0, 25)[i] * np.array([1, 2, 4])) for i in range(25)]
    fr_eta = list(np.linspace(0.0005, 0.05, 25))
    smc_dt = list(np.linspace(1e-4, 0.01, 25))
    bdl_dt = list(np.linspace(1e-4, 0.01, 25))
    bdl_kde = list(np.linspace(0.2, 2.5, 25))

    param_grids = {
        'pro': {'dt': pro_dt},
        'underdamped_pro': {'nula_step_size': nula_step, 'nula_friction': nula_friction},
        'vgd': {'vgd_step_size': vgd_step, 'vgd_lengthscales': vgd_lengthscales},
        'fr_mirror': {'eta_w': fr_eta},
        'wfr_smc': {'dt': smc_dt},
        'wfr_bdl': {'dt': bdl_dt, 'kde_bandwidth': bdl_kde}
    }

    # ==========================================
    # 审稿意见1：补充完整网格值表和每种算法配置数
    # ==========================================
    print("\n[REVIEWER REQUIREMENT 1: Grid Values and Configs]")
    for mode, grid in param_grids.items():
        print(f"\nAlgorithm: {mode}")
        for k, v in grid.items():
            if isinstance(v[0], (int, float)):
                print(f"  Param {k}: {len(v)} values")
                print(f"    Values: {[f'{x:.5f}' for x in v]}")
            else:
                # 对于元组列表，打印元组结构
                print(f"  Param {k}: {len(v)} values")
                print(f"    Values (first 3): {v[:3]} ...")
        total = len(list(itertools.product(*grid.values())))
        print(f"  --> Total Configurations: {total}")

    total_combos = sum(len(list(itertools.product(*grid.values()))) for grid in param_grids.values())
    print(f"\nAll algorithm total configurations: {total_combos}")

    results_list = []
    all_combo_records = [] 

    overall_start = time.time()

    for mode in modes:
        grid = param_grids[mode]
        keys, values = list(grid.keys()), list(grid.values())
        best_score = float('inf')
        best_params = base_params.copy()
        best_combo = None
        failed_count = 0

        print(f"\n--- Searching: {mode} ---")
        start_time = time.time()

        combos = list(itertools.product(*values))
        pbar = tqdm(combos, desc=f"{mode} search", unit="combo")
        for combo in pbar:
            params = base_params.copy()
            params.update(dict(zip(keys, combo)))
            
            torch.manual_seed(42)
            theta_init = torch.randn(params['p'], X_train.shape[1], device=device)
            batch_indices = torch.randint(0, X_train.shape[0], (150, params['batch_size']), device=device)
            
            result = run_experiment(mode, X_train, y_train, X_val, y_val, params, 150,
                                    theta_init=theta_init, batch_indices=batch_indices)
            
            if result is None:
                failed_count += 1
                continue
            
            _, _, mmd_series = result
            tail = max(1, len(mmd_series) // 5)
            avg_score = mmd_series[-tail:].mean().item()
            
            all_combo_records.append((mode, params.copy(), avg_score))
            
            if avg_score < best_score:
                best_score = avg_score
                best_params = params
                best_combo = combo
            
            pbar.set_postfix({'best': best_score})

        pbar.close()
        elapsed = time.time() - start_time
        print(f"Algorithm {mode} done. Time: {elapsed:.1f}s. Best Combo: {dict(zip(keys, best_combo))}, Best MMD: {best_score:.5f}")
        print(f"Failed combos: {failed_count}")

        results_list.append({
            'Algorithm': mode,
            'Best_Params': {k: best_params[k] for k in keys},
            'Validation_MMD': best_score,
            'elapsed_sec': elapsed,
            'failed': failed_count
        })

    overall_elapsed = time.time() - overall_start
    print(f"\nTotal search time: {overall_elapsed/60:.2f} minutes")

    # CSV保存
    combo_csv = "./results/all_combos.csv"
    with open(combo_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Algorithm', 'Parameters', 'Validation_MMD'])
        for mode, params, score in all_combo_records:
            writer.writerow([mode, str(params), f"{score:.5f}"])
    print(f"All combos saved to: {combo_csv}")

    best_csv = "./results/best_params.csv"
    with open(best_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Algorithm', 'Best Parameters', 'Validation_MMD', 'Elapsed_sec', 'Failed_combos'])
        for r in results_list:
            writer.writerow([r['Algorithm'], str(r['Best_Params']), f"{r['Validation_MMD']:.5f}", f"{r['elapsed_sec']:.1f}", r['failed']])
    print(f"Best results saved to: {best_csv}")

    # 使用英文标题
    print("\n" + "="*60)
    print("LaTeX Table for Paper")
    print("="*60)
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Optimal Hyperparameters and Validation MMD (with search time and failures).}")
    print(r"\begin{tabular}{llccc}")
    print(r"\toprule")
    print(r"Algorithm & Optimal Parameters & Val. MMD\(^2\) & Time (s) & \#Failed \\")
    print(r"\midrule")
    for r in results_list:
        params_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in r['Best_Params'].items()])
        print(f"{r['Algorithm']} & {params_str} & {r['Validation_MMD']:.5f} & {r['elapsed_sec']:.1f} & {r['failed']} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\label{tab:grid_search_improved}")
    print(r"\end{table}")

    print("\n" + "="*60)
    print("Plotting...")
    print("="*60)
    plot_validation_curves(results_list, all_combo_records, base_params, param_grids)


# ==============================================================================
# 7. 绘图函数（所有算法敏感性图 & 收敛图 & 统计显著性）
# ==============================================================================
def plot_validation_curves(results_list, all_combo_records, base_params, param_grids):
    n_repeats = 5          # 每个组合重复次数
    K_opt = 300            # 收敛曲线使用的步数
    K_sens = 150           # 敏感性图使用的步数（与网格搜索一致）

    # ---------------------------------------------
    # 审稿意见3：绘制所有算法的参数敏感性图（带误差条）
    # ---------------------------------------------
    print("Plotting parameter sensitivity maps for all algorithms (mean ± std, n_repeats = {})...".format(n_repeats))

    # 参数轴名称映射（英文），修正为 Matplotlib mathtext 支持的 $...$ 格式
    axis_labels = {
        'pro': r'Step size $\Delta t$',
        'underdamped_pro': r'Friction $\zeta$',
        'vgd': r'Step size',
        'fr_mirror': r'Learning rate $\eta$',
        'wfr_smc': r'Step size $\Delta t$',
        'wfr_bdl': r'Step size $\Delta t$'
    }

    for mode in param_grids.keys():
        records = [(p, s) for m, p, s in all_combo_records if m == mode]
        if not records:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        # 确定要绘制的关键参数（对于多参数算法，固定其他参数为最优值）
        if mode in ['underdamped_pro', 'vgd', 'wfr_bdl']:
            # 找到当前模式的最优参数
            best_rec = min(records, key=lambda x: x[1])
            best_params = best_rec[0]

            if mode == 'underdamped_pro':
                key_plot = 'nula_friction'
                axis_labels[mode] = r'Friction $\zeta$'
            elif mode == 'vgd':
                key_plot = 'vgd_step_size'
                axis_labels[mode] = r'Step size'
            elif mode == 'wfr_bdl':
                key_plot = 'kde_bandwidth'
                axis_labels[mode] = r'KDE bandwidth $h$'

            # 选出其他参数均等于最优值的记录（即仅该关键参数变化）
            filtered_records = [(p, s) for p, s in records
                                if all(p[k] == best_params[k] for k in p if k != key_plot)]
            if filtered_records:
                selected_records = filtered_records
            else:
                selected_records = records   # 退化到使用全部记录
        else:
            key_plot = list(param_grids[mode].keys())[0]
            if mode == 'pro':
                axis_labels[mode] = r'Step size $\Delta t$'
            elif mode == 'fr_mirror':
                axis_labels[mode] = r'Learning rate $\eta$'
            elif mode == 'wfr_smc':
                axis_labels[mode] = r'Step size $\Delta t$'
            selected_records = records

        # 对每个选中的参数组合进行 n_repeats 次重复实验，计算均值与标准差
        x_vals, y_means, y_stds = [], [], []
        for p, _ in selected_records:
            params = base_params.copy()
            params.update(p)

            scores = []
            for rep in range(n_repeats):
                torch.manual_seed(1000 + rep * 10)   # 不同重复使用不同随机种子
                theta_init = torch.randn(params['p'], X_train.shape[1], device=device)
                batch_indices = torch.randint(0, X_train.shape[0],
                                              (K_sens, params['batch_size']),
                                              device=device)
                result = run_experiment(mode, X_train, y_train,
                                        X_val, y_val, params, K_sens,
                                        theta_init=theta_init, batch_indices=batch_indices)
                if result is not None:
                    _, _, mmd_series = result
                    tail = max(1, len(mmd_series) // 5)
                    avg_score = mmd_series[-tail:].mean().item()
                    scores.append(avg_score)

            if scores:   # 至少有一次有效运行
                x_vals.append(p[key_plot])
                y_means.append(np.mean(scores))
                y_stds.append(np.std(scores))

        if not x_vals:
            print(f"Warning: No valid data for mode {mode}")
            continue

        # 按关键参数值排序，方便绘制
        sorted_idx = np.argsort(x_vals)
        x_sorted = np.array(x_vals)[sorted_idx]
        y_mean_sorted = np.array(y_means)[sorted_idx]
        y_std_sorted = np.array(y_stds)[sorted_idx]

        # 绘制误差条
        ax.errorbar(x_sorted, y_mean_sorted, yerr=y_std_sorted,
                    fmt='o', capsize=3, alpha=0.6, label='Mean ± std')

        # 标记最优（均值最小）点
        best_idx = np.argmin(y_mean_sorted)
        ax.scatter(x_sorted[best_idx], y_mean_sorted[best_idx],
                   color='red', s=100, label='Optimal')

        ax.set_xlabel(axis_labels[mode])
        ax.set_ylabel('Validation MMD$^2$')
        ax.set_title(f'Sensitivity of {mode} to {axis_labels[mode]}')
        ax.legend()
        ax.grid(True)

        fig.tight_layout()
        fig.savefig(f"./results/sensitivity_{mode}.png", dpi=300)
        print(f"Saved sensitivity plot for {mode}")
        plt.close(fig)

    # ---------------------------------------------
    # 以下部分（收敛曲线、显著性检验）保持不变
    # ---------------------------------------------
    modes = ['pro', 'underdamped_pro', 'vgd', 'fr_mirror', 'wfr_smc', 'wfr_bdl']
    best_params_dict = {r['Algorithm']: r['Best_Params'] for r in results_list}

    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    ax_nll, ax_mmd = axes[0], axes[1]
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

    final_mmd_data = {m: [] for m in modes}

    for idx, mode in enumerate(modes):
        params = base_params.copy()
        params.update(best_params_dict[mode])

        all_nll, all_mmd = [], []
        steps_np = None
        for run in range(n_repeats):
            torch.manual_seed(42 + run * 10)
            theta_init = torch.randn(params['p'], X_train.shape[1], device=device)
            batch_indices = torch.randint(0, X_train.shape[0], (K_opt, params['batch_size']), device=device)
            result = run_experiment(mode, X_train, y_train, X_val, y_val, params, K_opt,
                                    theta_init=theta_init, batch_indices=batch_indices)
            if result is None:
                continue
            steps, nll, mmd = result
            if steps_np is None:
                steps_np = steps.cpu().numpy()
            all_nll.append(nll.cpu().numpy())
            all_mmd.append(mmd.cpu().numpy())

        if len(all_nll) == 0:
            continue

        nll_arr = np.array(all_nll)
        mmd_arr = np.array(all_mmd)
        nll_mean = nll_arr.mean(axis=0)
        nll_std = nll_arr.std(axis=0)
        mmd_mean = mmd_arr.mean(axis=0)
        mmd_std = mmd_arr.std(axis=0)

        final_mmd_data[mode] = mmd_arr[:, -1]

        ax_nll.plot(steps_np, nll_mean, color=colors[idx], label=mode)
        ax_nll.fill_between(steps_np, nll_mean - nll_std, nll_mean + nll_std,
                            color=colors[idx], alpha=0.2)
        ax_mmd.plot(steps_np, mmd_mean, color=colors[idx], label=mode)
        ax_mmd.fill_between(steps_np, mmd_mean - mmd_std, mmd_mean + mmd_std,
                            color=colors[idx], alpha=0.2)

    ax_nll.set_xlabel('Iteration step')
    ax_nll.set_ylabel('Validation NLL')
    ax_nll.legend()
    ax_nll.grid(True)
    ax_nll.set_title('Validation NLL with 1σ confidence interval')

    ax_mmd.set_xlabel('Iteration step')
    ax_mmd.set_ylabel('Validation MMD$^2$')
    ax_mmd.legend()
    ax_mmd.grid(True)
    ax_mmd.set_title('Validation MMD with 1σ confidence interval')

    if 'wfr_smc' in final_mmd_data and 'wfr_bdl' in final_mmd_data:
        if len(final_mmd_data['wfr_smc']) > 1 and len(final_mmd_data['wfr_bdl']) > 1:
            t_stat, p_value = stats.ttest_ind(final_mmd_data['wfr_smc'], final_mmd_data['wfr_bdl'])
            if p_value > 0.05:
                note = (f"Note: Statistical test (p={p_value:.3f}) shows overlapping confidence intervals "
                        f"between wfr_smc and wfr_bdl; their differences in final performance are "
                        f"likely within Monte Carlo variation.")
                print(note)
                ax_mmd.text(0.02, 0.02, note, transform=ax_mmd.transAxes, fontsize=9,
                            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    fig.tight_layout()
    fig.savefig("./results/validation_curves.png", dpi=300)
    print("Saved validation curves to ./results/validation_curves.png")
    plt.close('all')

if __name__ == "__main__":
    perform_grid_search()