# PrO Posterior Comparison with Particle Flows

## Introduction

This repository provides a set of Python scripts for comparing particle-based algorithms that approximate the **Predictively Oriented (PrO) posterior** introduced in the paper:

> *"Predictively Oriented Posteriors"* – McLatchie et al. (2025), arXiv:2510.01915.

The goal is to solve the nonlinear variational problem:

    minimize_{Q}  F(Q) = λ_n * S_MMD(P_Q) + KL(Q || Π)

where S_MMD is the empirical conditional squared‑MMD score, P_Q is the predictive mixture, and Π is a standard Gaussian prior. The algorithms include:

- PrO (MFLD) – Mean‑Field Langevin Dynamics
- PrO (NULA) – Underdamped Langevin Dynamics (NULA)
- VGD – Variational Gradient Descent
- FR Mirror – Fixed‑support Fisher–Rao mirror descent
- SMC‑WFR – Sequential Monte Carlo approximation of Wasserstein–Fisher–Rao flow
- BDL‑WFR – Birth–Death Langevin
- KDE‑WFR / KDE‑FR – KDE‑based direct discretisations (baselines)
- Bayes MFLD – Standard Bayesian mean‑field Langevin (for comparison)

All code is written in PyTorch and uses GPU acceleration if available. The scripts are designed to be reproducible and include comprehensive experiments on synthetic and real datasets.

## Repository Files

| File | Description |
|------|-------------|
| pro_flows_comparison.py | Original script comparing eight algorithms on the California Housing dataset, with optional grid search. Produces main comparison plots. |
| pro_flows_comparison_improvedV3.py | Extended script with three experiments: (1) Bayes vs. PrO on synthetic data, (2) full algorithm comparison on synthetic data, (3) real dataset with separate hyperparameter search for PrO, SMC‑WFR and Bayes MFLD, including particle norm/gradient norm monitoring. |
| pro_flows_comparison_0816V2.py | Additional experiments (5–8) focusing on misspecification scenarios, particle count sensitivity, initialisation traps, and a low‑dimensional reference benchmark. |
| grid_search_for_paper.py | Systematic grid‑search on the California Housing dataset, producing parameter sensitivity plots, convergence curves, and LaTeX tables for the paper. Includes statistical significance testing between SMC‑WFR and BDL‑WFR. |

## Dependencies

The code requires the following Python packages:

- torch (>= 1.12)
- numpy
- matplotlib
- scikit-learn
- tqdm
- scipy (used in grid_search_for_paper.py)

Install missing packages with:

    pip install torch numpy matplotlib scikit-learn tqdm scipy

## Running the Scripts

Each script is self‑contained and can be run directly from the command line:

    python pro_flows_comparison.py
    python pro_flows_comparison_improvedV3.py
    python pro_flows_comparison_0816V2.py
    python grid_search_for_paper.py

All scripts automatically detect CUDA and use GPU when available. The outputs (figures, tables) are saved in the current working directory (or ./results for grid_search_for_paper.py).

## Key Features and Experiments

### 1. pro_flows_comparison.py
- Loads the California Housing dataset, splits into train/validation/test.
- Estimates noise variance (sigma^2) and MMD kernel bandwidth (gamma^2) from the training data.
- Runs 8 algorithms for 1000 iterations, 20 independent runs.
- Generates the main comparison plot: test NLL and test MMD vs. iteration.

### 2. pro_flows_comparison_improvedV3.py
Performs three extended experiments:

- Experiment 1: Compares Bayes MFLD with PrO MFLD on a synthetic nonlinear dataset, showing predictive NLL curves and particle line plots.
- Experiment 2: Compares all PrO algorithms on synthetic linear data, including trajectories and weight evolution plots.
- Experiment 3: Uses a real dataset (Concrete or Diabetes, falling back to Diabetes) and performs separate hyperparameter searches for PrO, SMC‑WFR and Bayes MFLD. It records particle norms and gradient norms for stability/convergence assessment, and compares against an exact Bayesian posterior (in the linear case). The final figure has four panels: NLL, MMD, particle norm, gradient norm.

### 3. pro_flows_comparison_0816V2.py
Contains four additional experiments (numbered 5–8):

- Experiment 5: Misspecification scenarios (outliers, heteroscedastic noise, nonlinearity) with per‑algorithm hyperparameter tuning.
- Experiment 6: FR Mirror performance as a function of particle count, with eta_w search.
- Experiment 7: Demonstrates the fixed‑support limitation of FR Mirror when initialised far from the true posterior; contrasts with MFLD which moves particles.
- Experiment 8: Low‑dimensional (1D) mixture‑of‑Gaussians benchmark against a high‑precision grid reference, measuring objective gap, gradient difference, and parameter‑space MMD.

### 4. grid_search_for_paper.py
- Performs a comprehensive grid search for each algorithm on California Housing.
- Prints every parameter combination and total number of configurations.
- Saves all results to ./results/all_combos.csv and ./results/best_params.csv.
- Generates parameter sensitivity plots for each algorithm (with mean ± std over repeated runs).
- Produces convergence curves with confidence intervals.
- Performs a t‑test between SMC‑WFR and BDL‑WFR (if both are present) and annotates the plot accordingly.
- Outputs a LaTeX table with optimal parameters and validation MMD.

## Expected Outputs

| Script | Output files |
|--------|--------------|
| pro_flows_comparison.py | pro_standard_flows_mmd.png |
| pro_flows_comparison_improvedV3.py | exp1_bayes_vs_pro.png, exp2_algo_comparison_synthetic.png, exp2_*_trajectory.png, exp2_*_weights.png, exp3_real_dataset_improved_final.png |
| pro_flows_comparison_0816V2.py | exp5_*.png, exp6_fr_particle_count.png, exp7_initialization_failure.png, exp8_low_dim_reference_fast.png, exp8_density_comparison.png |
| grid_search_for_paper.py | ./results/sensitivity_*.png, ./results/validation_curves.png, ./results/all_combos.csv, ./results/best_params.csv |

## Notes

- The scripts assume linear regression (y = Xθ + ε) with Gaussian observation noise.
- The MMD score uses a Gaussian RBF kernel in response space.
- All hyperparameters that define the PrO target (λ_n, γ², σ²) are estimated or fixed and shared across algorithms – only numerical discretisation parameters are tuned per algorithm.
- The KDE‑WFR and KDE‑FR baselines are direct implementations of the formal continuum equations; they are not canonical published algorithms and should be treated as simple baselines.
- Running all scripts may take a long time (especially grid_search_for_paper.py). Adjust num_runs, K, and grid sizes as needed.

## References

The algorithms are based on the following literature (see comments in the code for full details):

- PrO: McLatchie et al., arXiv:2510.01915


## License

This code is provided for research purposes.
