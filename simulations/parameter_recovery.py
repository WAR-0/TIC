#!/usr/bin/env python3
"""
Parameter Recovery Simulation for TIC Model v3 (Definitive)

Objective: Demonstrate that the 4-parameter TIC model is identifiable under
hardened conditions once the calibration block and design fixes are in place.

Model:
    T_s / T_o ≈ [1 + κ·N'^γ] / [λ · (D₀ + D_eff) · Φ']

Hardened conditions retained from v2:
1. Latent trait-state Φ'
2. Lapse-mixture Student-t noise
3. Novelty mis-specification

Key fixes implemented in v3:
1. Calibration Block C with N' = 0 and a grid of D_eff (including 0) to break
   the λ–D₀ ridge.
2. Φ'-independent D_eff mapping so the denominator no longer contains Φ'^2.
3. Novelty samples independent of density in Block A, reducing numerator–denominator coupling.
4. Robust two-stage estimation: Stage 1 recovers λ and D₀ from Block C via
   Huber regression; Stage 2 recovers κ and γ with λ, D₀ fixed and weak priors.
"""

import os
from typing import Dict

import numpy as np

np.random.seed(42)


class SimpleOptimizer:
    """Minimal differential evolution optimizer (NumPy only)."""

    def __init__(self, bounds, popsize=20, maxiter=150):
        self.bounds = np.array(bounds)
        self.popsize = popsize
        self.maxiter = maxiter
        self.dim = len(bounds)

    def optimize(self, objective_func):
        pop = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.popsize, self.dim))
        fitness = np.array([objective_func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best_solution = pop[best_idx].copy()
        best_fitness = fitness[best_idx]
        F, CR = 0.8, 0.9

        for _ in range(self.maxiter):
            for i in range(self.popsize):
                indices = [idx for idx in range(self.popsize) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = np.clip(pop[a] + F * (pop[b] - pop[c]), self.bounds[:, 0], self.bounds[:, 1])
                trial = np.where(np.random.rand(self.dim) < CR, mutant, pop[i])
                trial_fitness = objective_func(trial)
                if trial_fitness < fitness[i]:
                    pop[i], fitness[i] = trial, trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution, best_fitness = trial.copy(), trial_fitness
        return best_solution


def huber_line_regression(x: np.ndarray, y: np.ndarray, delta: float = 1.0, iters: int = 8):
    """Iteratively reweighted least squares fit for a robust line."""

    x = np.asarray(x)
    y = np.asarray(y)
    design = np.c_[np.ones_like(x), x]
    weights = np.ones_like(y)

    for _ in range(max(iters, 1)):
        wdesign = design * weights[:, None]
        wy = y * weights
        beta, *_ = np.linalg.lstsq(wdesign, wy, rcond=None)
        residuals = y - (beta[0] + beta[1] * x)
        abs_resid = np.abs(residuals)
        weights = np.where(abs_resid <= delta, 1.0, delta / np.maximum(abs_resid, 1e-8))

    return float(beta[1]), float(beta[0])  # slope, intercept


class TICParameterRecoveryV3:
    """Definitive parameter recovery simulation for the revised TIC model."""

    def __init__(self, n_simulations: int = 1000, n_participants: int = 35):
        self.n_simulations = n_simulations
        self.n_participants = n_participants
        self.T_o = 60.0
        self.param_ranges = {
            "lambda": (0.8, 2.5),
            "kappa": (0.3, 1.5),
            "gamma": (0.5, 1.5),
            "D0": (0.05, 0.2),
        }

        # Experimental design
        self.density_levels = np.array([0.125, 0.3125, 0.5, 0.6875, 0.875])
        self.n_trials_per_density = 6
        self.novelty_levels = np.array([0.2, 0.4, 0.6, 0.8])
        self.n_trials_per_novelty = 6
        self.fixed_density_block_b = 0.25
        self.calibration_density_levels = np.array([0.0, 0.05, 0.15, 0.3, 0.6, 1.0])
        self.n_trials_per_calib = 3
        self.n_trials = (
            len(self.density_levels) * self.n_trials_per_density
            + len(self.novelty_levels) * self.n_trials_per_novelty
            + len(self.calibration_density_levels) * self.n_trials_per_calib
        )

        # Hardened noise / mis-specification
        self.lapse_rate = 0.05
        self.student_t_df = 4
        self.novelty_mismatch_factor = 0.1

        self.true_params, self.est_params = [], []

    @staticmethod
    def effective_density_map(density: float) -> float:
        """Φ'-independent saturating throughput."""
        s50 = 0.30
        return density / (density + s50)

    def tic_model(self, D_eff, N_prime, Phi_prime, lam, kappa, gamma, D0):
        numerator = 1.0 + kappa * (N_prime ** gamma)
        denominator = lam * (D0 + D_eff) * Phi_prime
        return self.T_o * (numerator / np.maximum(denominator, 1e-6))

    def generate_synthetic_data(self, true_params: Dict) -> Dict:
        lam, kappa, gamma, D0 = [true_params[k] for k in ("lambda", "kappa", "gamma", "D0")]
        phi_trait = np.random.normal(1.0, 0.15, self.n_participants)
        data = {"D_eff": [], "N_prime_obs": [], "Phi_prime": [], "T_s": [], "block": []}

        for pid in range(self.n_participants):
            # Block A: density sweeps, novelty sampled independently
            for density in self.density_levels:
                for _ in range(self.n_trials_per_density):
                    phi_state = np.random.normal(0, 0.1)
                    Phi_prime = np.clip(phi_trait[pid] + phi_state, 0.2, 2.0)
                    D_eff = self.effective_density_map(density)
                    N_prime_true = np.clip(np.random.uniform(0.2, 0.8), 0.05, 0.95)
                    N_prime_obs = np.clip(
                        N_prime_true + np.random.normal(0, self.novelty_mismatch_factor), 0.05, 0.95
                    )
                    T_s_true = self.tic_model(D_eff, N_prime_true, Phi_prime, lam, kappa, gamma, D0)
                    if np.random.rand() < self.lapse_rate:
                        T_s = np.random.uniform(20.0, 100.0)
                    else:
                        noise = np.random.standard_t(self.student_t_df) * (0.07 * self.T_o)
                        T_s = np.clip(T_s_true + noise, 5.0, 120.0)
                    data["D_eff"].append(D_eff)
                    data["N_prime_obs"].append(N_prime_obs)
                    data["Phi_prime"].append(Phi_prime)
                    data["T_s"].append(T_s)
                    data["block"].append("A")

            # Block B: novelty sweeps at fixed density
            for novelty in self.novelty_levels:
                for _ in range(self.n_trials_per_novelty):
                    phi_state = np.random.normal(0, 0.1)
                    Phi_prime = np.clip(phi_trait[pid] + phi_state, 0.2, 2.0)
                    D_eff = self.effective_density_map(self.fixed_density_block_b)
                    N_prime_true = np.clip(novelty + np.random.normal(0, 0.05), 0.05, 0.95)
                    N_prime_obs = np.clip(
                        N_prime_true + np.random.normal(0, self.novelty_mismatch_factor), 0.05, 0.95
                    )
                    T_s_true = self.tic_model(D_eff, N_prime_true, Phi_prime, lam, kappa, gamma, D0)
                    if np.random.rand() < self.lapse_rate:
                        T_s = np.random.uniform(20.0, 100.0)
                    else:
                        noise = np.random.standard_t(self.student_t_df) * (0.07 * self.T_o)
                        T_s = np.clip(T_s_true + noise, 5.0, 120.0)
                    data["D_eff"].append(D_eff)
                    data["N_prime_obs"].append(N_prime_obs)
                    data["Phi_prime"].append(Phi_prime)
                    data["T_s"].append(T_s)
                    data["block"].append("B")

            # Block C: calibration trials (N' = 0)
            for density_calib in self.calibration_density_levels:
                for _ in range(self.n_trials_per_calib):
                    phi_state = np.random.normal(0, 0.1)
                    Phi_prime = np.clip(phi_trait[pid] + phi_state, 0.2, 2.0)
                    D_eff = self.effective_density_map(density_calib)
                    T_s_true = self.tic_model(D_eff, 0.0, Phi_prime, lam, kappa, gamma, D0)
                    if np.random.rand() < self.lapse_rate:
                        T_s = np.random.uniform(20.0, 100.0)
                    else:
                        noise = np.random.standard_t(self.student_t_df) * (0.05 * self.T_o)
                        T_s = np.clip(T_s_true + noise, 10.0, 120.0)
                    data["D_eff"].append(D_eff)
                    data["N_prime_obs"].append(0.0)
                    data["Phi_prime"].append(Phi_prime)
                    data["T_s"].append(T_s)
                    data["block"].append("C")

        return {key: np.array(val) for key, val in data.items()}

    def estimate_parameters(self, data: Dict) -> Dict:
        """Two-stage estimator using calibration trials to anchor λ and D₀."""

        calib_mask = data["block"] == "C"
        y = (self.T_o / data["T_s"][calib_mask]) / data["Phi_prime"][calib_mask]
        x = data["D_eff"][calib_mask]

        slope, intercept = huber_line_regression(x, y)
        if slope <= 0 or intercept <= 0:
            slope, intercept = 1.5, 0.18  # conservative fallback

        lam_est = np.clip(slope, 0.5, 3.0)
        D0_est = np.clip(intercept / lam_est, 0.01, 0.5)

        all_D_eff = data["D_eff"]
        all_N_obs = data["N_prime_obs"]
        all_Phi = data["Phi_prime"]
        all_Ts = data["T_s"]

        def novelty_objective(params):
            kappa, gamma = params
            if kappa <= 0 or gamma <= 0:
                return 1e12
            predictions = self.tic_model(all_D_eff, all_N_obs, all_Phi, lam_est, kappa, gamma, D0_est)
            error = all_Ts - predictions
            delta = 1.0
            huber_loss = np.sum(
                np.where(np.abs(error) < delta, 0.5 * error**2, delta * (np.abs(error) - 0.5 * delta))
            )
            kap_pen = ((np.log(kappa) - np.log(0.7)) / 0.6) ** 2
            gam_pen = ((gamma - 1.0) / 0.35) ** 2
            return huber_loss + 5.0 * (kap_pen + gam_pen)

        bounds_novelty = [(0.1, 2.0), (0.2, 2.0)]
        optimizer = SimpleOptimizer(bounds=bounds_novelty, popsize=20, maxiter=150)
        kappa_est, gamma_est = optimizer.optimize(novelty_objective)

        return {"lambda": lam_est, "kappa": kappa_est, "gamma": gamma_est, "D0": D0_est}

    def run_all_simulations(self):
        print(f"Running {self.n_simulations} definitive recovery simulations (v3)...")
        print(f"N = {self.n_participants} participants, {self.n_trials} trials each.")
        print("Design includes Calibration Block and critical decoupling fixes.\n")

        for sim_id in range(self.n_simulations):
            true_params = {p: np.random.uniform(*r) for p, r in self.param_ranges.items()}
            data = self.generate_synthetic_data(true_params)
            est_params = self.estimate_parameters(data)
            self.true_params.append(true_params)
            self.est_params.append(est_params)
            if (sim_id + 1) % max(int(self.n_simulations / 10), 1) == 0:
                print(f"  Completed {sim_id + 1}/{self.n_simulations} simulations...")

        print(f"\nCompleted all {self.n_simulations} simulations.")

    def calculate_statistics(self):
        params_order = ["lambda", "kappa", "gamma", "D0", "rho"]
        print("\n" + "=" * 80)
        print("Definitive Recovery Statistics (v3 with all fixes):")
        print("=" * 80)
        print(f"{'Parameter':<10} {'Recovery r':<12} {'Bias':<10} {'Rel.Bias%':<12} {'RMSE':<10} {'MAE':<10}")
        print("-" * 80)

        stats = {}
        for param in params_order:
            if param == "rho":
                true_vals = np.array([tp["kappa"] / tp["lambda"] for tp in self.true_params])
                est_vals = np.array([ep["kappa"] / ep["lambda"] for ep in self.est_params])
                param_range = (
                    self.param_ranges["kappa"][0] / self.param_ranges["lambda"][1]
                    - self.param_ranges["kappa"][1] / self.param_ranges["lambda"][0]
                )
            else:
                true_vals = np.array([tp[param] for tp in self.true_params])
                est_vals = np.array([ep[param] for ep in self.est_params])
                param_range = self.param_ranges[param][1] - self.param_ranges[param][0]

            corr = np.corrcoef(true_vals, est_vals)[0, 1]
            bias = np.mean(est_vals - true_vals)
            rel_bias_pct = 100 * bias / abs(param_range)
            rmse = np.sqrt(np.mean((est_vals - true_vals) ** 2))
            mae = np.mean(np.abs(est_vals - true_vals))

            stats[param] = {
                "r": corr,
                "bias": bias,
                "rel_bias_pct": rel_bias_pct,
                "rmse": rmse,
                "mae": mae,
            }
            print(f"{param:<10} {corr:<12.3f} {bias:<10.3f} {rel_bias_pct:<12.1f} {rmse:<10.3f} {mae:<10.3f}")

        print("=" * 80)
        return stats, params_order

    def save_results(self, stats, order, filepath):
        lines = []
        lines.append("TIC Parameter Recovery v3")
        lines.append("=" * 80)
        lines.append(f"Simulations      : {self.n_simulations}")
        lines.append(f"Participants     : {self.n_participants}")
        lines.append(f"Trials/participant: {self.n_trials}")
        lines.append("")
        lines.append("Parameter Table")
        lines.append("-" * 80)
        header = f"{'Parameter':<10} {'Recovery r':<12} {'Bias':<10} {'Rel.Bias%':<12} {'RMSE':<10} {'MAE':<10}"
        lines.append(header)
        lines.append("-" * 80)
        for param in order:
            s = stats[param]
            row = f"{param:<10} {s['r']:<12.3f} {s['bias']:<10.3f} {s['rel_bias_pct']:<12.1f} {s['rmse']:<10.3f} {s['mae']:<10.3f}"
            lines.append(row)
        lines.append("-" * 80)
        lines.append("Notes:")
        lines.append("  • Calibration Block C (N'=0) anchors λ and D₀.")
        lines.append("  • D_eff uses Φ'-independent saturating mapping with s50 = 0.30.")
        lines.append("  • Stage 2 employs Huber loss with weak priors on κ and γ.")
        lines.append("")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def main():
    n_sims = int(os.getenv("N_SIM", "1000"))
    simulator = TICParameterRecoveryV3(n_simulations=n_sims, n_participants=35)
    simulator.run_all_simulations()
    stats, order = simulator.calculate_statistics()

    results_path = os.path.join(os.path.dirname(__file__), "results_v3.txt")
    simulator.save_results(stats, order, results_path)

    print("\nSimulation complete. The statistics above should be used to update Appendix A.")
    print("Results written to:", results_path)


if __name__ == "__main__":
    main()
