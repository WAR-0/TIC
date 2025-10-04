#!/usr/bin/env python3
"""
Parameter Recovery Simulation for TIC Model v4 (Definitive)

Goal
----
Operationalize the "Earn the Complexity" strategy: retain the full 4-parameter TIC
model and demonstrate identifiability under hardened noise using an enriched
experimental design, hierarchical estimation, and strong literature-backed
priors.

Model
-----
    T_s / T_o ≈ [1 + κ·N'^γ] / [λ · (D₀ + D_eff) · Φ']

Key Upgrades Over v3
--------------------
1. Priors-as-penalties (from cognitive anchors) enter both calibration and
   participant-level objectives.
2. Hierarchical estimation: a single group-level D₀ is inferred jointly with
   per-participant λ, κ, γ in a unified optimization.
3. Calibration block strengthened: more trials, reduced noise, and no lapses to
   anchor D₀ and λ cleanly.
4. Retains v3 design fixes: Φ'-independent D_eff mapping, novelty-density
   orthogonalization, Huber losses, and lapse mixture elsewhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

np.random.seed(42)


@dataclass
class ParticipantData:
    D_eff: np.ndarray
    N_prime_obs: np.ndarray
    Phi_prime: np.ndarray
    T_s: np.ndarray


class TICParameterRecoveryV4:
    """Definitive parameter recovery for the 4-parameter TIC model."""

    def __init__(self, n_simulations: int = 1000, n_participants: int = 35):
        self.n_simulations = n_simulations
        self.n_participants = n_participants
        self.T_o = 60.0

        # Parameter bounds for transforms / priors
        self.bounds = {
            "D0": (0.05, 0.20),
            "lambda": (0.5, 3.0),
            "kappa": (0.1, 2.0),
            "gamma": (0.2, 2.0),
        }

        # Priors (cognitive anchors)
        self.prior_means = {
            "D0": 0.12,
            "lambda": 1.2,
            "kappa": 0.7,
            "gamma": 1.0,
        }
        self.prior_scales = {
            "D0": 0.03,
            "lambda": 0.35,   # on log scale
            "kappa": 0.5,     # on log scale
            "gamma": 0.3,
        }
        self.prior_weights = {
            "D0": 80.0,   # strong
            "lambda": 10.0,
            "kappa": 10.0,
            "gamma": 10.0,
        }

        # Experimental design parameters
        self.density_levels = np.array([0.125, 0.3125, 0.5, 0.6875, 0.875])  # Block A
        self.novelty_levels = np.array([0.2, 0.4, 0.6, 0.8])                 # Block B
        self.fixed_density_block_b = 0.25
        self.calibration_density_levels = np.array([0.0, 0.05, 0.15, 0.30, 0.60, 1.00])

        self.n_trials_per_density = 6
        self.n_trials_per_novelty = 6
        self.n_trials_per_calib = 5   # beefed up calibration block

        self.lapse_rate = 0.05
        self.student_t_df = 4
        self.novelty_mismatch_sd = 0.1
        self.calibration_noise_scale = 0.025  # halved noise, no lapses

        self.true_params: List[Dict[str, np.ndarray]] = []
        self.est_params: List[Dict[str, np.ndarray]] = []

        self.delta = 1.0  # Huber threshold

    # ------------------------------------------------------------------
    # Synthetic data generation
    # ------------------------------------------------------------------
    @staticmethod
    def effective_density_map(density: float) -> float:
        """Φ'-independent saturating throughput mapping."""
        s50 = 0.30
        return density / (density + s50)

    @staticmethod
    def sample_truncated_normal(mean: float, sd: float, lower: float, upper: float, size: int) -> np.ndarray:
        values = np.random.normal(mean, sd, size)
        mask = (values < lower) | (values > upper)
        while np.any(mask):
            values[mask] = np.random.normal(mean, sd, mask.sum())
            mask = (values < lower) | (values > upper)
        return values

    def generate_synthetic_data(self) -> Tuple[List[ParticipantData], Dict[str, np.ndarray]]:
        """Simulate participant data under hardened conditions."""
        # Draw true parameters from prior-centered distributions (realistic variability)
        D0_true = self.sample_truncated_normal(
            self.prior_means["D0"], self.prior_scales["D0"], *self.bounds["D0"], size=1
        )[0]
        lambda_true = self.sample_truncated_normal(
            self.prior_means["lambda"], 0.25, *self.bounds["lambda"], size=self.n_participants
        )
        kappa_true = self.sample_truncated_normal(
            self.prior_means["kappa"], 0.18, *self.bounds["kappa"], size=self.n_participants
        )
        gamma_true = self.sample_truncated_normal(
            self.prior_means["gamma"], 0.15, *self.bounds["gamma"], size=self.n_participants
        )

        # Latent trait Φ'
        phi_trait = np.clip(np.random.normal(1.0, 0.15, self.n_participants), 0.5, 1.6)

        participants: List[ParticipantData] = []

        for pid in range(self.n_participants):
            lam = lambda_true[pid]
            kap = kappa_true[pid]
            gam = gamma_true[pid]

            D_eff_vals = []
            N_obs_vals = []
            Phi_vals = []
            Ts_vals = []

            # Block A: density manipulation, novelty sampled independently
            for density in self.density_levels:
                for _ in range(self.n_trials_per_density):
                    phi_state = np.random.normal(0, 0.1)
                    Phi_prime = np.clip(phi_trait[pid] + phi_state, 0.2, 2.0)
                    D_eff = self.effective_density_map(density)
                    N_prime_true = np.clip(np.random.uniform(0.2, 0.8), 0.05, 0.95)
                    N_prime_obs = np.clip(
                        N_prime_true + np.random.normal(0, self.novelty_mismatch_sd), 0.05, 0.95
                    )
                    numerator = 1.0 + kap * (N_prime_true ** gam)
                    denom = lam * (D0_true + D_eff) * Phi_prime
                    T_s_true = self.T_o * numerator / np.maximum(denom, 1e-6)

                    if np.random.rand() < self.lapse_rate:
                        T_s = np.random.uniform(20.0, 100.0)
                    else:
                        noise = np.random.standard_t(self.student_t_df) * (0.07 * self.T_o)
                        T_s = np.clip(T_s_true + noise, 5.0, 120.0)

                    D_eff_vals.append(D_eff)
                    N_obs_vals.append(N_prime_obs)
                    Phi_vals.append(Phi_prime)
                    Ts_vals.append(T_s)

            # Block B: novelty manipulation at fixed density
            D_eff_const = self.effective_density_map(self.fixed_density_block_b)
            for novelty in self.novelty_levels:
                for _ in range(self.n_trials_per_novelty):
                    phi_state = np.random.normal(0, 0.1)
                    Phi_prime = np.clip(phi_trait[pid] + phi_state, 0.2, 2.0)
                    N_prime_true = np.clip(novelty + np.random.normal(0, 0.05), 0.05, 0.95)
                    N_prime_obs = np.clip(
                        N_prime_true + np.random.normal(0, self.novelty_mismatch_sd), 0.05, 0.95
                    )
                    numerator = 1.0 + kap * (N_prime_true ** gam)
                    denom = lam * (D0_true + D_eff_const) * Phi_prime
                    T_s_true = self.T_o * numerator / np.maximum(denom, 1e-6)

                    if np.random.rand() < self.lapse_rate:
                        T_s = np.random.uniform(20.0, 100.0)
                    else:
                        noise = np.random.standard_t(self.student_t_df) * (0.07 * self.T_o)
                        T_s = np.clip(T_s_true + noise, 5.0, 120.0)

                    D_eff_vals.append(D_eff_const)
                    N_obs_vals.append(N_prime_obs)
                    Phi_vals.append(Phi_prime)
                    Ts_vals.append(T_s)

            # Block C: calibration (N' = 0, boosted trials, low noise, no lapses)
            for density_calib in self.calibration_density_levels:
                D_eff = self.effective_density_map(density_calib)
                for _ in range(self.n_trials_per_calib):
                    phi_state = np.random.normal(0, 0.05)
                    Phi_prime = np.clip(phi_trait[pid] + phi_state, 0.2, 2.0)
                    numerator = 1.0
                    denom = lam * (D0_true + D_eff) * Phi_prime
                    T_s_true = self.T_o * numerator / np.maximum(denom, 1e-6)
                    noise = np.random.standard_t(self.student_t_df) * (self.calibration_noise_scale * self.T_o)
                    T_s = np.clip(T_s_true + noise, 10.0, 120.0)

                    D_eff_vals.append(D_eff)
                    N_obs_vals.append(0.0)
                    Phi_vals.append(Phi_prime)
                    Ts_vals.append(T_s)

            participants.append(
                ParticipantData(
                    D_eff=np.array(D_eff_vals, dtype=np.float64),
                    N_prime_obs=np.array(N_obs_vals, dtype=np.float64),
                    Phi_prime=np.array(Phi_vals, dtype=np.float64),
                    T_s=np.array(Ts_vals, dtype=np.float64),
                )
            )

        true_param_dict = {
            "D0": D0_true,
            "lambda": lambda_true,
            "kappa": kappa_true,
            "gamma": gamma_true,
        }
        return participants, true_param_dict

    # ------------------------------------------------------------------
    # Optimization helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def _transform(self, z: np.ndarray, lower: float, upper: float) -> Tuple[np.ndarray, np.ndarray]:
        sig = self._sigmoid(z)
        value = lower + (upper - lower) * sig
        derivative = sig * (1.0 - sig) * (upper - lower)
        return value, derivative

    def _loss_and_grad(self, z: np.ndarray, participants: List[ParticipantData]) -> Tuple[float, np.ndarray]:
        nP = len(participants)
        idx = 0
        D0_z = z[idx]
        idx += 1
        lam_z = z[idx:idx + nP]
        idx += nP
        kap_z = z[idx:idx + nP]
        idx += nP
        gam_z = z[idx:idx + nP]

        D0, dD0_dz = self._transform(D0_z, *self.bounds["D0"])
        lam, dlam_dz = self._transform(lam_z, *self.bounds["lambda"])
        kap, dkap_dz = self._transform(kap_z, *self.bounds["kappa"])
        gam, dgam_dz = self._transform(gam_z, *self.bounds["gamma"])

        total_loss = 0.0
        grad_D0 = 0.0
        grad_lam = np.zeros_like(lam)
        grad_kap = np.zeros_like(kap)
        grad_gam = np.zeros_like(gam)

        for i, pdata in enumerate(participants):
            D_eff = pdata.D_eff
            N_obs = pdata.N_prime_obs
            Phi = pdata.Phi_prime
            Ts = pdata.T_s

            lam_i = lam[i]
            kap_i = kap[i]
            gam_i = gam[i]

            N_is_zero = N_obs <= 1e-12
            safe_N = np.where(N_is_zero, 1e-6, N_obs)
            N_pow = np.where(N_is_zero, 0.0, safe_N ** gam_i)
            logN = np.where(N_is_zero, 0.0, np.log(safe_N))

            numerator = 1.0 + kap_i * N_pow
            denom = lam_i * (D0 + D_eff) * Phi
            denom = np.maximum(denom, 1e-6)
            preds = self.T_o * numerator / denom

            error = Ts - preds
            abs_error = np.abs(error)
            in_quadratic = abs_error <= self.delta

            # Huber loss
            total_loss += 0.5 * np.sum((error[in_quadratic]) ** 2)
            total_loss += self.delta * np.sum(abs_error[~in_quadratic] - 0.5 * self.delta)

            dL_de = np.zeros_like(error)
            dL_de[in_quadratic] = error[in_quadratic]
            dL_de[~in_quadratic] = self.delta * np.sign(error[~in_quadratic])

            # Derivatives of predictions
            denom_common = lam_i * (D0 + D_eff) * Phi
            denom_common = np.maximum(denom_common, 1e-6)

            d_pred_d_lam = -self.T_o * numerator / (lam_i ** 2 * (D0 + D_eff) * Phi)
            d_pred_d_kap = self.T_o * N_pow / denom_common
            d_pred_d_gamma = self.T_o * kap_i * N_pow * logN / denom_common
            d_pred_d_gamma[N_is_zero] = 0.0
            d_pred_d_D0 = -self.T_o * numerator / (lam_i * (D0 + D_eff) ** 2 * Phi)

            grad_lam[i] += -np.sum(dL_de * d_pred_d_lam)
            grad_kap[i] += -np.sum(dL_de * d_pred_d_kap)
            grad_gam[i] += -np.sum(dL_de * d_pred_d_gamma)
            grad_D0 += -np.sum(dL_de * d_pred_d_D0)

        # Priors / penalties
        # D0 penalty (strong)
        d0_pen = ((D0 - self.prior_means["D0"]) / self.prior_scales["D0"]) ** 2
        total_loss += self.prior_weights["D0"] * d0_pen
        grad_D0 += self.prior_weights["D0"] * 2.0 * (D0 - self.prior_means["D0"]) / (self.prior_scales["D0"] ** 2)

        # λ penalties (medium, log scale)
        lam_pen = ((np.log(lam) - np.log(self.prior_means["lambda"])) / self.prior_scales["lambda"]) ** 2
        total_loss += self.prior_weights["lambda"] * np.sum(lam_pen)
        grad_lam += self.prior_weights["lambda"] * 2.0 * (
            (np.log(lam) - np.log(self.prior_means["lambda"])) / (self.prior_scales["lambda"] ** 2 * lam)
        )

        # κ penalties
        kap_pen = ((np.log(kap) - np.log(self.prior_means["kappa"])) / self.prior_scales["kappa"]) ** 2
        total_loss += self.prior_weights["kappa"] * np.sum(kap_pen)
        grad_kap += self.prior_weights["kappa"] * 2.0 * (
            (np.log(kap) - np.log(self.prior_means["kappa"])) / (self.prior_scales["kappa"] ** 2 * kap)
        )

        # γ penalties
        gam_pen = ((gam - self.prior_means["gamma"]) / self.prior_scales["gamma"]) ** 2
        total_loss += self.prior_weights["gamma"] * np.sum(gam_pen)
        grad_gam += self.prior_weights["gamma"] * 2.0 * (
            (gam - self.prior_means["gamma"]) / (self.prior_scales["gamma"] ** 2)
        )

        # Chain rule to z-space
        grad = np.zeros_like(z)
        grad_D0_z = grad_D0 * dD0_dz
        grad[:1] = grad_D0_z
        grad[1:1 + nP] = grad_lam * dlam_dz
        grad[1 + nP:1 + 2 * nP] = grad_kap * dkap_dz
        grad[1 + 2 * nP:] = grad_gam * dgam_dz

        return float(total_loss), grad

    def _adam_optimize(self, participants: List[ParticipantData], verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        nP = len(participants)
        dim = 1 + 3 * nP

        # Initialize at prior means
        def init_val(mean, lower, upper):
            x = (mean - lower) / (upper - lower)
            x = np.clip(x, 1e-4, 1 - 1e-4)
            return np.log(x / (1 - x))

        z0 = np.zeros(dim)
        z0[0] = init_val(self.prior_means["D0"], *self.bounds["D0"])
        lam_start = init_val(self.prior_means["lambda"], *self.bounds["lambda"])
        kap_start = init_val(self.prior_means["kappa"], *self.bounds["kappa"])
        gam_start = init_val(self.prior_means["gamma"], *self.bounds["gamma"])
        z0[1:1 + nP] = lam_start
        z0[1 + nP:1 + 2 * nP] = kap_start
        z0[1 + 2 * nP:] = gam_start

        z = z0.copy()
        m = np.zeros_like(z)
        v = np.zeros_like(z)
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        lr = 0.01

        best_loss = np.inf
        best_z = z.copy()

        for t in range(1, 2001):
            loss, grad = self._loss_and_grad(z, participants)
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            z -= lr * m_hat / (np.sqrt(v_hat) + eps)

            if loss < best_loss:
                best_loss = loss
                best_z = z.copy()

            if verbose and t % 200 == 0:
                grad_norm = np.linalg.norm(grad) / np.sqrt(dim)
                print(f"    iter {t:4d} | loss {loss:10.3f} | grad_norm {grad_norm:7.4f}")
            if np.linalg.norm(grad) / np.sqrt(dim) < 1e-3:
                if verbose:
                    print(f"    converged at iter {t} with grad_norm {np.linalg.norm(grad) / np.sqrt(dim):.4f}")
                break

        return best_z[:1], best_z[1:1 + nP], best_z[1 + nP:1 + 2 * nP], best_z[1 + 2 * nP:]

    def estimate_parameters(self, participants: List[ParticipantData], verbose: bool = False) -> Dict[str, np.ndarray]:
        d0_z, lam_z, kap_z, gam_z = self._adam_optimize(participants, verbose=verbose)

        D0, _ = self._transform(d0_z, *self.bounds["D0"])
        lam, _ = self._transform(lam_z, *self.bounds["lambda"])
        kap, _ = self._transform(kap_z, *self.bounds["kappa"])
        gam, _ = self._transform(gam_z, *self.bounds["gamma"])

        return {
            "D0": D0.item(),
            "lambda": lam,
            "kappa": kap,
            "gamma": gam,
        }

    # ------------------------------------------------------------------
    # Simulation driver & statistics
    # ------------------------------------------------------------------
    def run_all_simulations(self):
        print(f"Running {self.n_simulations} definitive recovery simulations (v4)...")
        total_trials = (
            len(self.density_levels) * self.n_trials_per_density
            + len(self.novelty_levels) * self.n_trials_per_novelty
            + len(self.calibration_density_levels) * self.n_trials_per_calib
        )
        print(f"Participants: {self.n_participants} | Trials per participant: {total_trials}")
        print("Design: calibration emphasis, hierarchical estimation, strong priors.\n")

        for sim in range(self.n_simulations):
            participants, true_params = self.generate_synthetic_data()
            verbose = sim < 3
            est_params = self.estimate_parameters(participants, verbose=verbose)

            self.true_params.append(true_params)
            self.est_params.append(est_params)

            if (sim + 1) % max(self.n_simulations // 10, 1) == 0:
                print(f"  Completed {sim + 1}/{self.n_simulations} simulations...")

        print("\nCompleted all simulations.")

    def _flatten_arrays(self, key: str) -> Tuple[np.ndarray, np.ndarray]:
        if key == "D0":
            true_vals = np.array([tp["D0"] for tp in self.true_params])
            est_vals = np.array([ep["D0"] for ep in self.est_params])
            return true_vals, est_vals
        true_vals = np.concatenate([tp[key] for tp in self.true_params])
        est_vals = np.concatenate([ep[key] for ep in self.est_params])
        return true_vals, est_vals

    def calculate_statistics(self) -> Dict[str, Dict[str, float]]:
        params = ["D0", "lambda", "kappa", "gamma", "rho"]
        stats = {}

        print("\n" + "=" * 90)
        print("Hierarchical Recovery Statistics (v4)")
        print("=" * 90)
        print(f"{'Parameter':<10} {'r':>8} {'Bias':>10} {'RelBias%':>10} {'RMSE':>10} {'MAE':>10}")
        print("-" * 90)

        for param in params:
            if param == "rho":
                true_lambda, est_lambda = self._flatten_arrays("lambda")
                true_kappa, est_kappa = self._flatten_arrays("kappa")
                true_vals = true_kappa / true_lambda
                est_vals = est_kappa / est_lambda
                rho_min = self.bounds["kappa"][0] / self.bounds["lambda"][1]
                rho_max = self.bounds["kappa"][1] / self.bounds["lambda"][0]
                param_range = rho_max - rho_min
            else:
                true_vals, est_vals = self._flatten_arrays(param)
                param_range = self.bounds[param][1] - self.bounds[param][0]

            if np.allclose(true_vals, true_vals[0]):
                corr = float("nan")
            else:
                corr = np.corrcoef(true_vals, est_vals)[0, 1]
            bias = float(np.mean(est_vals - true_vals))
            rel_bias_pct = 100.0 * bias / param_range
            rmse = float(np.sqrt(np.mean((est_vals - true_vals) ** 2)))
            mae = float(np.mean(np.abs(est_vals - true_vals)))

            stats[param] = {
                "r": corr,
                "bias": bias,
                "rel_bias_pct": rel_bias_pct,
                "rmse": rmse,
                "mae": mae,
            }
            corr_disp = f"{corr:8.3f}" if not np.isnan(corr) else "   nan"
            print(f"{param:<10}{corr_disp}{bias:10.3f}{rel_bias_pct:10.1f}{rmse:10.3f}{mae:10.3f}")

        print("=" * 90)
        return stats

    @staticmethod
    def _write_results(path: str, stats: Dict[str, Dict[str, float]], n_sim: int, n_participants: int, trials_per_participant: int):
        lines = []
        lines.append("TIC Parameter Recovery v4 (Definitive)")
        lines.append("=" * 90)
        lines.append(f"Simulations       : {n_sim}")
        lines.append(f"Participants      : {n_participants}")
        lines.append(f"Trials/participant: {trials_per_participant}")
        lines.append("")
        lines.append("Parameter Table")
        lines.append("-" * 90)
        header = f"{'Parameter':<10} {'r':>8} {'Bias':>10} {'RelBias%':>10} {'RMSE':>10} {'MAE':>10}"
        lines.append(header)
        lines.append("-" * 90)
        for param, values in stats.items():
            r = values["r"]
            r_str = f"{r:8.3f}" if not np.isnan(r) else "   nan"
            row = f"{param:<10}{r_str}{values['bias']:10.3f}{values['rel_bias_pct']:10.1f}{values['rmse']:10.3f}{values['mae']:10.3f}"
            lines.append(row)
        lines.append("-" * 90)
        lines.append("Notes:")
        lines.append("  • Hierarchical estimation of a shared D₀ with participant-level λ, κ, γ.")
        lines.append("  • Calibration block: 6 density levels × 5 trials, zero lapses, 2.5% noise.")
        lines.append("  • Priors (λ, κ, γ, D₀) enforced via quadratic penalties grounded in literature.")
        lines.append("  • Retains Φ'-independent D_eff and Huber loss with lapse mixture noise elsewhere.")
        lines.append("")

        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))


def main():
    n_sims = int(os.getenv("N_SIM", "1000"))
    n_participants = int(os.getenv("N_PARTICIPANTS", "35"))
    simulator = TICParameterRecoveryV4(n_simulations=n_sims, n_participants=n_participants)
    simulator.run_all_simulations()
    stats = simulator.calculate_statistics()

    trials_per_participant = (
        len(simulator.density_levels) * simulator.n_trials_per_density
        + len(simulator.novelty_levels) * simulator.n_trials_per_novelty
        + len(simulator.calibration_density_levels) * simulator.n_trials_per_calib
    )
    results_path = os.path.join(os.path.dirname(__file__), "results_v4.txt")
    simulator._write_results(results_path, stats, simulator.n_simulations, simulator.n_participants, trials_per_participant)
    print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    main()
