#!/usr/bin/env python3
"""
Parameter Recovery Simulation for TIC Model v5 (Definitive)

This script is the culmination of the evidence-driven revisions for TIC.
It implements:
1. Hierarchical estimation with evidence-aligned priors (broader, uncertainty-aware).
2. Ex-Gaussian noise to model asymmetric attention lapses.
3. No calibration block; design consists only of density (Block A) and novelty (Block B).
4. Optimisation via Adam with logistic reparameterisation of constrained parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

@dataclass
class ParticipantData:
    D_eff: np.ndarray
    N_obs: np.ndarray
    Phi: np.ndarray
    Ts: np.ndarray


class TICParameterRecoveryV5:
    def __init__(self, n_simulations: int = 1000, n_participants: int = 35, seed: int = 42) -> None:
        self.n_simulations = n_simulations
        self.n_participants = n_participants
        self.T_o = 60.0
        self.rng = np.random.default_rng(seed)

        self.bounds = {
            "D0": (0.02, 0.40),
            "lambda": (0.4, 3.0),
            "kappa": (0.05, 2.0),
            "gamma": (0.2, 2.0),
        }

        self.prior_means = {
            "D0": 0.15,
            "lambda": 1.0,
            "kappa": 0.2,
            "gamma": 0.8,
        }
        self.prior_scales = {
            "D0": 0.08,
            "lambda": 0.4,
            "kappa": 0.5,
            "gamma": 0.4,
        }
        self.prior_weights = {
            "D0": 40.0,
            "lambda": 8.0,
            "kappa": 8.0,
            "gamma": 8.0,
        }

        self.density_levels = np.array([0.125, 0.3125, 0.5, 0.6875, 0.875])
        self.low_density_levels = np.array([0.05, 0.15, 0.30])
        self.novelty_levels = np.array([0.2, 0.4, 0.6, 0.8])
        self.fixed_density_block_b = 0.25

        self.n_trials_per_density = 6
        self.n_trials_per_low_density = 5
        self.n_trials_per_novelty = 6

        self.lapse_rate = 0.05
        self.gaussian_noise_sd = 0.04 * self.T_o
        self.exponential_tau = 0.10 * self.T_o
        self.novelty_mismatch_sd = 0.1

        self.true_params: List[Dict[str, np.ndarray]] = []
        self.est_params: List[Dict[str, np.ndarray]] = []

        self.delta = 1.0

    @staticmethod
    def effective_density_map(density: float) -> float:
        s50 = 0.30
        return density / (density + s50)

    def sample_truncated(self, mean: float, sd: float, lower: float, upper: float, size: int) -> np.ndarray:
        values = self.rng.normal(mean, sd, size)
        mask = (values < lower) | (values > upper)
        while np.any(mask):
            values[mask] = self.rng.normal(mean, sd, mask.sum())
            mask = (values < lower) | (values > upper)
        return values

    def generate_synthetic_data(self) -> Tuple[List[ParticipantData], Dict[str, np.ndarray]]:
        nP = self.n_participants

        D0_true = float(self.sample_truncated(0.15, 0.06, *self.bounds["D0"], size=1)[0])
        lambda_true = self.sample_truncated(self.prior_means["lambda"], 0.25, *self.bounds["lambda"], size=nP)
        kappa_true = self.sample_truncated(self.prior_means["kappa"], 0.15, *self.bounds["kappa"], size=nP)
        gamma_true = self.sample_truncated(self.prior_means["gamma"], 0.15, *self.bounds["gamma"], size=nP)

        phi_trait = np.clip(self.rng.normal(1.0, 0.15, nP), 0.5, 1.6)

        participants: List[ParticipantData] = []

        for pid in range(nP):
            lam = lambda_true[pid]
            kap = kappa_true[pid]
            gam = gamma_true[pid]

            D_eff_vals: List[float] = []
            N_obs_vals: List[float] = []
            Phi_vals: List[float] = []
            Ts_vals: List[float] = []

            for density in self.density_levels:
                D_eff = self.effective_density_map(density)
                for _ in range(self.n_trials_per_density):
                    phi_state = self.rng.normal(0.0, 0.10)
                    Phi_prime = float(np.clip(phi_trait[pid] + phi_state, 0.2, 2.0))
                    N_true = float(np.clip(self.rng.uniform(0.2, 0.8), 0.05, 0.95))
                    N_obs = float(np.clip(N_true + self.rng.normal(0.0, self.novelty_mismatch_sd), 0.05, 0.95))

                    numerator = 1.0 + kap * (N_true ** gam)
                    denom = lam * (D0_true + D_eff) * Phi_prime
                    T_s_true = self.T_o * numerator / max(denom, 1e-6)

                    if self.rng.random() < self.lapse_rate:
                        lapse = self.rng.uniform(20.0, 130.0)
                        T_s = float(lapse)
                    else:
                        gaussian_part = self.rng.normal(0.0, self.gaussian_noise_sd)
                        exponential_part = self.rng.exponential(self.exponential_tau)
                        noise = gaussian_part + exponential_part
                        T_s = float(np.clip(T_s_true + noise, 5.0, 150.0))

                    D_eff_vals.append(D_eff)
                    N_obs_vals.append(N_obs)
                    Phi_vals.append(Phi_prime)
                    Ts_vals.append(T_s)

            D_eff_const = self.effective_density_map(self.fixed_density_block_b)
            for novelty in self.novelty_levels:
                for _ in range(self.n_trials_per_novelty):
                    phi_state = self.rng.normal(0.0, 0.10)
                    Phi_prime = float(np.clip(phi_trait[pid] + phi_state, 0.2, 2.0))
                    N_true = float(np.clip(novelty + self.rng.normal(0.0, 0.05), 0.05, 0.95))
                    N_obs = float(np.clip(N_true + self.rng.normal(0.0, self.novelty_mismatch_sd), 0.05, 0.95))
                    numerator = 1.0 + kap * (N_true ** gam)
                    denom = lam * (D0_true + D_eff_const) * Phi_prime
                    T_s_true = self.T_o * numerator / max(denom, 1e-6)

                    if self.rng.random() < self.lapse_rate:
                        T_s = float(self.rng.uniform(20.0, 130.0))
                    else:
                        gaussian_part = self.rng.normal(0.0, self.gaussian_noise_sd)
                        exponential_part = self.rng.exponential(self.exponential_tau)
                        noise = gaussian_part + exponential_part
                        T_s = float(np.clip(T_s_true + noise, 5.0, 150.0))

                    D_eff_vals.append(D_eff_const)
                    N_obs_vals.append(N_obs)
                    Phi_vals.append(Phi_prime)
                    Ts_vals.append(T_s)

            for low_density in self.low_density_levels:
                D_eff_low = self.effective_density_map(low_density)
                for _ in range(self.n_trials_per_low_density):
                    phi_state = self.rng.normal(0.0, 0.08)
                    Phi_prime = float(np.clip(phi_trait[pid] + phi_state, 0.2, 2.0))
                    N_true = float(np.clip(self.rng.normal(0.12, 0.04), 0.01, 0.35))
                    N_obs = float(np.clip(N_true + self.rng.normal(0.0, self.novelty_mismatch_sd / 2.0), 0.01, 0.45))
                    numerator = 1.0 + kap * (N_true ** gam)
                    denom = lam * (D0_true + D_eff_low) * Phi_prime
                    T_s_true = self.T_o * numerator / max(denom, 1e-6)

                    if self.rng.random() < self.lapse_rate:
                        T_s = float(self.rng.uniform(20.0, 130.0))
                    else:
                        gaussian_part = self.rng.normal(0.0, self.gaussian_noise_sd)
                        exponential_part = self.rng.exponential(self.exponential_tau)
                        noise = gaussian_part + exponential_part
                        T_s = float(np.clip(T_s_true + noise, 5.0, 150.0))

                    D_eff_vals.append(D_eff_low)
                    N_obs_vals.append(N_obs)
                    Phi_vals.append(Phi_prime)
                    Ts_vals.append(T_s)

            participants.append(
                ParticipantData(
                    D_eff=np.array(D_eff_vals, dtype=np.float64),
                    N_obs=np.array(N_obs_vals, dtype=np.float64),
                    Phi=np.array(Phi_vals, dtype=np.float64),
                    Ts=np.array(Ts_vals, dtype=np.float64),
                )
            )

        true_params = {
            "D0": D0_true,
            "lambda": lambda_true,
            "kappa": kappa_true,
            "gamma": gamma_true,
        }
        return participants, true_params

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
            N_obs = pdata.N_obs
            Phi = pdata.Phi
            Ts = pdata.Ts

            lam_i = lam[i]
            kap_i = kap[i]
            gam_i = gam[i]

            safe_N = np.clip(N_obs, 1e-6, 1.0)
            N_pow = safe_N ** gam_i
            logN = np.log(safe_N)

            numerator = 1.0 + kap_i * N_pow
            denom = lam_i * (D0 + D_eff) * Phi
            denom = np.maximum(denom, 1e-6)
            preds = self.T_o * numerator / denom

            error = Ts - preds
            abs_error = np.abs(error)
            in_quad = abs_error <= self.delta

            total_loss += 0.5 * np.sum(error[in_quad] ** 2)
            total_loss += self.delta * np.sum(abs_error[~in_quad] - 0.5 * self.delta)

            dL_de = np.zeros_like(error)
            dL_de[in_quad] = error[in_quad]
            dL_departial = self.delta * np.sign(error[~in_quad])
            dL_de[~in_quad] = dL_departial

            denom_common = lam_i * (D0 + D_eff) * Phi
            denom_common = np.maximum(denom_common, 1e-6)

            d_pred_d_lam = -self.T_o * numerator / (lam_i ** 2 * (D0 + D_eff) * Phi)
            d_pred_d_kap = self.T_o * N_pow / denom_common
            d_pred_d_gamma = self.T_o * kap_i * N_pow * logN / denom_common
            d_pred_d_D0 = -self.T_o * numerator / (lam_i * (D0 + D_eff) ** 2 * Phi)

            grad_lam[i] += -np.sum(dL_de * d_pred_d_lam)
            grad_kap[i] += -np.sum(dL_de * d_pred_d_kap)
            grad_gam[i] += -np.sum(dL_de * d_pred_d_gamma)
            grad_D0 += -np.sum(dL_de * d_pred_d_D0)

        d0_pen = ((D0 - self.prior_means["D0"]) / self.prior_scales["D0"]) ** 2
        total_loss += self.prior_weights["D0"] * d0_pen
        grad_D0 += self.prior_weights["D0"] * 2.0 * (D0 - self.prior_means["D0"]) / (self.prior_scales["D0"] ** 2)

        lam_pen = ((np.log(lam) - np.log(self.prior_means["lambda"])) / self.prior_scales["lambda"]) ** 2
        total_loss += self.prior_weights["lambda"] * np.sum(lam_pen)
        grad_lam += self.prior_weights["lambda"] * 2.0 * (
            (np.log(lam) - np.log(self.prior_means["lambda"])) / (self.prior_scales["lambda"] ** 2 * lam)
        )

        kap_pen = ((np.log(kap) - np.log(self.prior_means["kappa"])) / self.prior_scales["kappa"]) ** 2
        total_loss += self.prior_weights["kappa"] * np.sum(kap_pen)
        grad_kap += self.prior_weights["kappa"] * 2.0 * (
            (np.log(kap) - np.log(self.prior_means["kappa"])) / (self.prior_scales["kappa"] ** 2 * kap)
        )

        gam_pen = ((gam - self.prior_means["gamma"]) / self.prior_scales["gamma"]) ** 2
``PY
        total_loss += self.prior_weights["gamma"] * np.sum(gam_pen)
        grad_gam += self.prior_weights["gamma"] * 2.0 * (
            (gam - self.prior_means["gamma"]) / (self.prior_scales["gamma"] ** 2)
        )

        grad = np.zeros_like(z)
        grad[0] = grad_D0 * dD0_dz
        grad[1 : 1 + nP] = grad_lam * dlam_dz
        grad[1 + nP : 1 + 2 * nP] = grad_kap * dkap_dz
        grad[1 + 2 * nP :] = grad_gam * dgam_dz

        return float(total_loss), grad

    def _adam_optimize(
        self,
        participants: List[ParticipantData],
        max_iter: int = 2000,
        lr: float = 0.01,
        tol: float = 1e-3,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        nP = len(participants)
        dim = 1 + 3 * nP

        def init_val(mean: float, lower: float, upper: float) -> float:
            x = (mean - lower) / (upper - lower)
            x = np.clip(x, 1e-4, 1 - 1e-4)
            return np.log(x / (1 - x))

        z = np.zeros(dim)
        z[0] = init_val(self.prior_means["D0"], *self.bounds["D0"])
        z[1 : 1 + nP] = init_val(self.prior_means["lambda"], *self.bounds["lambda"])
        z[1 + nP : 1 + 2 * nP] = init_val(self.prior_means["kappa"], *self.bounds["kappa"])
        z[1 + 2 * nP :] = init_val(self.prior_means["gamma"], *self.bounds["gamma"])

        m = np.zeros_like(z)
        v = np.zeros_like(z)
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8

        best_z = z.copy()
        best_loss = np.inf

        for t in range(1, max_iter + 1):
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

            grad_norm = np.linalg.norm(grad) / np.sqrt(dim)
            if grad_norm < tol:
                if verbose:
                    print(
                        f"    converged at iter {t:4d} | loss {loss:10.3f} | grad_norm {grad_norm:7.4f}"
                    )
                break

        return (
            best_z[:1],
            best_z[1 : 1 + nP],
            best_z[1 + nP : 1 + 2 * nP],
            best_z[1 + 2 * nP :],
        )

    def estimate_parameters(
        self,
        participants: List[ParticipantData],
        verbose: bool = False,
    ) -> Dict[str, np.ndarray]:
        d0_z, lam_z, kap_z, gam_z = self._adam_optimize(participants, verbose=verbose)

        D0, _ = self._transform(d0_z, *self.bounds["D0"])
        lam, _ = self._transform(lam_z, *self.bounds["lambda"])
        kap, _ = self._transform(kap_z, *self.bounds["kappa"])
        gam, _ = self._transform(gam_z, *self.bounds["gamma"])

        return {
            "D0": float(D0),
            "lambda": lam,
            "kappa": kap,
            "gamma": gam,
        }

    def run_all_simulations(self) -> None:
        trials_per_participant = (
            len(self.density_levels) * self.n_trials_per_density
            + len(self.low_density_levels) * self.n_trials_per_low_density
            + len(self.novelty_levels) * self.n_trials_per_novelty
        )
        print(
            f"Running {self.n_simulations} simulations | Participants per sim: {self.n_participants}"
        )
        print(f"Trials per participant: {trials_per_participant}\n")

        for sim in range(self.n_simulations):
            participants, true_params = self.generate_synthetic_data()
            est_params = self.estimate_parameters(participants, verbose=(sim < 3))

            self.true_params.append(true_params)
            self.est_params.append(est_params)

            if (sim + 1) % max(self.n_simulations // 10, 1) == 0:
                print(f"  Completed {sim + 1}/{self.n_simulations} simulations...")

        print("\nSimulations complete.")

    def _stack_param(self, key: str) -> Tuple[np.ndarray, np.ndarray]:
        if key == "D0":
            true_vals = np.array([tp["D0"] for tp in self.true_params])
            est_vals = np.array([ep["D0"] for ep in self.est_params])
        else:
            true_vals = np.concatenate([tp[key] for tp in self.true_params])
            est_vals = np.concatenate([ep[key] for ep in self.est_params])
        return true_vals, est_vals

    def calculate_statistics(self) -> Dict[str, Dict[str, float]]:
        params = ["D0", "lambda", "kappa", "gamma", "rho"]
        stats: Dict[str, Dict[str, float]] = {}

        print("\n" + "=" * 90)
        print("Recovery Statistics (v5)")
        print("=" * 90)
        print(f"{'Param':<10} {'r':>8} {'Bias':>10} {'RelBias%':>10} {'RMSE':>10} {'MAE':>10}")
        print("-" * 90)

        for param in params:
            if param == "rho":
                true_lambda, est_lambda = self._stack_param("lambda")
                true_kappa, est_kappa = self._stack_param("kappa")
                true_vals = true_kappa / true_lambda
                est_vals = est_kappa / est_lambda
                rho_range = self.bounds["kappa"][1] / self.bounds["lambda"][0] - self.bounds["kappa"][0] / self.bounds["lambda"][1]
                param_range = rho_range
            else:
                true_vals, est_vals = self._stack_param(param)
                param_range = self.bounds[param][1] - self.bounds[param][0]

            corr = np.corrcoef(true_vals, est_vals)[0, 1] if np.std(true_vals) > 1e-8 else np.nan
            bias = float(np.mean(est_vals - true_vals))
            rel_bias_pct = float(100.0 * bias / param_range)
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
            print(
                f"{param:<10}{corr_disp}{bias:10.3f}{rel_bias_pct:10.1f}{rmse:10.3f}{mae:10.3f}"
            )

        print("=" * 90)
        return stats

    @staticmethod
    def write_results(path: str, stats: Dict[str, Dict[str, float]], sim_count: int, participants: int, trials_per_participant: int) -> None:
        lines: List[str] = []
        lines.append("TIC Parameter Recovery v5")
        lines.append("=" * 90)
        lines.append(f"Simulations       : {sim_count}")
        lines.append(f"Participants      : {participants}")
        lines.append(f"Trials/participant: {trials_per_participant}")
        lines.append("")
        lines.append("Parameter Table")
        lines.append("-" * 90)
        header = f"{'Parameter':<10} {'r':>8} {'Bias':>10} {'RelBias%':>10} {'RMSE':>10} {'MAE':>10}"
        lines.append(header)
        lines.append("-" * 90)
        for param, vals in stats.items():
            r = vals["r"]
            r_str = f"{r:8.3f}" if not np.isnan(r) else "   nan"
            row = f"{param:<10}{r_str}{vals['bias']:10.3f}{vals['rel_bias_pct']:10.1f}{vals['rmse']:10.3f}{vals['mae']:10.3f}"
            lines.append(row)
        lines.append("-" * 90)
        lines.append("Notes:")
        lines.append("  • Evidence-aligned priors reflecting literature uncertainty.")
        lines.append("  • Ex-Gaussian noise for asymmetric attentional lapses.")
        lines.append("  • Hierarchical shared D₀ with per-participant λ, κ, γ.")
        lines.append("  • Design: density, novelty, and structured low-density blocks only.")
        lines.append("")

        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))


def main() -> None:
    n_sims = int(os.getenv("N_SIM", "200"))
    n_participants = int(os.getenv("N_PARTICIPANTS", "35"))

    simulator = TICParameterRecoveryV5(n_simulations=n_sims, n_participants=n_participants)
    simulator.run_all_simulations()
    stats = simulator.calculate_statistics()

    trials_per_participant = (
        len(simulator.density_levels) * simulator.n_trials_per_density
        + len(simulator.low_density_levels) * simulator.n_trials_per_low_density
        + len(simulator.novelty_levels) * simulator.n_trials_per_novelty
    )
    results_path = os.path.join(os.path.dirname(__file__), "results_v5.txt")
    simulator.write_results(
        results_path,
        stats,
        simulator.n_simulations,
        simulator.n_participants,
        trials_per_participant,
    )
    print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    main()
