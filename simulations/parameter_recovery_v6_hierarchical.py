#!/usr/bin/env python3
"""
Parameter Recovery Simulation for TIC Model v6 (Hierarchical Bayesian)

This script implements the definitive hierarchical Bayesian simulation/estimation
pipeline for the TIC model. Key features:

- Generative model includes group-level (hyper) parameters and participant-level
  parameters for D0, lambda, kappa, gamma.
- Observations (T_s) are modelled using an Ex-Gaussian distribution to capture
  asymmetric attentional lapses.
- Estimation performed with PyMC (NUTS) to fully explore posterior geometry.
- Recovery statistics computed by comparing true participant-level parameters to
  posterior means.

Requirements: PyMC >= 5.x, ArviZ.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import arviz as az
import numpy as np
import pymc as pm


@dataclass
class ParticipantData:
    D_eff: np.ndarray
    N_obs: np.ndarray
    Phi: np.ndarray
    Ts: np.ndarray


class TICParameterRecoveryV6:
    """Hierarchical Bayesian parameter recovery for the TIC model."""

    def __init__(
        self,
        n_simulations: int = 50,
        n_participants: int = 35,
        seed: int = 42,
        draws: int = 1000,
        tune: int = 1000,
    ) -> None:
        self.n_simulations = n_simulations
        self.n_participants = n_participants
        self.draws = draws
        self.tune = tune
        self.rng = np.random.default_rng(seed)

        # Constants for the TIC equation
        self.T_o = 60.0

        # Hyper-prior specifications (population-level beliefs)
        self.hyper_means = {
            "D0": 0.15,
            "lambda": np.log(1.0),  # log space for lognormal
            "kappa": np.log(0.2),
            "gamma": 0.8,
        }
        self.hyper_scales = {
            "D0": 0.08,
            "lambda": 0.4,
            "kappa": 0.5,
            "gamma": 0.4,
        }

        # Prior bounds for individual-level draws (to avoid unrealistic extremes)
        self.bounds = {
            "D0": (0.02, 0.40),
            "lambda": (0.3, 3.0),
            "kappa": (0.02, 2.5),
            "gamma": (0.2, 2.0),
        }

        # Design parameters
        self.density_levels = np.array([0.125, 0.3125, 0.5, 0.6875, 0.875])
        self.low_density_levels = np.array([0.05, 0.15, 0.30])
        self.n_trials_per_density = 6
        self.n_trials_per_low_density = 5

        self.novelty_levels = np.array([0.2, 0.4, 0.6, 0.8])
        self.n_trials_per_novelty = 6
        self.fixed_density_block_b = 0.25

        # Noise model parameters (ex-Gaussian)
        self.gaussian_sd = 0.04 * self.T_o
        self.exponential_tau = 0.10 * self.T_o
        self.lapse_rate = 0.05

        # Placeholders for simulation results
        self.true_params: List[Dict[str, np.ndarray]] = []
        self.posterior_means: List[Dict[str, np.ndarray]] = []

    # ------------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------------
    @staticmethod
    def effective_density_map(density: float) -> float:
        s50 = 0.30
        return density / (density + s50)

    def sample_truncated(self, mean: float, sd: float, lower: float, upper: float, size: Tuple[int, ...]) -> np.ndarray:
        values = self.rng.normal(mean, sd, size)
        mask = (values < lower) | (values > upper)
        while np.any(mask):
            values[mask] = self.rng.normal(mean, sd, mask.sum())
            mask = (values < lower) | (values > upper)
        return values

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------
    def generate_synthetic_data(self) -> Tuple[List[ParticipantData], Dict[str, np.ndarray]]:
        nP = self.n_participants

        # Sample true population hyper-parameters
        true_mu_D0 = self.rng.normal(self.hyper_means["D0"], self.hyper_scales["D0"])
        true_sigma_D0 = self.rng.lognormal(np.log(0.05), 0.4)

        true_mu_lambda = self.rng.normal(self.hyper_means["lambda"], self.hyper_scales["lambda"])
        true_sigma_lambda = self.rng.lognormal(np.log(0.4), 0.3)

        true_mu_kappa = self.rng.normal(self.hyper_means["kappa"], self.hyper_scales["kappa"])
        true_sigma_kappa = self.rng.lognormal(np.log(0.5), 0.3)

        true_mu_gamma = self.rng.normal(self.hyper_means["gamma"], self.hyper_scales["gamma"])
        true_sigma_gamma = self.rng.lognormal(np.log(0.3), 0.3)

        # Draw participant-level parameters
        D0_true = self.sample_truncated(true_mu_D0, true_sigma_D0, *self.bounds["D0"], size=(nP,))
        lambda_true = self.sample_truncated(np.exp(true_mu_lambda), true_sigma_lambda, *self.bounds["lambda"], size=(nP,))
        kappa_true = self.sample_truncated(np.exp(true_mu_kappa), true_sigma_kappa, *self.bounds["kappa"], size=(nP,))
        gamma_true = self.sample_truncated(true_mu_gamma, true_sigma_gamma, *self.bounds["gamma"], size=(nP,))

        phi_trait = np.clip(self.rng.normal(1.0, 0.15, nP), 0.5, 1.6)

        participants: List[ParticipantData] = []

        for pid in range(nP):
            lam = lambda_true[pid]
            kap = kappa_true[pid]
            gam = gamma_true[pid]
            D0_i = D0_true[pid]

            D_eff_vals: List[float] = []
            N_obs_vals: List[float] = []
            Phi_vals: List[float] = []
            Ts_vals: List[float] = []

            # Block A
            for density in self.density_levels:
                D_eff = self.effective_density_map(density)
                for _ in range(self.n_trials_per_density):
                    Phi_prime = float(np.clip(phi_trait[pid] + self.rng.normal(0.0, 0.10), 0.2, 2.0))
                    N_true = float(np.clip(self.rng.uniform(0.2, 0.8), 0.05, 0.95))
                    N_obs = float(np.clip(N_true + self.rng.normal(0.0, 0.1), 0.05, 0.95))

                    numerator = 1.0 + kap * (N_true ** gam)
                    denom = lam * (D0_i + D_eff) * Phi_prime
                    T_s_true = self.T_o * numerator / max(denom, 1e-6)

                    if self.rng.random() < self.lapse_rate:
                        T_s = float(self.rng.uniform(20.0, 130.0))
                    else:
                        noise = self.rng.normal(0.0, self.gaussian_sd) + self.rng.exponential(self.exponential_tau)
                        T_s = float(np.clip(T_s_true + noise, 5.0, 150.0))

                    D_eff_vals.append(D_eff)
                    N_obs_vals.append(N_obs)
                    Phi_vals.append(Phi_prime)
                    Ts_vals.append(T_s)

            # Block B
            D_eff_const = self.effective_density_map(self.fixed_density_block_b)
            for novelty in self.novelty_levels:
                for _ in range(self.n_trials_per_novelty):
                    Phi_prime = float(np.clip(phi_trait[pid] + self.rng.normal(0.0, 0.10), 0.2, 2.0))
                    N_true = float(np.clip(novelty + self.rng.normal(0.0, 0.05), 0.05, 0.95))
                    N_obs = float(np.clip(N_true + self.rng.normal(0.0, 0.1), 0.05, 0.95))

                    numerator = 1.0 + kap * (N_true ** gam)
                    denom = lam * (D0_i + D_eff_const) * Phi_prime
                    T_s_true = self.T_o * numerator / max(denom, 1e-6)

                    if self.rng.random() < self.lapse_rate:
                        T_s = float(self.rng.uniform(20.0, 130.0))
                    else:
                        noise = self.rng.normal(0.0, self.gaussian_sd) + self.rng.exponential(self.exponential_tau)
                        T_s = float(np.clip(T_s_true + noise, 5.0, 150.0))

                    D_eff_vals.append(D_eff_const)
                    N_obs_vals.append(N_obs)
                    Phi_vals.append(Phi_prime)
                    Ts_vals.append(T_s)

            # Low-density block
            for low_density in self.low_density_levels:
                D_eff_low = self.effective_density_map(low_density)
                for _ in range(self.n_trials_per_low_density):
                    Phi_prime = float(np.clip(phi_trait[pid] + self.rng.normal(0.0, 0.08), 0.2, 2.0))
                    N_true = float(np.clip(self.rng.normal(0.12, 0.04), 0.01, 0.35))
                    N_obs = float(np.clip(N_true + self.rng.normal(0.0, 0.05), 0.01, 0.45))

                    numerator = 1.0 + kap * (N_true ** gam)
                    denom = lam * (D0_i + D_eff_low) * Phi_prime
                    T_s_true = self.T_o * numerator / max(denom, 1e-6)

                    if self.rng.random() < self.lapse_rate:
                        T_s = float(self.rng.uniform(20.0, 130.0))
                    else:
                        noise = self.rng.normal(0.0, self.gaussian_sd) + self.rng.exponential(self.exponential_tau)
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

        true_param_dict = {
            "hyper": {
                "mu_D0": true_mu_D0,
                "sigma_D0": true_sigma_D0,
                "mu_lambda": true_mu_lambda,
                "sigma_lambda": true_sigma_lambda,
                "mu_kappa": true_mu_kappa,
                "sigma_kappa": true_sigma_kappa,
                "mu_gamma": true_mu_gamma,
                "sigma_gamma": true_sigma_gamma,
                "sigma_obs": self.gaussian_sd,
                "tau_obs": self.exponential_tau,
            },
            "individual": {
                "D0": D0_true,
                "lambda": lambda_true,
                "kappa": kappa_true,
                "gamma": gamma_true,
            },
        }
        return participants, true_param_dict

    # ------------------------------------------------------------------
    # PyMC model definition
    # ------------------------------------------------------------------
    def build_model(
        self,
        participants: List[ParticipantData],
    ) -> pm.Model:
        nP = len(participants)
        D_eff = np.stack([p.D_eff for p in participants], axis=0)
        N_obs = np.stack([p.N_obs for p in participants], axis=0)
        Phi = np.stack([p.Phi for p in participants], axis=0)
        Ts = np.stack([p.Ts for p in participants], axis=0)

        with pm.Model() as model:
            # Population-level priors
            mu_D0 = pm.Normal("mu_D0", mu=self.hyper_means["D0"], sigma=self.hyper_scales["D0"])
            sigma_D0 = pm.HalfNormal("sigma_D0", sigma=0.10)

            mu_lambda = pm.Normal("mu_lambda", mu=self.hyper_means["lambda"], sigma=self.hyper_scales["lambda"], shape=1)
            sigma_lambda = pm.HalfNormal("sigma_lambda", sigma=0.5)

            mu_kappa = pm.Normal("mu_kappa", mu=self.hyper_means["kappa"], sigma=self.hyper_scales["kappa"], shape=1)
            sigma_kappa = pm.HalfNormal("sigma_kappa", sigma=0.6)

            mu_gamma = pm.Normal("mu_gamma", mu=self.hyper_means["gamma"], sigma=self.hyper_scales["gamma"])
            sigma_gamma = pm.HalfNormal("sigma_gamma", sigma=0.5)

            # Observation noise
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.10 * self.T_o)
            tau_obs = pm.HalfNormal("tau_obs", sigma=0.20 * self.T_o)

            # Participant-level parameters (non-centred parameterisation)
            D0_offset = pm.Normal("D0_offset", mu=0.0, sigma=1.0, shape=nP)
            D0_i = pm.Deterministic("D0_i", mu_D0 + sigma_D0 * D0_offset)

            lambda_offset = pm.Normal("lambda_offset", mu=0.0, sigma=1.0, shape=nP)
            lambda_i = pm.Deterministic("lambda_i", pm.math.exp(mu_lambda + sigma_lambda * lambda_offset))

            kappa_offset = pm.Normal("kappa_offset", mu=0.0, sigma=1.0, shape=nP)
            kappa_i = pm.Deterministic("kappa_i", pm.math.exp(mu_kappa + sigma_kappa * kappa_offset))

            gamma_offset = pm.Normal("gamma_offset", mu=0.0, sigma=1.0, shape=nP)
            gamma_i = pm.Deterministic("gamma_i", mu_gamma + sigma_gamma * gamma_offset)

            D0_clipped = pm.math.clip(D0_i, self.bounds["D0"][0], self.bounds["D0"][1])
            lambda_clipped = pm.math.clip(lambda_i, self.bounds["lambda"][0], self.bounds["lambda"][1])
            kappa_clipped = pm.math.clip(kappa_i, self.bounds["kappa"][0], self.bounds["kappa"][1])
            gamma_clipped = pm.math.clip(gamma_i, self.bounds["gamma"][0], self.bounds["gamma"][1])

            numerator = 1.0 + kappa_clipped[:, None] * pm.math.pow(N_obs, gamma_clipped[:, None])
            denom = lambda_clipped[:, None] * (D0_clipped[:, None] + D_eff) * Phi
            denom = pm.math.maximum(denom, 1e-6)
            mu_preds = self.T_o * numerator / denom

            pm.ExGaussian(
                "Ts_obs",
                mu=mu_preds,
                sigma=sigma_obs,
                nu=tau_obs,
                observed=Ts,
            )

        return model

    # ------------------------------------------------------------------
    # Sampling and recovery
    # ------------------------------------------------------------------
    def estimate_with_pymc(
        self,
        participants: List[ParticipantData],
        draws: int = None,
        tune: int = None,
        chains: int = 2,
        target_accept: float = 0.9,
    ) -> az.InferenceData:
        draws = draws or self.draws
        tune = tune or self.tune
        model = self.build_model(participants)

        with model:
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                progressbar=True,
                random_seed=self.rng.integers(0, 1_000_000),
                compute_convergence_checks=True,
            )
        return idata

    def compute_posterior_means(
        self,
        idata: az.InferenceData,
    ) -> Dict[str, np.ndarray]:
        posterior = idata.posterior
        quants = {}
        for name in ["D0_i", "lambda_i", "kappa_i", "gamma_i"]:
            mean_vals = posterior[name].mean(dim=("chain", "draw"))
            quants[name] = np.array(mean_vals)
        return quants

    def compute_statistics(
        self,
        true_params: Dict[str, np.ndarray],
        estimated_params: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[str, float]]:
        stats = {}
        for param in ["D0", "lambda", "kappa", "gamma"]:
            true_vals = true_params[param]
            est_vals = estimated_params[f"{param}_i"]
            corr = np.corrcoef(true_vals, est_vals)[0, 1]
            bias = float(np.mean(est_vals - true_vals))
            rmse = float(np.sqrt(np.mean((est_vals - true_vals) ** 2)))
            mae = float(np.mean(np.abs(est_vals - true_vals)))
            stats[param] = {
                "r": corr,
                "bias": bias,
                "rmse": rmse,
                "mae": mae,
            }
        true_rho = true_params["kappa"] / true_params["lambda"]
        est_rho = estimated_params["kappa_i"] / estimated_params["lambda_i"]
        corr_rho = np.corrcoef(true_rho, est_rho)[0, 1]
        stats["rho"] = {
            "r": corr_rho,
            "bias": float(np.mean(est_rho - true_rho)),
            "rmse": float(np.sqrt(np.mean((est_rho - true_rho) ** 2))),
            "mae": float(np.mean(np.abs(est_rho - true_rho))),
        }
        return stats

    def run(self, chains: int = 2) -> None:
        print("Running TIC Parameter Recovery v6 (Hierarchical Bayesian)")
        print(f"Simulations: {self.n_simulations} | Participants per simulation: {self.n_participants}\n")

        all_stats = []

        for sim in range(self.n_simulations):
            print(f"--- Simulation {sim + 1}/{self.n_simulations} ---")
            participants, true_params = self.generate_synthetic_data()
            idata = self.estimate_with_pymc(participants, chains=chains)
            est_means = self.compute_posterior_means(idata)
            stats = self.compute_statistics(true_params["individual"], est_means)
            all_stats.append(stats)

            self.true_params.append(true_params)
            self.posterior_means.append(est_means)

            print("Parameter recovery for this simulation:")
            for param, vals in stats.items():
                print(
                    f"  {param:<6} | r = {vals['r']:+.3f} | bias = {vals['bias']:+.3f} | "
                    f"RMSE = {vals['rmse']:.3f} | MAE = {vals['mae']:.3f}"
                )
            print()

        agg_stats = self.aggregate_statistics(all_stats)
        self.report(agg_stats)

    @staticmethod
    def aggregate_statistics(all_stats: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
        params = all_stats[0].keys()
        agg = {}
        for param in params:
            r_vals = np.array([s[param]["r"] for s in all_stats])
            bias_vals = np.array([s[param]["bias"] for s in all_stats])
            rmse_vals = np.array([s[param]["rmse"] for s in all_stats])
            mae_vals = np.array([s[param]["mae"] for s in all_stats])
            agg[param] = {
                "r_mean": float(np.nanmean(r_vals)),
                "r_std": float(np.nanstd(r_vals)),
                "bias_mean": float(np.mean(bias_vals)),
                "bias_std": float(np.std(bias_vals)),
                "rmse_mean": float(np.mean(rmse_vals)),
                "mae_mean": float(np.mean(mae_vals)),
            }
        return agg

    def report(self, agg_stats: Dict[str, Dict[str, float]]) -> None:
        print("\n" + "=" * 90)
        print("Aggregate Recovery Statistics")
        print("=" * 90)
        print(f"{'Param':<10}{'r_mean':>10}{'r_std':>10}{'bias':>12}{'RMSE':>12}{'MAE':>12}")
        print("-" * 90)
        for param, vals in agg_stats.items():
            r_mean = vals['r_mean']
            r_std = vals['r_std']
            bias_mean = vals['bias_mean']
            rmse_mean = vals['rmse_mean']
            mae_mean = vals['mae_mean']
            print(
                f"{param:<10}{r_mean:10.3f}{r_std:10.3f}{bias_mean:12.3f}{rmse_mean:12.3f}{mae_mean:12.3f}"
            )
        print("=" * 90)


# ----------------------------------------------------------------------
# Command-line interface
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hierarchical TIC parameter recovery (v6)")
    parser.add_argument("--n-sim", type=int, default=5, help="Number of simulations (default: 5)")
    parser.add_argument(
        "--n-participants",
        type=int,
        default=18,
        help="Participants per simulation (default: 18)",
    )
    parser.add_argument("--draws", type=int, default=1000, help="Posterior draws per chain (default: 1000)")
    parser.add_argument("--tune", type=int, default=1000, help="Tuning steps (default: 1000)")
    parser.add_argument("--chains", type=int, default=2, help="Number of chains (default: 2)")
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed (default: 12345)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recovery = TICParameterRecoveryV6(
        n_simulations=args.n_sim,
        n_participants=args.n_participants,
        seed=args.seed,
        draws=args.draws,
        tune=args.tune,
    )
    recovery.run(chains=args.chains)


if __name__ == "__main__":
    main()
