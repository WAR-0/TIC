#!/usr/bin/env python3
"""
TIC Parameter Recovery Simulation v9 (Gold-Standard Validation)

- Hierarchical PyMC/NUTS model (as in v8).
- Two-block "bookend" design:
    * Structured counting-task anchor block (stabilises D₀).
    * High-power 3×6 factorial block with perceptually spaced novelty levels (identifies γ).
- Empirically grounded Ex-Gaussian noise (≈20% CV overall).
- Weakly informative priors reflecting dossier consensus.

Default CLI values run a fast smoke test. For publication-grade validation use:
  python simulations/parameter_recovery_v9_gold_standard.py \
      --n-sim 200 --n-participants 35 --draws 1500 --tune 2000 --target-accept 0.98
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm


@dataclass
class ParticipantData:
    D_eff: np.ndarray
    N_obs: np.ndarray
    Phi: np.ndarray
    Ts: np.ndarray


class TICParameterRecoveryV9:
    """Hierarchical Bayesian parameter recovery for the TIC model."""

    def __init__(
        self,
        n_simulations: int = 5,
        n_participants: int = 20,
        seed: int = 42,
        draws: int = 1000,
        tune: int = 1000,
    ) -> None:
        self.n_simulations = n_simulations
        self.n_participants = n_participants
        self.draws = draws
        self.tune = tune
        self.rng = np.random.default_rng(seed)

        self.T_o = 60.0

        self.hyper_means = {
            "D0": 0.15,
            "lambda": np.log(1.0),
            "kappa": np.log(0.2),
            "gamma": 0.8,
        }
        self.bounds = {
            "D0": (0.05, 0.30),
            "lambda": (0.4, 2.5),
            "kappa": (0.05, 1.5),
            "gamma": (0.3, 2.0),
        }

        # Ex-Gaussian noise parameters (≈20% CV total)
        self.gaussian_sd = 0.12 * self.T_o
        self.exponential_tau = 0.08 * self.T_o
        self.lapse_rate = 0.05

        # Counting-task anchor block
        self.anchor_density = 0.15
        self.anchor_novelty = 0.05
        self.n_trials_anchor = 25

        # High-power factorial block (3×6×20 = 360 trials)
        self.factorial_densities = np.array([0.2, 0.5, 0.8])
        self.factorial_novelties = np.array([0.10, 0.25, 0.45, 0.65, 0.80, 0.95])
        self.n_trials_per_cell = 20

        self.true_params: List[Dict[str, np.ndarray]] = []
        self.posterior_means: List[Dict[str, np.ndarray]] = []

    @staticmethod
    def effective_density_map(density: float) -> float:
        s50 = 0.30
        return density / (density + s50)

    def tic_model(self, D_eff: float, N_prime: float, Phi_prime: float,
                  lam: float, kap: float, gam: float, D0: float) -> float:
        numerator = 1.0 + kap * (N_prime ** gam)
        denom = lam * (D0 + D_eff) * Phi_prime
        denom = max(denom, 1e-6)
        return self.T_o * numerator / denom

    def sample_truncated(self, mean: float, sd: float, lower: float, upper: float, size: Tuple[int, ...]) -> np.ndarray:
        values = self.rng.normal(mean, sd, size)
        mask = (values < lower) | (values > upper)
        while np.any(mask):
            values[mask] = self.rng.normal(mean, sd, mask.sum())
            mask = (values < lower) | (values > upper)
        return values

    def generate_synthetic_data(self) -> Tuple[List[ParticipantData], Dict[str, np.ndarray]]:
        nP = self.n_participants

        true_mu_D0 = np.clip(self.rng.normal(self.hyper_means["D0"], 0.08), 0.07, 0.23)
        true_sigma_D0 = self.rng.lognormal(np.log(0.05), 0.4)

        true_mu_lambda = self.rng.normal(self.hyper_means["lambda"], 0.4)
        true_sigma_lambda = self.rng.lognormal(np.log(0.3), 0.3)

        true_mu_kappa = self.rng.normal(self.hyper_means["kappa"], 0.5)
        true_sigma_kappa = self.rng.lognormal(np.log(0.4), 0.3)

        true_mu_gamma = np.clip(self.rng.normal(self.hyper_means["gamma"], 0.4), 0.4, 1.4)
        true_sigma_gamma = self.rng.lognormal(np.log(0.3), 0.3)

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

            # --- BLOCK ANCHOR: COUNTING TASK ---
            D_eff_anchor = self.effective_density_map(self.anchor_density)
            for _ in range(self.n_trials_anchor):
                Phi_prime = float(np.clip(phi_trait[pid] + self.rng.normal(0.0, 0.05), 0.2, 2.0))
                N_true = float(np.clip(self.rng.normal(self.anchor_novelty, 0.02), 0.01, 0.10))
                N_obs = N_true  # Controlled task: assume measured perfectly

                T_s_true = self.tic_model(D_eff_anchor, N_true, Phi_prime, lam, kap, gam, D0_i)
                noise = self.rng.normal(0.0, self.gaussian_sd * 0.7) + self.rng.exponential(self.exponential_tau * 0.5)
                T_s = float(np.clip(T_s_true + noise, 5.0, 150.0))

                D_eff_vals.append(D_eff_anchor)
                N_obs_vals.append(N_obs)
                Phi_vals.append(Phi_prime)
                Ts_vals.append(T_s)

            # --- BLOCK FACTORIAL (3 × 6 × 20) ---
            for density_f in self.factorial_densities:
                D_eff_f = self.effective_density_map(density_f)
                for novelty_f in self.factorial_novelties:
                    for _ in range(self.n_trials_per_cell):
                        Phi_prime = float(np.clip(phi_trait[pid] + self.rng.normal(0.0, 0.10), 0.2, 2.0))
                        N_true = float(np.clip(novelty_f + self.rng.normal(0.0, 0.05), 0.05, 0.98))
                        N_obs = float(np.clip(N_true + self.rng.normal(0.0, 0.08), 0.05, 0.98))

                        T_s_true = self.tic_model(D_eff_f, N_true, Phi_prime, lam, kap, gam, D0_i)
                        if self.rng.random() < self.lapse_rate:
                            T_s = float(self.rng.uniform(20.0, 140.0))
                        else:
                            noise = self.rng.normal(0.0, self.gaussian_sd) + self.rng.exponential(self.exponential_tau)
                            T_s = float(np.clip(T_s_true + noise, 5.0, 150.0))

                        D_eff_vals.append(D_eff_f)
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

    def build_model(self, participants: List[ParticipantData]) -> pm.Model:
        nP = len(participants)
        D_eff = np.stack([p.D_eff for p in participants], axis=0)
        N_obs = np.stack([p.N_obs for p in participants], axis=0)
        Phi = np.stack([p.Phi for p in participants], axis=0)
        Ts = np.stack([p.Ts for p in participants], axis=0)

        with pm.Model() as model:
            mu_D0 = pm.TruncatedNormal("mu_D0", mu=0.15, sigma=0.08, lower=0.05, upper=0.30)
            sigma_D0 = pm.HalfNormal("sigma_D0", sigma=0.05)

            mu_lambda = pm.Normal("mu_lambda", mu=np.log(1.0), sigma=0.4, shape=1)
            sigma_lambda = pm.HalfNormal("sigma_lambda", sigma=0.3)

            mu_kappa = pm.Normal("mu_kappa", mu=np.log(0.2), sigma=0.5, shape=1)
            sigma_kappa = pm.HalfNormal("sigma_kappa", sigma=0.4)

            mu_gamma = pm.TruncatedNormal("mu_gamma", mu=0.8, sigma=0.4, lower=0.3, upper=2.0)
            sigma_gamma = pm.HalfNormal("sigma_gamma", sigma=0.3)

            sigma_obs = pm.HalfNormal("sigma_obs", sigma=self.gaussian_sd)
            tau_obs = pm.HalfNormal("tau_obs", sigma=self.exponential_tau)

            D0_offset = pm.Normal("D0_offset", mu=0.0, sigma=1.0, shape=nP)
            D0_i = pm.Deterministic("D0_i", mu_D0 + sigma_D0 * D0_offset)

            lambda_offset = pm.Normal("lambda_offset", mu=0.0, sigma=1.0, shape=nP)
            lambda_i = pm.Deterministic("lambda_i", pm.math.exp(mu_lambda + sigma_lambda * lambda_offset))

            kappa_offset = pm.Normal("kappa_offset", mu=0.0, sigma=1.0, shape=nP)
            kappa_i = pm.Deterministic("kappa_i", pm.math.exp(mu_kappa + sigma_kappa * kappa_offset))

            gamma_offset = pm.Normal("gamma_offset", mu=0.0, sigma=1.0, shape=nP)
            gamma_i = pm.Deterministic("gamma_i", mu_gamma + sigma_gamma * gamma_offset)

            D0_clipped = pm.math.clip(D0_i, *self.bounds["D0"])
            lambda_clipped = pm.math.clip(lambda_i, *self.bounds["lambda"])
            kappa_clipped = pm.math.clip(kappa_i, *self.bounds["kappa"])
            gamma_clipped = pm.math.clip(gamma_i, *self.bounds["gamma"])

            numerator = 1.0 + kappa_clipped[:, None] * (N_obs ** gamma_clipped[:, None])
            denom = lambda_clipped[:, None] * (D0_clipped[:, None] + D_eff) * Phi
            denom = pm.math.maximum(denom, 1e-6)
            mu_preds = self.T_o * numerator / denom

            pm.ExGaussian("Ts_obs", mu=mu_preds, sigma=sigma_obs, nu=tau_obs, observed=Ts)

        return model

    def estimate_with_pymc(
        self,
        participants: List[ParticipantData],
        draws: int | None = None,
        tune: int | None = None,
        chains: int = 4,
        target_accept: float = 0.98,
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

    def compute_posterior_means(self, idata: az.InferenceData) -> Dict[str, np.ndarray]:
        posterior = idata.posterior
        return {
            name: np.array(posterior[name].mean(dim=("chain", "draw")))
            for name in ["D0_i", "lambda_i", "kappa_i", "gamma_i"]
        }

    def compute_statistics(
        self,
        true_params: Dict[str, np.ndarray],
        estimated_params: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[str, float]]:
        stats = {}
        for param in ["D0", "lambda", "kappa", "gamma"]:
            true_vals = true_params[param]
            est_vals = estimated_params[f"{param}_i"]
            if np.std(true_vals) < 1e-8 or np.std(est_vals) < 1e-8:
                corr = np.nan
            else:
                corr = np.corrcoef(true_vals, est_vals)[0, 1]
            bias = float(np.mean(est_vals - true_vals))
            rmse = float(np.sqrt(np.mean((est_vals - true_vals) ** 2)))
            mae = float(np.mean(np.abs(est_vals - true_vals)))
            stats[param] = {"r": corr, "bias": bias, "rmse": rmse, "mae": mae}

        true_rho = true_params["kappa"] / true_params["lambda"]
        est_rho = estimated_params["kappa_i"] / estimated_params["lambda_i"]
        if np.std(true_rho) < 1e-8 or np.std(est_rho) < 1e-8:
            corr_rho = np.nan
        else:
            corr_rho = np.corrcoef(true_rho, est_rho)[0, 1]
        stats["rho"] = {
            "r": corr_rho,
            "bias": float(np.mean(est_rho - true_rho)),
            "rmse": float(np.sqrt(np.mean((est_rho - true_rho) ** 2))),
            "mae": float(np.mean(np.abs(est_rho - true_rho))),
        }
        return stats

    def save_diagnostics(self, idata: az.InferenceData, sim_idx: int, output_dir: str = "simulations/diagnostics_v9") -> None:
        os.makedirs(output_dir, exist_ok=True)
        summary = az.summary(
            idata,
            var_names=[
                "mu_D0", "sigma_D0", "mu_lambda", "sigma_lambda",
                "mu_kappa", "sigma_kappa", "mu_gamma", "sigma_gamma",
                "sigma_obs", "tau_obs",
            ],
        )
        summary_file = os.path.join(output_dir, f"sim_{sim_idx:03d}_summary.txt")
        with open(summary_file, "w") as f:
            f.write(summary.to_string())

        rhat_threshold = 1.01
        ess_threshold = 400
        warnings = []
        for param in summary.index:
            rhat = summary.loc[param, "r_hat"]
            ess_bulk = summary.loc[param, "ess_bulk"]
            ess_tail = summary.loc[param, "ess_tail"]
            if rhat > rhat_threshold:
                warnings.append(f"    {param}: R̂={rhat:.4f} (>1.01)")
            if ess_bulk < ess_threshold:
                warnings.append(f"    {param}: ESS_bulk={ess_bulk:.0f} (<400)")
            if ess_tail < ess_threshold:
                warnings.append(f"    {param}: ESS_tail={ess_tail:.0f} (<400)")
        if warnings:
            print(f"  ⚠ Convergence warnings for sim {sim_idx}:")
            for w in warnings:
                print(w)

        fig, axes = plt.subplots(10, 2, figsize=(12, 20))
        az.plot_trace(
            idata,
            var_names=[
                "mu_D0", "sigma_D0", "mu_lambda", "sigma_lambda",
                "mu_kappa", "sigma_kappa", "mu_gamma", "sigma_gamma",
                "sigma_obs", "tau_obs",
            ],
            axes=axes,
        )
        fig.tight_layout()
        trace_file = os.path.join(output_dir, f"sim_{sim_idx:03d}_trace.png")
        fig.savefig(trace_file, dpi=150)
        plt.close(fig)
        print(f"  Diagnostics saved to {output_dir}/sim_{sim_idx:03d}_*")

    def run(self, chains: int = 4, target_accept: float = 0.98) -> None:
        print("Running TIC Parameter Recovery v9 (hierarchical, gold-standard design)")
        print(f"Simulations: {self.n_simulations} | Participants per simulation: {self.n_participants}\n")

        all_stats = []
        for sim in range(self.n_simulations):
            print(f"--- Simulation {sim + 1}/{self.n_simulations} ---")
            participants, true_params = self.generate_synthetic_data()
            idata = self.estimate_with_pymc(participants, chains=chains, target_accept=target_accept)
            self.save_diagnostics(idata, sim)
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
            print(
                f"{param:<10}{vals['r_mean']:10.3f}{vals['r_std']:10.3f}"
                f"{vals['bias_mean']:12.3f}{vals['rmse_mean']:12.3f}{vals['mae_mean']:12.3f}"
            )
        print("=" * 90)


# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hierarchical TIC parameter recovery (v9 gold-standard design)",
        epilog="Full validation: --n-sim 200 --n-participants 35 --draws 1500 --tune 2000 --target-accept 0.98",
    )
    parser.add_argument("--n-sim", type=int, default=5, help="Number of simulations (default: 5; recommend 200 for publication)")
    parser.add_argument("--n-participants", type=int, default=20, help="Participants per simulation (default: 20; recommend 35)")
    parser.add_argument("--draws", type=int, default=1000, help="Posterior draws per chain (default: 1000)")
    parser.add_argument("--tune", type=int, default=1000, help="Tuning steps (default: 1000)")
    parser.add_argument("--chains", type=int, default=4, help="Number of chains (default: 4)")
    parser.add_argument("--target-accept", type=float, default=0.98, help="Target accept rate for NUTS (default: 0.98)")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed (default: 12345)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recovery = TICParameterRecoveryV9(
        n_simulations=args.n_sim,
        n_participants=args.n_participants,
        seed=args.seed,
        draws=args.draws,
        tune=args.tune,
    )
    recovery.run(chains=args.chains, target_accept=args.target_accept)


if __name__ == "__main__":
    main()
