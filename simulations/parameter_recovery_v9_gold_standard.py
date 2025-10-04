#!/usr/bin/env python3
"""
TIC Parameter Recovery Simulation v9 (Gold-Standard)

Implements the final, publication-ready design addressing fatigue, noise, and
novelty concerns raised in the Red Team audit. Key features:

1. **Feasible single-session design**
   - Counting-task anchor block (25 trials) for a clean D₀ baseline.
   - High-power factorial block (3 densities × 6 novelty conditions × 8 trials).
   - Total trials per participant: 145.

2. **Gaussian mixture noise model**
   - Mixture of focused (TIC-governed) and lapse distributions.
   - Lapse distribution is compressive (centered on 0.75·T₀).

3. **Principled novelty generation**
   - Novelty computed from stimulus repetitions via
       N' = 1 - exp(-β_N · log(k_reps + 1))
   - Stimulus sequences generated per cell with controllable repetition bias.

4. **Hierarchical PyMC/NUTS estimation**
   - Weakly-informative priors reflecting dossier consensus.
   - Real-time progress prints after each simulation with running aggregates.

Usage:
  # Smoke test (default)
  python simulations/parameter_recovery_v9_gold_standard.py

  # Full validation (watch on-screen)
  python simulations/parameter_recovery_v9_gold_standard.py \
      --mode full --seed 12345
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
        n_simulations: int,
        n_participants: int,
        seed: int,
        draws: int,
        tune: int,
        run_label: str = "default",
    ) -> None:
        self.n_simulations = n_simulations
        self.n_participants = n_participants
        self.draws = draws
        self.tune = tune
        self.run_label = run_label
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

        # Mixture noise parameters (~20% CV overall)
        self.gaussian_sd = 0.12 * self.T_o
        self.exponential_tau = 0.08 * self.T_o

        # Counting-task anchor block
        self.anchor_density = 0.15
        self.anchor_novelty = 0.05
        self.n_trials_anchor = 25

        # Factorial block (3 × 6 × 8 = 144; using 8 trials per cell)
        self.factorial_densities = np.array([0.2, 0.5, 0.8])
        self.factorial_novelties = np.array([0.10, 0.25, 0.45, 0.65, 0.80, 0.95])
        self.n_trials_per_cell = 8

        self.true_params: List[Dict[str, np.ndarray]] = []
        self.posterior_means: List[Dict[str, np.ndarray]] = []

    # ------------------------------------------------------------------
    @staticmethod
    def effective_density_map(density: float) -> float:
        s50 = 0.30
        return density / (density + s50)

    @staticmethod
    def compute_novelty_from_repetitions(k_reps: int, beta_N: float = 1.0) -> float:
        return 1.0 - np.exp(-beta_N * np.log(k_reps + 1))

    def tic_model(
        self,
        D_eff: float,
        N_prime: float,
        Phi_prime: float,
        lam: float,
        kap: float,
        gam: float,
        D0: float,
    ) -> float:
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

            # Mixture parameters (participant specific)
            p_lapse_true = self.rng.beta(2.0, 30.0)
            mu_lapse_true = float(np.clip(self.rng.normal(0.75 * self.T_o, 0.1 * self.T_o), 5.0, 150.0))
            sigma_lapse_true = float(np.clip(self.rng.normal(0.15 * self.T_o, 0.02 * self.T_o), 0.01, 0.30 * self.T_o))
            sigma_focused_true = float(np.clip(self.rng.normal(self.gaussian_sd, 0.02 * self.T_o), 0.01, 0.25 * self.T_o))

            # Counting-task anchor block
            D_eff_anchor = self.effective_density_map(self.anchor_density)
            for _ in range(self.n_trials_anchor):
                Phi_prime = float(np.clip(phi_trait[pid] + self.rng.normal(0.0, 0.05), 0.2, 2.0))
                N_true = float(np.clip(self.rng.normal(self.anchor_novelty, 0.02), 0.01, 0.10))
                N_obs = N_true  # controlled task

                T_s_true = self.tic_model(D_eff_anchor, N_true, Phi_prime, lam, kap, gam, D0_i)
                noise = self.rng.normal(0.0, self.gaussian_sd * 0.7) + self.rng.exponential(self.exponential_tau * 0.5)
                T_s = float(np.clip(T_s_true + noise, 5.0, 150.0))

                D_eff_vals.append(D_eff_anchor)
                N_obs_vals.append(N_obs)
                Phi_vals.append(Phi_prime)
                Ts_vals.append(T_s)

            # Factorial block
            stimuli = np.array(list("ABCDEF"))
            for density_f in self.factorial_densities:
                D_eff_f = self.effective_density_map(density_f)

                for novelty_target in self.factorial_novelties:
                    # Repetition bias (low novelty target -> high repeat prob)
                    repeat_prob = float(np.clip(1.0 - novelty_target, 0.05, 0.95))
                    current_stim = self.rng.choice(stimuli)
                    k_reps = 0

                    for _ in range(self.n_trials_per_cell):
                        if self.rng.random() < repeat_prob:
                            stim = current_stim
                            k_reps += 1
                        else:
                            options = stimuli[stimuli != current_stim]
                            stim = self.rng.choice(options)
                            k_reps = 0
                        current_stim = stim

                        N_true = float(np.clip(self.compute_novelty_from_repetitions(k_reps), 0.0, 1.0))
                        N_obs = float(np.clip(N_true + self.rng.normal(0.0, 0.05), 0.0, 1.0))
                        Phi_prime = float(np.clip(phi_trait[pid] + self.rng.normal(0.0, 0.10), 0.2, 2.0))

                        T_s_true = self.tic_model(D_eff_f, N_true, Phi_prime, lam, kap, gam, D0_i)
                        if self.rng.random() < p_lapse_true:
                            T_s = float(np.clip(self.rng.normal(mu_lapse_true, sigma_lapse_true), 5.0, 150.0))
                        else:
                            T_s = float(np.clip(T_s_true + self.rng.normal(0.0, sigma_focused_true), 5.0, 150.0))

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

            sigma_focused = pm.HalfNormal("sigma_focused", sigma=self.gaussian_sd)
            p_lapse = pm.Beta("p_lapse", alpha=2.0, beta=30.0)
            mu_lapse = pm.Normal("mu_lapse", mu=0.75 * self.T_o, sigma=0.1 * self.T_o)
            sigma_lapse = pm.HalfNormal("sigma_lapse", sigma=0.15 * self.T_o)

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

            numerator = 1.0 + kappa_clipped[:, None] * pm.math.pow(N_obs, gamma_clipped[:, None])
            denom = lambda_clipped[:, None] * (D0_clipped[:, None] + D_eff) * Phi
            denom = pm.math.maximum(denom, 1e-6)
            mu_preds = self.T_o * numerator / denom

            mu_components = pm.math.stack([
                pm.math.full_like(mu_preds, mu_lapse),
                mu_preds,
            ])
            sigma_components = pm.math.stack([
                pm.math.full_like(mu_preds, sigma_lapse),
                pm.math.full_like(mu_preds, sigma_focused),
            ])
            w = pm.math.stack([p_lapse, 1 - p_lapse])

            pm.NormalMixture(
                "Ts_obs",
                w=w,
                mu=mu_components,
                sigma=sigma_components,
                observed=Ts,
            )

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
        path = os.path.join(output_dir, self.run_label)
        os.makedirs(path, exist_ok=True)
        summary = az.summary(
            idata,
            var_names=[
                "mu_D0", "sigma_D0", "mu_lambda", "sigma_lambda",
                "mu_kappa", "sigma_kappa", "mu_gamma", "sigma_gamma",
                "sigma_focused", "p_lapse", "mu_lapse", "sigma_lapse",
            ],
        )
        summary_file = os.path.join(path, f"sim_{sim_idx:03d}_summary.txt")
        with open(summary_file, "w") as f:
            f.write(summary.to_string())

        fig, axes = plt.subplots(12, 2, figsize=(12, 24))
        az.plot_trace(
            idata,
            var_names=[
                "mu_D0", "sigma_D0", "mu_lambda", "sigma_lambda",
                "mu_kappa", "sigma_kappa", "mu_gamma", "sigma_gamma",
                "sigma_focused", "p_lapse", "mu_lapse", "sigma_lapse",
            ],
            axes=axes,
        )
        fig.tight_layout()
        fig.savefig(os.path.join(path, f"sim_{sim_idx:03d}_trace.png"), dpi=150)
        plt.close(fig)

    def run(self, chains: int, target_accept: float) -> None:
        print("Running TIC Parameter Recovery v9 (gold-standard design)")
        print(f"Simulations: {self.n_simulations} | Participants per simulation: {self.n_participants}\n")

        records: List[Dict[str, float]] = []

        for sim in range(self.n_simulations):
            print(f"--- Simulation {sim + 1}/{self.n_simulations} ---", flush=True)
            participants, true_params = self.generate_synthetic_data()
            idata = self.estimate_with_pymc(participants, chains=chains, target_accept=target_accept)
            self.save_diagnostics(idata, sim)
            est_means = self.compute_posterior_means(idata)
            stats = self.compute_statistics(true_params["individual"], est_means)

            for param, vals in stats.items():
                record = {"param": param, **vals}
                records.append(record)

            df = pd.DataFrame(records)
            agg = df.groupby("param").agg(["mean", "std", "count"])

            print("Parameter recovery for this simulation:")
            for param, vals in stats.items():
                print(
                    f"  {param:<6} | r = {vals['r']:+.3f} | bias = {vals['bias']:+.3f} | RMSE = {vals['rmse']:.3f} | MAE = {vals['mae']:.3f}"
                )
            print("\nRunning aggregate (mean ± std | n):")
            for param in agg.index:
                row = agg.loc[param]
                print(
                    f"  {param:<6} | r={row['r','mean']:+.3f}±{row['r','std']:.3f} (n={int(row['r','count'])}) | "
                    f"bias={row['bias','mean']:+.3f}±{row['bias','std']:.3f}"
                )
            print("", flush=True)

    # ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hierarchical TIC parameter recovery (v9 gold-standard)",
        epilog="Full run: --mode full --seed 12345",
    )
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke", help="Preset run configuration")
    parser.add_argument("--n-sim", type=int, help="Override number of simulations")
    parser.add_argument("--n-participants", type=int, help="Override participants per simulation")
    parser.add_argument("--draws", type=int, help="Posterior draws per chain")
    parser.add_argument("--tune", type=int, help="Tuning steps")
    parser.add_argument("--chains", type=int, default=4, help="Number of NUTS chains")
    parser.add_argument("--target-accept", type=float, help="Target accept rate (default 0.95 smoke / 0.98 full)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> Tuple[int, int, int, int, float, str]:
    if args.mode == "smoke":
        n_sim = args.n_sim or 2
        n_participants = args.n_participants or 15
        draws = args.draws or 600
        tune = args.tune or 800
        target_accept = args.target_accept or 0.95
        label = f"smoke_seed_{args.seed}"
    else:
        n_sim = args.n_sim or 200
        n_participants = args.n_participants or 35
        draws = args.draws or 1500
        tune = args.tune or 2000
        target_accept = args.target_accept or 0.98
        label = f"full_seed_{args.seed}"
    return n_sim, n_participants, draws, tune, target_accept, label


def main() -> None:
    args = parse_args()
    n_sim, n_participants, draws, tune, target_accept, label = resolve_config(args)
    recovery = TICParameterRecoveryV9(
        n_simulations=n_sim,
        n_participants=n_participants,
        seed=args.seed,
        draws=draws,
        tune=tune,
        run_label=label,
    )
    recovery.run(chains=args.chains, target_accept=target_accept)


if __name__ == "__main__":
    main()
