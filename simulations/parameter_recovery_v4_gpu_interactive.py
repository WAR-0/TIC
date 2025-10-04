#!/usr/bin/env python3
"""
TIC Parameter Recovery v4 (GPU-Ready) - Interactive Version
==========================================================

Enhanced with real-time progress monitoring and intermediate statistics.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Tuple

import numpy as np

import jax
import jax.numpy as jnp
import optax

# Enable float64 precision
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Helper transforms
# ---------------------------------------------------------------------------
def sigmoid_bounds(z: jnp.ndarray, lower: float, upper: float) -> jnp.ndarray:
    """Map unconstrained values to (lower, upper) via logistic transform."""
    return lower + (upper - lower) * jax.nn.sigmoid(z)


def sigmoid_bounds_np(z: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """NumPy equivalent of the logistic transform."""
    return lower + (upper - lower) / (1.0 + np.exp(-z))


def huber_loss(residual: jnp.ndarray, delta: float) -> jnp.ndarray:
    """Huber loss elementwise."""
    abs_r = jnp.abs(residual)
    quad = 0.5 * residual ** 2
    linear = delta * (abs_r - 0.5 * delta)
    return jnp.where(abs_r <= delta, quad, linear)


# ---------------------------------------------------------------------------
# Progress display helpers
# ---------------------------------------------------------------------------
def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """Print a progress bar to terminal."""
    percent = 100 * (iteration / float(total))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent:5.1f}% {suffix}', end='', flush=True)
    if iteration == total:
        print()


def print_stats_table(stats: Dict[str, Dict[str, float]], n_completed: int, n_total: int):
    """Print intermediate statistics table."""
    print(f"\n{'='*90}")
    print(f"INTERIM RESULTS ({n_completed}/{n_total} simulations)")
    print(f"{'='*90}")
    print(f"{'Parameter':<12} {'r':>8} {'Bias':>10} {'RelBias%':>10} {'RMSE':>10} {'MAE':>10}")
    print(f"{'-'*90}")
    for param, values in stats.items():
        r = values["r"]
        r_str = f"{r:8.3f}" if not np.isnan(r) else "   nan"
        print(f"{param:<12}{r_str}{values['bias']:10.3f}{values['rel_bias_pct']:10.1f}"
              f"{values['rmse']:10.3f}{values['mae']:10.3f}")
    print(f"{'='*90}\n")


# ---------------------------------------------------------------------------
# Core simulator
# ---------------------------------------------------------------------------
class TICParameterRecoveryV4GPU:
    def __init__(self, n_simulations: int = 500, n_participants: int = 35, *, seed: int = 42):
        self.n_simulations = n_simulations
        self.n_participants = n_participants
        self.T_o = 60.0
        self.rng = np.random.default_rng(seed)

        # Parameter bounds
        self.bounds = {
            "D0": (0.05, 0.20),
            "lambda": (0.5, 3.0),
            "kappa": (0.1, 2.0),
            "gamma": (0.2, 2.0),
        }
        # Priors
        self.prior_means = {"D0": 0.12, "lambda": 1.2, "kappa": 0.7, "gamma": 1.0}
        self.prior_scales = {"D0": 0.03, "lambda": 0.35, "kappa": 0.5, "gamma": 0.3}
        self.prior_weights = {"D0": 80.0, "lambda": 10.0, "kappa": 10.0, "gamma": 10.0}

        # Design
        self.density_levels = np.array([0.125, 0.3125, 0.5, 0.6875, 0.875])
        self.novelty_levels = np.array([0.2, 0.4, 0.6, 0.8])
        self.fixed_density_block_b = 0.25
        self.calibration_density_levels = np.array([0.0, 0.05, 0.15, 0.30, 0.60, 1.00])

        self.n_trials_per_density = 6
        self.n_trials_per_novelty = 6
        self.n_trials_per_calib = 5

        self.lapse_rate = 0.05
        self.student_t_df = 4
        self.novelty_mismatch_sd = 0.1
        self.calibration_noise_scale = 0.025

        self.trials_per_participant = (
            len(self.density_levels) * self.n_trials_per_density
            + len(self.novelty_levels) * self.n_trials_per_novelty
            + len(self.calibration_density_levels) * self.n_trials_per_calib
        )

        self.delta = 1.0

    @staticmethod
    def effective_density_map(density: float) -> float:
        s50 = 0.30
        return density / (density + s50)

    def sample_truncated(self, mean: float, sd: float, lower: float, upper: float, size: int) -> np.ndarray:
        vals = self.rng.normal(mean, sd, size)
        mask = (vals < lower) | (vals > upper)
        while np.any(mask):
            vals[mask] = self.rng.normal(mean, sd, mask.sum())
            mask = (vals < lower) | (vals > upper)
        return vals

    def generate_single_dataset(self) -> Tuple[Tuple[jnp.ndarray, ...], Dict[str, np.ndarray]]:
        nP = self.n_participants
        nT = self.trials_per_participant

        # True parameters
        D0_true = self.sample_truncated(self.prior_means["D0"], 0.02, *self.bounds["D0"], size=1)[0]
        lambda_true = self.sample_truncated(self.prior_means["lambda"], 0.20, *self.bounds["lambda"], size=nP)
        kappa_true = self.sample_truncated(self.prior_means["kappa"], 0.15, *self.bounds["kappa"], size=nP)
        gamma_true = self.sample_truncated(self.prior_means["gamma"], 0.12, *self.bounds["gamma"], size=nP)

        phi_trait = np.clip(self.rng.normal(1.0, 0.15, nP), 0.5, 1.6)

        D_eff_mat = np.empty((nP, nT), dtype=np.float64)
        N_obs_mat = np.empty((nP, nT), dtype=np.float64)
        Phi_mat = np.empty((nP, nT), dtype=np.float64)
        Ts_mat = np.empty((nP, nT), dtype=np.float64)

        for pid in range(nP):
            lam = lambda_true[pid]
            kap = kappa_true[pid]
            gam = gamma_true[pid]
            idx = 0

            # Block A: Density manipulation
            for density in self.density_levels:
                D_eff = self.effective_density_map(density)
                for _ in range(self.n_trials_per_density):
                    Phi_prime = np.clip(phi_trait[pid] + self.rng.normal(0, 0.1), 0.2, 2.0)
                    N_true = np.clip(self.rng.uniform(0.2, 0.8), 0.05, 0.95)
                    N_obs = np.clip(N_true + self.rng.normal(0, self.novelty_mismatch_sd), 0.05, 0.95)
                    numerator = 1.0 + kap * (N_true ** gam)
                    denom = lam * (D0_true + D_eff) * Phi_prime
                    T_s_true = self.T_o * numerator / max(denom, 1e-6)
                    if self.rng.random() < self.lapse_rate:
                        T_s = self.rng.uniform(20.0, 100.0)
                    else:
                        noise = self.rng.standard_t(self.student_t_df) * (0.07 * self.T_o)
                        T_s = np.clip(T_s_true + noise, 5.0, 120.0)
                    D_eff_mat[pid, idx] = D_eff
                    N_obs_mat[pid, idx] = N_obs
                    Phi_mat[pid, idx] = Phi_prime
                    Ts_mat[pid, idx] = T_s
                    idx += 1

            # Block B: Novelty manipulation
            D_eff_const = self.effective_density_map(self.fixed_density_block_b)
            for novelty in self.novelty_levels:
                for _ in range(self.n_trials_per_novelty):
                    Phi_prime = np.clip(phi_trait[pid] + self.rng.normal(0, 0.1), 0.2, 2.0)
                    N_true = np.clip(novelty + self.rng.normal(0, 0.05), 0.05, 0.95)
                    N_obs = np.clip(N_true + self.rng.normal(0, self.novelty_mismatch_sd), 0.05, 0.95)
                    numerator = 1.0 + kap * (N_true ** gam)
                    denom = lam * (D0_true + D_eff_const) * Phi_prime
                    T_s_true = self.T_o * numerator / max(denom, 1e-6)
                    if self.rng.random() < self.lapse_rate:
                        T_s = self.rng.uniform(20.0, 100.0)
                    else:
                        noise = self.rng.standard_t(self.student_t_df) * (0.07 * self.T_o)
                        T_s = np.clip(T_s_true + noise, 5.0, 120.0)
                    D_eff_mat[pid, idx] = D_eff_const
                    N_obs_mat[pid, idx] = N_obs
                    Phi_mat[pid, idx] = Phi_prime
                    Ts_mat[pid, idx] = T_s
                    idx += 1

            # Block C: Calibration
            for density_calib in self.calibration_density_levels:
                D_eff = self.effective_density_map(density_calib)
                for _ in range(self.n_trials_per_calib):
                    Phi_prime = np.clip(phi_trait[pid] + self.rng.normal(0, 0.05), 0.2, 2.0)
                    denominator = lam * (D0_true + D_eff) * Phi_prime
                    T_s_true = self.T_o / max(denominator, 1e-6)
                    noise = self.rng.standard_t(self.student_t_df) * (self.calibration_noise_scale * self.T_o)
                    T_s = np.clip(T_s_true + noise, 10.0, 120.0)
                    D_eff_mat[pid, idx] = D_eff
                    N_obs_mat[pid, idx] = 0.0
                    Phi_mat[pid, idx] = Phi_prime
                    Ts_mat[pid, idx] = T_s
                    idx += 1

        batch = (
            jnp.array(D_eff_mat),
            jnp.array(N_obs_mat),
            jnp.array(Phi_mat),
            jnp.array(Ts_mat),
        )
        true_params = {
            "D0": np.array(D0_true),
            "lambda": lambda_true,
            "kappa": kappa_true,
            "gamma": gamma_true,
        }
        return batch, true_params

    def build_loss_fn(self) -> Tuple[int, callable]:
        nP = self.n_participants
        bounds = self.bounds
        priors = self.prior_means
        scales = self.prior_scales
        weights = self.prior_weights
        delta = self.delta
        T_o = self.T_o

        def loss_fn(params: jnp.ndarray, D_eff: jnp.ndarray, N_obs: jnp.ndarray, 
                   Phi: jnp.ndarray, Ts: jnp.ndarray) -> jnp.ndarray:
            D0 = sigmoid_bounds(params[0], *bounds["D0"])
            lam = sigmoid_bounds(params[1:1 + nP], *bounds["lambda"])
            kap = sigmoid_bounds(params[1 + nP:1 + 2 * nP], *bounds["kappa"])
            gam = sigmoid_bounds(params[1 + 2 * nP:], *bounds["gamma"])

            safe_N = jnp.clip(N_obs, 1e-6, 1.0)
            n_pow = jnp.where(N_obs > 0, safe_N ** gam[:, None], 0.0)
            numerator = 1.0 + kap[:, None] * n_pow
            denom = lam[:, None] * (D0 + D_eff) * Phi
            denom = jnp.maximum(denom, 1e-6)
            preds = T_o * numerator / denom
            residual = Ts - preds
            data_loss = jnp.sum(huber_loss(residual, delta))

            d0_pen = ((D0 - priors["D0"]) / scales["D0"]) ** 2 * weights["D0"]
            lam_pen = jnp.sum(
                ((jnp.log(lam) - jnp.log(priors["lambda"])) / scales["lambda"]) ** 2
            ) * weights["lambda"]
            kap_pen = jnp.sum(
                ((jnp.log(kap) - jnp.log(priors["kappa"])) / scales["kappa"]) ** 2
            ) * weights["kappa"]
            gam_pen = jnp.sum(
                ((gam - priors["gamma"]) / scales["gamma"]) ** 2
            ) * weights["gamma"]

            return data_loss + d0_pen + lam_pen + kap_pen + gam_pen

        dim = 1 + 3 * nP
        return dim, jax.jit(jax.value_and_grad(loss_fn))

    def initialize_params(self) -> jnp.ndarray:
        def inv_sigmoid_bounds(value: float, lower: float, upper: float) -> float:
            x = (value - lower) / (upper - lower)
            x = np.clip(x, 1e-6, 1 - 1e-6)
            return np.log(x / (1 - x))

        D0_z = inv_sigmoid_bounds(self.prior_means["D0"], *self.bounds["D0"])
        lam_z = inv_sigmoid_bounds(self.prior_means["lambda"], *self.bounds["lambda"])
        kap_z = inv_sigmoid_bounds(self.prior_means["kappa"], *self.bounds["kappa"])
        gam_z = inv_sigmoid_bounds(self.prior_means["gamma"], *self.bounds["gamma"])

        z0 = np.empty(1 + 3 * self.n_participants, dtype=np.float64)
        z0[0] = D0_z
        z0[1:1 + self.n_participants] = lam_z
        z0[1 + self.n_participants:1 + 2 * self.n_participants] = kap_z
        z0[1 + 2 * self.n_participants:] = gam_z
        return jnp.array(z0)

    def fit_single_dataset(
        self,
        loss_grad_fn,
        batch: Tuple[jnp.ndarray, ...],
        *,
        max_steps: int = 1500,
        lr: float = 0.01,
        tol: float = 1e-3,
        show_progress: bool = True,
    ) -> Dict[str, np.ndarray]:
        params = self.initialize_params()
        opt = optax.adam(lr)
        opt_state = opt.init(params)

        best_params = params
        best_loss = jnp.inf

        for step in range(1, max_steps + 1):
            loss, grads = loss_grad_fn(params, *batch)
            updates, opt_state = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            if loss < best_loss:
                best_loss = loss
                best_params = params

            grad_norm = jnp.linalg.norm(grads) / jnp.sqrt(grads.size)
            
            if show_progress and step % 100 == 0:
                print_progress_bar(step, max_steps, 
                                 prefix=f'Optimizing', 
                                 suffix=f'loss={float(loss):>8.1f} grad={float(grad_norm):>6.2f}',
                                 length=30)
            
            if grad_norm < tol:
                if show_progress:
                    print_progress_bar(max_steps, max_steps, 
                                     prefix=f'Optimizing', 
                                     suffix=f'Converged at step {step}            ',
                                     length=30)
                break
        else:
            if show_progress:
                print_progress_bar(max_steps, max_steps, 
                                 prefix=f'Optimizing', 
                                 suffix=f'Completed {max_steps} steps            ',
                                 length=30)

        # Transform back to constrained space
        best_params.block_until_ready()
        params_np = np.array(best_params)
        D0 = sigmoid_bounds_np(params_np[0], *self.bounds["D0"])
        lam = sigmoid_bounds_np(params_np[1:1 + self.n_participants], *self.bounds["lambda"])
        kap = sigmoid_bounds_np(params_np[1 + self.n_participants:1 + 2 * self.n_participants], *self.bounds["kappa"])
        gam = sigmoid_bounds_np(params_np[1 + 2 * self.n_participants:], *self.bounds["gamma"])

        return {
            "D0": np.asarray(D0, dtype=np.float64),
            "lambda": np.asarray(lam, dtype=np.float64),
            "kappa": np.asarray(kap, dtype=np.float64),
            "gamma": np.asarray(gam, dtype=np.float64),
        }

    @staticmethod
    def compute_stats(true_vals: np.ndarray, est_vals: np.ndarray, param_range: float) -> Dict[str, float]:
        corr = np.corrcoef(true_vals, est_vals)[0, 1] if np.std(true_vals) > 1e-6 else np.nan
        bias = float(np.mean(est_vals - true_vals))
        rel_bias = 100.0 * bias / param_range
        rmse = float(np.sqrt(np.mean((est_vals - true_vals) ** 2)))
        mae = float(np.mean(np.abs(est_vals - true_vals)))
        return {
            "r": float(corr),
            "bias": bias,
            "rel_bias_pct": rel_bias,
            "rmse": rmse,
            "mae": mae,
        }

    def summarise(self, true_params_list, est_params_list) -> Dict[str, Dict[str, float]]:
        true_D0 = np.array([tp["D0"] for tp in true_params_list])
        est_D0 = np.array([ep["D0"] for ep in est_params_list])
        true_lambda = np.concatenate([tp["lambda"] for tp in true_params_list])
        est_lambda = np.concatenate([ep["lambda"] for ep in est_params_list])
        true_kappa = np.concatenate([tp["kappa"] for tp in true_params_list])
        est_kappa = np.concatenate([ep["kappa"] for ep in est_params_list])
        true_gamma = np.concatenate([tp["gamma"] for tp in true_params_list])
        est_gamma = np.concatenate([ep["gamma"] for ep in est_params_list])

        stats = {}
        stats["D0"] = self.compute_stats(true_D0, est_D0, self.bounds["D0"][1] - self.bounds["D0"][0])
        stats["lambda"] = self.compute_stats(true_lambda, est_lambda, self.bounds["lambda"][1] - self.bounds["lambda"][0])
        stats["kappa"] = self.compute_stats(true_kappa, est_kappa, self.bounds["kappa"][1] - self.bounds["kappa"][0])
        stats["gamma"] = self.compute_stats(true_gamma, est_gamma, self.bounds["gamma"][1] - self.bounds["gamma"][0])

        rho_true = true_kappa / true_lambda
        rho_est = est_kappa / est_lambda
        rho_range = self.bounds["kappa"][1] / self.bounds["lambda"][0] - self.bounds["kappa"][0] / self.bounds["lambda"][1]
        stats["rho"] = self.compute_stats(rho_true, rho_est, rho_range)
        return stats

    @staticmethod
    def write_results(path: str, stats: Dict[str, Dict[str, float]], n_sim: int, n_participants: int, trials_per_participant: int):
        lines = []
        lines.append("TIC Parameter Recovery v4 (GPU - Interactive)")
        lines.append("=" * 90)
        lines.append(f"Simulations       : {n_sim}")
        lines.append(f"Participants      : {n_participants}")
        lines.append(f"Trials/participant: {trials_per_participant}")
        lines.append("")
        lines.append("Parameter Table")
        lines.append("-" * 90)
        header = f"{'Parameter':<12} {'r':>8} {'Bias':>10} {'RelBias%':>10} {'RMSE':>10} {'MAE':>10}"
        lines.append(header)
        lines.append("-" * 90)
        for param, values in stats.items():
            r = values["r"]
            r_str = f"{r:8.3f}" if not np.isnan(r) else "   nan"
            row = f"{param:<12}{r_str}{values['bias']:10.3f}{values['rel_bias_pct']:10.1f}{values['rmse']:10.3f}{values['mae']:10.3f}"
            lines.append(row)
        lines.append("-" * 90)
        lines.append("Notes:")
        lines.append("  • Hierarchical D₀ with strong priors, JAX-accelerated optimization.")
        lines.append("  • Interactive mode with real-time progress monitoring.")
        lines.append("")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

    def run(self, max_steps: int, lr: float, tol: float, update_interval: int = 10):
        loss_grad_dim, loss_grad_fn = self.build_loss_fn()

        true_params_list = []
        est_params_list = []

        print(f"\n{'='*90}")
        print(f"GPU-Accelerated TIC Parameter Recovery (Interactive Mode)")
        print(f"{'='*90}")
        print(f"Simulations: {self.n_simulations} | Participants: {self.n_participants} | Trials: {self.trials_per_participant}")
        print(f"Update interval: Every {update_interval} simulations")
        print(f"{'='*90}\n")

        for sim in range(self.n_simulations):
            print(f"\nSimulation {sim + 1}/{self.n_simulations}")
            
            batch, true_params = self.generate_single_dataset()
            est_params = self.fit_single_dataset(
                loss_grad_fn,
                batch,
                max_steps=max_steps,
                lr=lr,
                tol=tol,
                show_progress=True,
            )
            true_params_list.append(true_params)
            est_params_list.append(est_params)

            # Show interim results at intervals
            if (sim + 1) % update_interval == 0 or sim == 0:
                stats = self.summarise(true_params_list, est_params_list)
                print_stats_table(stats, sim + 1, self.n_simulations)

        # Final results
        stats = self.summarise(true_params_list, est_params_list)
        print(f"\n{'='*90}")
        print("FINAL RESULTS")
        print(f"{'='*90}")
        print(f"{'Parameter':<12} {'r':>8} {'Bias':>10} {'RelBias%':>10} {'RMSE':>10} {'MAE':>10}")
        print(f"{'-'*90}")
        for param, values in stats.items():
            r = values["r"]
            r_str = f"{r:8.3f}" if not np.isnan(r) else "   nan"
            print(f"{param:<12}{r_str}{values['bias']:10.3f}{values['rel_bias_pct']:10.1f}{values['rmse']:10.3f}{values['mae']:10.3f}")
        print(f"{'='*90}\n")

        results_path = os.path.join(os.path.dirname(__file__), "results_v4_gpu_interactive.txt")
        self.write_results(results_path, stats, self.n_simulations, self.n_participants, self.trials_per_participant)
        print(f"Results written to {results_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU-accelerated TIC parameter recovery v4 (Interactive)")
    parser.add_argument("--n-sim", type=int, default=200, help="Number of simulations (default: 200)")
    parser.add_argument("--n-participants", type=int, default=35, help="Participants per simulation (default: 35)")
    parser.add_argument("--max-steps", type=int, default=1500, help="Optimization steps (default: 1500)")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Adam learning rate (default: 0.01)")
    parser.add_argument("--tol", type=float, default=1e-3, help="Gradient norm tolerance (default: 1e-3)")
    parser.add_argument("--update-interval", type=int, default=10, help="Show stats every N simulations (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return parser.parse_args()


def main():
    args = parse_args()
    simulator = TICParameterRecoveryV4GPU(
        n_simulations=args.n_sim,
        n_participants=args.n_participants,
        seed=args.seed,
    )
    simulator.run(
        max_steps=args.max_steps,
        lr=args.learning_rate,
        tol=args.tol,
        update_interval=args.update_interval,
    )


if __name__ == "__main__":
    main()
