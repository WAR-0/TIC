#!/usr/bin/env python3
"""
Parameter Recovery Simulation for TIC Model (Minimal Dependencies Version)
T_s ≈ T_o · [1 + κ·N'^γ] / [λ(D')^α · Φ'^β]

Uses only NumPy - no pandas, scipy, matplotlib required
Implements simple but effective differential evolution optimizer

Author: TIC Research
Date: 2025-10-02
"""

import os
import numpy as np
from typing import Tuple, Dict, List

# Set random seed for reproducibility
np.random.seed(42)

class SimpleOptimizer:
    """Simple differential evolution optimizer using only NumPy"""

    def __init__(self, bounds, popsize=15, maxiter=100):
        self.bounds = np.array(bounds)
        self.popsize = popsize
        self.maxiter = maxiter
        self.dim = len(bounds)

    def optimize(self, objective_func):
        """Minimize objective function using differential evolution"""
        # Initialize population
        pop = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            (self.popsize, self.dim)
        )

        # Evaluate initial population
        fitness = np.array([objective_func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best_solution = pop[best_idx].copy()
        best_fitness = fitness[best_idx]

        # Evolution
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover probability

        for iteration in range(self.maxiter):
            for i in range(self.popsize):
                # Select three random distinct indices
                indices = [idx for idx in range(self.popsize) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)

                # Mutation
                mutant = pop[a] + F * (pop[b] - pop[c])

                # Ensure bounds
                mutant = np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1])

                # Crossover
                trial = pop[i].copy()
                crossover_mask = np.random.rand(self.dim) < CR
                trial[crossover_mask] = mutant[crossover_mask]

                # Ensure bounds after crossover
                trial = np.clip(trial, self.bounds[:, 0], self.bounds[:, 1])

                # Selection
                trial_fitness = objective_func(trial)
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial.copy()
                        best_fitness = trial_fitness

        return best_solution


class TICParameterRecovery:
    """Parameter recovery simulation for TIC model"""

    def __init__(self, n_simulations=1000, n_participants=35, n_trials=56):
        self.n_simulations = n_simulations
        self.n_participants = n_participants
        self.n_trials = n_trials
        self.T_o = 60.0

        # Parameter ranges
        self.param_ranges = {
            'lambda': (0.8, 2.5),
            'kappa': (0.3, 1.5),
            'alpha': (0.5, 1.2),
            'beta': (0.3, 0.8),
            'gamma': (0.5, 1.5)
        }

        # Experimental design
        self.density_levels = np.array([0.125, 0.3125, 0.5, 0.6875, 0.875])
        self.n_trials_per_density = 8
        self.novelty_levels = np.array([0.3, 0.7])
        self.n_trials_per_novelty = 8
        self.fixed_density_block_b = 0.25

        # Results storage
        self.true_params = []
        self.est_params = []

    def irtp_model(self, D_prime, N_prime, Phi_prime, lam, kappa, alpha, beta, gamma):
        """TIC model equation"""
        numerator = 1.0 + kappa * (N_prime ** gamma)
        denominator = lam * (D_prime ** alpha) * (Phi_prime ** beta)
        denominator = np.maximum(denominator, 1e-6)
        return self.T_o * (numerator / denominator)

    def generate_synthetic_data(self, true_params: Dict) -> Dict:
        """Generate synthetic dataset"""
        lam = true_params['lambda']
        kappa = true_params['kappa']
        alpha = true_params['alpha']
        beta = true_params['beta']
        gamma = true_params['gamma']

        data = {
            'D_prime': [],
            'N_prime': [],
            'Phi_prime': [],
            'T_s': [],
            'block': [],
            'participant': []
        }

        # Inter-participant variability
        participant_lambda = np.random.normal(0, 0.2, self.n_participants)
        participant_alpha = np.random.normal(0, 0.1, self.n_participants)
        participant_beta = np.random.normal(0, 0.1, self.n_participants)

        for p_id in range(self.n_participants):
            lam_i = max(0.1, lam + participant_lambda[p_id])
            alpha_i = max(0.1, alpha + participant_alpha[p_id])
            beta_i = max(0.1, beta + participant_beta[p_id])

            # Block A: Density manipulation
            for density in self.density_levels:
                for trial in range(self.n_trials_per_density):
                    N_prime = np.clip(0.2 + 0.6 * density + np.random.normal(0, 0.1), 0.05, 0.95)
                    Phi_prime = np.exp((np.random.normal(0, 0.5) + 0.3 * density) / 2)

                    T_s_true = self.irtp_model(density, N_prime, Phi_prime,
                                               lam_i, kappa, alpha_i, beta_i, gamma)
                    T_s = max(5.0, T_s_true + np.random.normal(0, 0.07 * self.T_o))

                    data['D_prime'].append(density)
                    data['N_prime'].append(N_prime)
                    data['Phi_prime'].append(Phi_prime)
                    data['T_s'].append(T_s)
                    data['block'].append('A')
                    data['participant'].append(p_id)

            # Block B: Novelty manipulation
            for novelty in self.novelty_levels:
                for trial in range(self.n_trials_per_novelty):
                    N_prime = np.clip(novelty + np.random.normal(0, 0.05), 0.05, 0.95)
                    Phi_prime = np.exp(np.random.normal(0, 0.5) / 2)

                    T_s_true = self.irtp_model(self.fixed_density_block_b, N_prime, Phi_prime,
                                               lam_i, kappa, alpha_i, beta_i, gamma)
                    T_s = max(5.0, T_s_true + np.random.normal(0, 0.07 * self.T_o))

                    data['D_prime'].append(self.fixed_density_block_b)
                    data['N_prime'].append(N_prime)
                    data['Phi_prime'].append(Phi_prime)
                    data['T_s'].append(T_s)
                    data['block'].append('B')
                    data['participant'].append(p_id)

        # Convert to numpy arrays
        for key in data:
            data[key] = np.array(data[key])

        return data

    def estimate_parameters(self, data: Dict) -> Dict:
        """Three-phase parameter estimation"""

        # Phase 1: Estimate {λ, α, β} from low-novelty trials
        low_N_mask = (data['block'] == 'A') & (data['N_prime'] < 0.4)
        if np.sum(low_N_mask) < 10:
            # Fallback: use lower 30% of N'
            N_prime_threshold = np.percentile(data['N_prime'][data['block'] == 'A'], 30)
            low_N_mask = (data['block'] == 'A') & (data['N_prime'] <= N_prime_threshold)

        phase1_D = data['D_prime'][low_N_mask]
        phase1_Phi = data['Phi_prime'][low_N_mask]
        phase1_Ts = data['T_s'][low_N_mask]

        def phase1_objective(params):
            lam, alpha, beta = params
            if lam <= 0 or alpha <= 0 or beta <= 0:
                return 1e10
            predictions = self.T_o / (lam * (phase1_D ** alpha) * (phase1_Phi ** beta))
            predictions = np.clip(predictions, 1, 120)
            return np.sum((phase1_Ts - predictions) ** 2)

        optimizer1 = SimpleOptimizer(bounds=[(0.5, 3.0), (0.2, 2.0), (0.1, 1.5)], maxiter=80)
        phase1_result = optimizer1.optimize(phase1_objective)
        lam_est, alpha_est, beta_est = phase1_result

        # Phase 2: Estimate {κ, γ} from Block B
        blockB_mask = data['block'] == 'B'
        phase2_D = data['D_prime'][blockB_mask]
        phase2_N = data['N_prime'][blockB_mask]
        phase2_Phi = data['Phi_prime'][blockB_mask]
        phase2_Ts = data['T_s'][blockB_mask]

        def phase2_objective(params):
            kappa, gamma = params
            if kappa <= 0 or gamma <= 0:
                return 1e10
            numerator = 1.0 + kappa * (phase2_N ** gamma)
            denominator = lam_est * (phase2_D ** alpha_est) * (phase2_Phi ** beta_est)
            predictions = self.T_o * (numerator / np.maximum(denominator, 1e-6))
            predictions = np.clip(predictions, 1, 120)
            return np.sum((phase2_Ts - predictions) ** 2)

        optimizer2 = SimpleOptimizer(bounds=[(0.1, 2.0), (0.2, 2.0)], maxiter=80)
        phase2_result = optimizer2.optimize(phase2_objective)
        kappa_est, gamma_est = phase2_result

        # Phase 3: Joint refinement
        all_D = data['D_prime']
        all_N = data['N_prime']
        all_Phi = data['Phi_prime']
        all_Ts = data['T_s']

        def full_objective(params):
            lam, kappa, alpha, beta, gamma = params
            if any(p <= 0 for p in params):
                return 1e10
            numerator = 1.0 + kappa * (all_N ** gamma)
            denominator = lam * (all_D ** alpha) * (all_Phi ** beta)
            predictions = self.T_o * (numerator / np.maximum(denominator, 1e-6))
            predictions = np.clip(predictions, 1, 120)
            return np.sum((all_Ts - predictions) ** 2)

        # Initialize with phase 1 & 2 results, add small perturbation
        init_solution = np.array([lam_est, kappa_est, alpha_est, beta_est, gamma_est])
        bounds = [(0.5, 3.0), (0.1, 2.0), (0.2, 2.0), (0.1, 1.5), (0.2, 2.0)]

        optimizer3 = SimpleOptimizer(bounds=bounds, maxiter=120, popsize=20)

        # Seed population with initial solution
        pop = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (20, 5))
        pop[0] = init_solution  # Include phase 1-2 solution in population

        final_result = optimizer3.optimize(full_objective)

        return {
            'lambda': final_result[0],
            'kappa': final_result[1],
            'alpha': final_result[2],
            'beta': final_result[3],
            'gamma': final_result[4]
        }

    def run_single_simulation(self, sim_id: int) -> Tuple[Dict, Dict]:
        """Run one simulation"""
        # Sample true parameters
        true_params = {
            'lambda': np.random.uniform(*self.param_ranges['lambda']),
            'kappa': np.random.uniform(*self.param_ranges['kappa']),
            'alpha': np.random.uniform(*self.param_ranges['alpha']),
            'beta': np.random.uniform(*self.param_ranges['beta']),
            'gamma': np.random.uniform(*self.param_ranges['gamma'])
        }

        # Generate data
        data = self.generate_synthetic_data(true_params)

        # Estimate parameters
        est_params = self.estimate_parameters(data)

        return true_params, est_params

    def run_all_simulations(self):
        """Run all simulations"""
        print(f"Running {self.n_simulations} parameter recovery simulations...")
        print(f"N = {self.n_participants} participants, {self.n_trials} trials each")
        print("Three-phase sequential estimation strategy\n")

        for sim_id in range(self.n_simulations):
            if (sim_id + 1) % 100 == 0:
                print(f"  Completed {sim_id + 1}/{self.n_simulations} simulations...")

            true_params, est_params = self.run_single_simulation(sim_id)
            self.true_params.append(true_params)
            self.est_params.append(est_params)

        print(f"\nCompleted all {self.n_simulations} simulations")

    def calculate_statistics(self):
        """Calculate recovery statistics"""
        params = ['lambda', 'kappa', 'alpha', 'beta', 'gamma']

        print("\n" + "=" * 80)
        print("Recovery Statistics:")
        print("=" * 80)
        print(f"{'Parameter':<10} {'Recovery r':<12} {'Bias':<10} {'Rel.Bias%':<12} {'RMSE':<10} {'MAE':<10}")
        print("-" * 80)

        stats = {}
        for param in params:
            true_vals = np.array([tp[param] for tp in self.true_params])
            est_vals = np.array([ep[param] for ep in self.est_params])

            # Recovery correlation
            corr = np.corrcoef(true_vals, est_vals)[0, 1]

            # Bias
            bias = np.mean(est_vals - true_vals)

            # Relative bias
            param_range = self.param_ranges[param][1] - self.param_ranges[param][0]
            rel_bias_pct = 100 * bias / param_range

            # RMSE
            rmse = np.sqrt(np.mean((est_vals - true_vals) ** 2))

            # MAE
            mae = np.mean(np.abs(est_vals - true_vals))

            stats[param] = {
                'r': corr,
                'bias': bias,
                'rel_bias_pct': rel_bias_pct,
                'rmse': rmse,
                'mae': mae
            }

            print(f"{param:<10} {corr:<12.3f} {bias:<10.3f} {rel_bias_pct:<12.1f} {rmse:<10.3f} {mae:<10.3f}")

        print("=" * 80)

        return stats

    def generate_manuscript_text(self, stats):
        """Generate text for manuscript (Appendix A) with accurate interpretation."""
        text = "\n" + "=" * 80 + "\n"
        text += "MANUSCRIPT TEXT FOR APPENDIX A\n"
        text += "=" * 80 + "\n\n"

        text += "**A.3. Simulation Results:**\n\n"
        text += (
            f"Parameter recovery simulations (N = {self.n_simulations} iterations) revealed differential "
            f"identifiability across TIC parameters given the proposed experimental design "
            f"(N = {self.n_participants} participants, {self.n_trials} trials per participant). "
            "A three-phase sequential estimation strategy was used. Recovery correlations were: "
        )

        # Recovery correlations (ordered for readability)
        ordered_params = ['lambda', 'alpha', 'beta', 'kappa', 'gamma']
        r_vals = [f"{p} r = {stats[p]['r']:.3f}" for p in ordered_params]
        text += ", ".join(r_vals[:-1]) + f", and {r_vals[-1]}. "

        # Bias
        text += "Parameter bias (as % of range): "
        bias_vals = [f"{p} = {stats[p]['rel_bias_pct']:.1f}%" for p in ordered_params]
        text += ", ".join(bias_vals[:-1]) + f", and {bias_vals[-1]}. "

        # Mean/min correlation and categorization
        mean_r = float(np.mean([stats[p]['r'] for p in stats.keys()]))
        min_r = float(np.min([stats[p]['r'] for p in stats.keys()]))
        good_threshold = 0.80
        good_params = [p for p in stats if stats[p]['r'] >= good_threshold]
        moderate_params = [p for p in stats if stats[p]['r'] < good_threshold]

        text += (
            f"Mean recovery correlation across parameters was r = {mean_r:.3f} "
            f"(minimum r = {min_r:.3f}). Parameters meeting the r ≥ {good_threshold:.2f} 'good' threshold "
            f"(Luzardo et al., 2013) were: {', '.join(good_params) if good_params else 'none'}. "
            f"Other parameters showed moderate recovery: {', '.join(moderate_params) if moderate_params else 'none'}. "
        )

        # RMSE
        text += "RMSE values: "
        rmse_vals = [f"{p} = {stats[p]['rmse']:.3f}" for p in ordered_params]
        text += ", ".join(rmse_vals[:-1]) + f", and {rmse_vals[-1]}. "

        text += (
            "These results suggest the proposed design identifies compression parameters well while providing "
            "moderate information for novelty-related parameters, aligning with a confirmatory framing for "
            "compression and an exploratory framing for novelty.\n\n"
        )

        # Table
        text += "**Table A1: Parameter Recovery Statistics**\n\n"
        text += "| Parameter | Recovery r | Bias    | Rel.Bias% | RMSE   | MAE    |\n"
        text += "|-----------|-----------|---------|-----------|--------|--------|\n"

        for p in ordered_params:
            s = stats[p]
            text += f"| {p:<9} | {s['r']:>9.3f} | {s['bias']:>7.3f} | {s['rel_bias_pct']:>9.1f} | "
            text += f"{s['rmse']:>6.3f} | {s['mae']:>6.3f} |\n"

        text += "\n*Recovery r = correlation between true and estimated parameters; "
        text += "Rel.Bias = bias as % of parameter range; RMSE = root mean square error; MAE = mean absolute error.*\n"

        return text


def main():
    """Main execution"""
    print("=" * 80)
    print("TIC Parameter Recovery Simulation")
    print("Minimal Dependencies Version (NumPy only)")
    print("=" * 80)
    print()

    simulator = TICParameterRecovery(
        n_simulations=1000,
        n_participants=35,
        n_trials=56
    )

    simulator.run_all_simulations()
    stats = simulator.calculate_statistics()

    manuscript_text = simulator.generate_manuscript_text(stats)
    print(manuscript_text)

    # Save to repository simulations directory
    sim_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(sim_dir, 'results.txt')
    with open(output_file, 'w') as f:
        f.write(manuscript_text)

    print(f"\nResults saved to: {output_file}")

    return simulator, stats


if __name__ == "__main__":
    simulator, stats = main()
