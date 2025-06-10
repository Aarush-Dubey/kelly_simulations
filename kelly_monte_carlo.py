"""
Kelly Criterion Monte Carlo Simulation Engine
============================================
Core simulation functions for Kelly betting strategy evaluation
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from dataclasses import dataclass

@dataclass
class SimulationParams:
    """Parameters for Monte Carlo simulation"""
    win_prob: float
    payout_multiplier: float
    initial_bankroll: float = 1000.0
    num_rounds: int = 500
    num_simulations: int = 10000
    min_bet_fraction: float = 0.001  # Minimum bet as fraction of bankroll

class KellyEngine:
    """Monte Carlo simulation engine for Kelly Criterion betting"""
    
    def __init__(self):
        self.results_cache = {}
    
    def calculate_kelly_fraction(self, win_prob: float, payout_multiplier: float) -> float:
        """
        Calculate optimal Kelly fraction for betting
        
        Kelly formula: f* = (bp - q) / b
        where:
        - b = payout multiplier (odds)
        - p = probability of winning
        - q = probability of losing (1-p)
        """
        if win_prob <= 0 or win_prob >= 1:
            return 0.0
        
        q = 1 - win_prob
        kelly_fraction = (payout_multiplier * win_prob - q) / payout_multiplier
        
        # Cap at reasonable maximum (50% of bankroll)
        return max(0, min(kelly_fraction, 0.5))
    
    def simulate_single_path(self, params: SimulationParams) -> Dict:
        """
        Simulate a single betting path using Kelly Criterion
        
        Returns:
            Dict with path statistics
        """
        bankroll_history = np.zeros(params.num_rounds + 1)
        bankroll_history[0] = params.initial_bankroll
        
        current_bankroll = params.initial_bankroll
        kelly_fraction = self.calculate_kelly_fraction(params.win_prob, params.payout_multiplier)
        
        max_bankroll = current_bankroll
        max_drawdown = 0.0
        
        # Generate all random outcomes at once for speed
        random_outcomes = np.random.random(params.num_rounds) < params.win_prob
        
        for round_num in range(params.num_rounds):
            if current_bankroll <= 0:
                bankroll_history[round_num + 1:] = 0
                break
            
            # Calculate bet size using Kelly fraction
            bet_fraction = max(kelly_fraction, params.min_bet_fraction)
            bet_amount = current_bankroll * bet_fraction
            
            # Determine outcome
            if random_outcomes[round_num]:  # Win
                current_bankroll += bet_amount * params.payout_multiplier
            else:  # Loss
                current_bankroll -= bet_amount
            
            bankroll_history[round_num + 1] = current_bankroll
            
            # Track maximum and drawdown
            if current_bankroll > max_bankroll:
                max_bankroll = current_bankroll
            
            current_drawdown = (max_bankroll - current_bankroll) / max_bankroll
            max_drawdown = max(max_drawdown, current_drawdown)
        
        # Calculate geometric mean return
        if params.initial_bankroll > 0 and current_bankroll > 0:
            geometric_return = (current_bankroll / params.initial_bankroll) ** (1/params.num_rounds) - 1
        else:
            geometric_return = -1.0
        
        return {
            'final_bankroll': current_bankroll,
            'bankroll_history': bankroll_history,
            'max_drawdown': max_drawdown,
            'geometric_return': geometric_return,
            'kelly_fraction': kelly_fraction,
            'total_return': (current_bankroll - params.initial_bankroll) / params.initial_bankroll,
            'bankroll_ratio': current_bankroll / params.initial_bankroll,
            'ruined': current_bankroll <= 0.01 * params.initial_bankroll
        }
    
    def run_monte_carlo_batch(self, params: SimulationParams) -> pd.DataFrame:
        """
        Run Monte Carlo simulation batch for given parameters
        
        Returns:
            DataFrame with simulation results
        """
        print(f"Running {params.num_simulations:,} simulations for p={params.win_prob:.3f}, b={params.payout_multiplier:.2f}")
        
        results = []
        bankroll_histories = []
        
        # Set random seed for reproducibility in each batch
        np.random.seed(42)
        
        for sim_idx in range(params.num_simulations):
            if sim_idx % 1000 == 0:
                print(f"  Progress: {sim_idx:,}/{params.num_simulations:,}")
            
            path_result = self.simulate_single_path(params)
            
            results.append({
                'simulation_id': sim_idx,
                'win_prob': params.win_prob,
                'payout_multiplier': params.payout_multiplier,
                'final_bankroll': path_result['final_bankroll'],
                'max_drawdown': path_result['max_drawdown'],
                'geometric_return': path_result['geometric_return'],
                'kelly_fraction': path_result['kelly_fraction'],
                'total_return': path_result['total_return'],
                'bankroll_ratio': path_result['bankroll_ratio'],
                'ruined': path_result['ruined']
            })
            
            # Store every 100th history for visualization
            if sim_idx % 100 == 0:
                bankroll_histories.append(path_result['bankroll_history'])
        
        df = pd.DataFrame(results)
        df['sample_histories'] = [bankroll_histories] * len(df) if bankroll_histories else [[]]
        
        return df
    
    def calculate_batch_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate summary statistics for a batch of simulations"""
        if df.empty:
            return {}
        
        return {
            'mean_final_bankroll': df['final_bankroll'].mean(),
            'median_final_bankroll': df['final_bankroll'].median(),
            'std_final_bankroll': df['final_bankroll'].std(),
            'percentile_5': df['final_bankroll'].quantile(0.05),
            'percentile_25': df['final_bankroll'].quantile(0.25),
            'percentile_75': df['final_bankroll'].quantile(0.75),
            'percentile_95': df['final_bankroll'].quantile(0.95),
            'mean_geometric_return': df['geometric_return'].mean(),
            'median_geometric_return': df['geometric_return'].median(),
            'mean_max_drawdown': df['max_drawdown'].mean(),
            'max_drawdown_worst': df['max_drawdown'].max(),
            'ruin_probability': df['ruined'].mean(),
            'kelly_fraction': df['kelly_fraction'].iloc[0],  # Same for all sims
            'positive_return_rate': (df['total_return'] > 0).mean(),
            'sharpe_approximation': df['geometric_return'].mean() / df['geometric_return'].std() if df['geometric_return'].std() > 0 else 0
        }