"""
Parameter Sweep Engine
======================
Automated grid search across probability and payout parameter spaces
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import itertools
from kelly_monte_carlo import KellyEngine, SimulationParams

class ParameterSweep:
    """Handles automated parameter sweeps across p and b values"""
    
    def __init__(self):
        self.kelly_engine = KellyEngine()
        self.sweep_results = pd.DataFrame()
        self.summary_stats = pd.DataFrame()
    
    def create_parameter_grid(self, 
                            p_range: Tuple[float, float], 
                            b_range: Tuple[float, float],
                            p_steps: int = 10, 
                            b_steps: int = 10) -> List[Tuple[float, float]]:
        """
        Create parameter grid for sweep
        
        Args:
            p_range: (min_prob, max_prob)
            b_range: (min_payout, max_payout)  
            p_steps: Number of probability steps
            b_steps: Number of payout steps
            
        Returns:
            List of (p, b) parameter combinations
        """
        p_values = np.linspace(p_range[0], p_range[1], p_steps)
        b_values = np.linspace(b_range[0], b_range[1], b_steps)
        
        return list(itertools.product(p_values, b_values))
    
    def run_parameter_sweep(self,
                          p_range: Tuple[float, float] = (0.2, 0.8),
                          b_range: Tuple[float, float] = (1.0, 5.0),
                          p_steps: int = 7,
                          b_steps: int = 7,
                          num_rounds: int = 500,
                          num_simulations: int = 1000,
                          initial_bankroll: float = 1000.0) -> pd.DataFrame:
        """
        Execute complete parameter sweep
        
        Returns:
            DataFrame with all simulation results
        """
        parameter_grid = self.create_parameter_grid(p_range, b_range, p_steps, b_steps)
        
        all_results = []
        summary_data = []
        
        total_combinations = len(parameter_grid)
        print(f"Starting parameter sweep: {total_combinations} combinations")
        print(f"Grid: p={p_range} ({p_steps} steps), b={b_range} ({b_steps} steps)")
        
        for idx, (p, b) in enumerate(parameter_grid):
            print(f"\n--- Combination {idx+1}/{total_combinations} ---")
            
            # Skip invalid combinations (negative Kelly)
            kelly_fraction = self.kelly_engine.calculate_kelly_fraction(p, b)
            if kelly_fraction <= 0:
                print(f"Skipping p={p:.3f}, b={b:.2f} (negative Kelly fraction)")
                continue
            
            # Create simulation parameters
            params = SimulationParams(
                win_prob=p,
                payout_multiplier=b,
                initial_bankroll=initial_bankroll,
                num_rounds=num_rounds,
                num_simulations=num_simulations
            )
            
            # Run simulation batch
            batch_results = self.kelly_engine.run_monte_carlo_batch(params)
            all_results.append(batch_results)
            
            # Calculate summary statistics
            stats = self.kelly_engine.calculate_batch_statistics(batch_results)
            stats.update({
                'win_prob': p,
                'payout_multiplier': b,
                'combination_id': idx
            })
            summary_data.append(stats)
        
        # Combine all results
        if all_results:
            self.sweep_results = pd.concat(all_results, ignore_index=True)
            self.summary_stats = pd.DataFrame(summary_data)
            
            print(f"\nSweep completed! {len(self.summary_stats)} valid combinations")
            return self.sweep_results
        else:
            print("No valid parameter combinations found!")
            return pd.DataFrame()
    
    def get_summary_matrix(self, metric: str = 'mean_geometric_return') -> pd.DataFrame:
        """
        Convert summary statistics to matrix format for heatmap visualization
        
        Args:
            metric: Which metric to extract as matrix
            
        Returns:
            Pivot table with p as rows, b as columns
        """
        if self.summary_stats.empty:
            return pd.DataFrame()
        
        return self.summary_stats.pivot(
            index='win_prob', 
            columns='payout_multiplier', 
            values=metric
        )
    
    def find_optimal_parameters(self, metric: str = 'mean_geometric_return') -> Dict:
        """
        Find parameter combination that maximizes given metric
        
        Args:
            metric: Optimization target metric
            
        Returns:
            Dict with optimal parameters and metric value
        """
        if self.summary_stats.empty:
            return {}
        
        optimal_idx = self.summary_stats[metric].idxmax()
        optimal_row = self.summary_stats.loc[optimal_idx]
        
        return {
            'win_prob': optimal_row['win_prob'],
            'payout_multiplier': optimal_row['payout_multiplier'],
            'metric_value': optimal_row[metric],
            'kelly_fraction': optimal_row['kelly_fraction'],
            'ruin_probability': optimal_row['ruin_probability'],
            'mean_max_drawdown': optimal_row['mean_max_drawdown']
        }
    
    def export_results(self, detailed_filename: str = "kelly_detailed_results.csv",
                      summary_filename: str = "kelly_summary_results.csv"):
        """Export results to CSV files"""
        if not self.sweep_results.empty:
            # Export detailed results (excluding sample histories for size)
            export_df = self.sweep_results.drop(columns=['sample_histories'], errors='ignore')
            export_df.to_csv(detailed_filename, index=False)
            print(f"Detailed results exported to {detailed_filename}")
        
        if not self.summary_stats.empty:
            self.summary_stats.to_csv(summary_filename, index=False)
            print(f"Summary statistics exported to {summary_filename}")
    
    def get_parameter_recommendations(self) -> Dict:
        """Get recommendations based on different optimization criteria"""
        if self.summary_stats.empty:
            return {}
        
        recommendations = {}
        
        # Best geometric return
        best_growth = self.find_optimal_parameters('mean_geometric_return')
        recommendations['best_growth'] = best_growth
        
        # Lowest risk (min drawdown)
        min_drawdown_idx = self.summary_stats['mean_max_drawdown'].idxmin()
        recommendations['lowest_risk'] = {
            'win_prob': self.summary_stats.loc[min_drawdown_idx, 'win_prob'],
            'payout_multiplier': self.summary_stats.loc[min_drawdown_idx, 'payout_multiplier'],
            'mean_max_drawdown': self.summary_stats.loc[min_drawdown_idx, 'mean_max_drawdown'],
            'mean_geometric_return': self.summary_stats.loc[min_drawdown_idx, 'mean_geometric_return']
        }
        
        # Best risk-adjusted (Sharpe approximation)
        if 'sharpe_approximation' in self.summary_stats.columns:
            best_sharpe = self.find_optimal_parameters('sharpe_approximation')
            recommendations['best_risk_adjusted'] = best_sharpe
        
        # Lowest ruin probability with decent returns
        decent_returns = self.summary_stats[self.summary_stats['mean_geometric_return'] > 0.01]
        if not decent_returns.empty:
            safest_idx = decent_returns['ruin_probability'].idxmin()
            recommendations['safest_profitable'] = {
                'win_prob': decent_returns.loc[safest_idx, 'win_prob'],
                'payout_multiplier': decent_returns.loc[safest_idx, 'payout_multiplier'],
                'ruin_probability': decent_returns.loc[safest_idx, 'ruin_probability'],
                'mean_geometric_return': decent_returns.loc[safest_idx, 'mean_geometric_return']
            }
        
        return recommendations