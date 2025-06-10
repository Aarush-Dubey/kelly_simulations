"""
Visualization Utilities
=======================
Charts and plots for Kelly Criterion simulation results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import List, Dict, Optional

class KellyVisualizer:
    """Visualization utilities for Kelly simulation results"""
    
    def __init__(self):
        self.setup_style()
    
    def setup_style(self):
        """Configure plotting style"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def create_heatmap(self, matrix_data: pd.DataFrame, 
                      title: str = "Heatmap",
                      cmap: str = "RdYlGn",
                      figsize: tuple = (10, 8)) -> plt.Figure:
        """
        Create heatmap from matrix data
        
        Args:
            matrix_data: DataFrame with metrics in matrix form
            title: Plot title
            cmap: Colormap name
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Handle NaN values
        matrix_clean = matrix_data.fillna(0)
        
        sns.heatmap(matrix_clean, 
                   annot=True, 
                   fmt='.3f', 
                   cmap=cmap,
                   ax=ax,
                   cbar_kws={'label': title})
        
        ax.set_title(f'{title}\n(Win Probability vs. Payout Multiplier)', fontsize=14, pad=20)
        ax.set_xlabel('Payout Multiplier (b)', fontsize=12)
        ax.set_ylabel('Win Probability (p)', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def create_interactive_heatmap(self, matrix_data: pd.DataFrame, 
                                 title: str = "Interactive Heatmap") -> go.Figure:
        """Create interactive heatmap using Plotly"""
        fig = px.imshow(matrix_data,
                       labels=dict(x="Payout Multiplier (b)", 
                                 y="Win Probability (p)", 
                                 color=title),
                       aspect="auto",
                       color_continuous_scale="RdYlGn")
        
        fig.update_layout(
            title=f'{title}<br><sub>Win Probability vs. Payout Multiplier</sub>',
            title_x=0.5,
            width=700,
            height=500
        )
        
        return fig
    
    def plot_bankroll_evolution(self, histories: List[np.ndarray], 
                              title: str = "Bankroll Evolution",
                              max_paths: int = 50) -> plt.Figure:
        """
        Plot sample bankroll evolution paths
        
        Args:
            histories: List of bankroll history arrays
            title: Plot title
            max_paths: Maximum number of paths to plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Select subset of paths to avoid overcrowding
        selected_histories = histories[:min(max_paths, len(histories))]
        
        for i, history in enumerate(selected_histories):
            alpha = 0.6 if len(selected_histories) <= 20 else 0.3
            ax.plot(history, alpha=alpha, linewidth=1)
        
        # Add mean path
        if histories:
            mean_history = np.mean([h for h in histories if len(h) > 0], axis=0)
            ax.plot(mean_history, color='red', linewidth=3, label='Mean Path')
            ax.legend()
        
        ax.set_title(f'{title}\n({len(selected_histories)} Sample Paths)', fontsize=14)
        ax.set_xlabel('Betting Round', fontsize=12)
        ax.set_ylabel('Bankroll Value', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_distribution(self, data: pd.Series, 
                         title: str = "Distribution",
                         bins: int = 50) -> plt.Figure:
        """
        Plot distribution with statistics
        
        Args:
            data: Series of values to plot
            title: Plot title
            bins: Number of histogram bins
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(data, bins=bins, alpha=0.7, density=True, edgecolor='black')
        ax1.axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.2f}')
        ax1.axvline(data.median(), color='orange', linestyle='--', label=f'Median: {data.median():.2f}')
        ax1.set_title(f'{title} - Histogram')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(data, vert=True)
        ax2.set_title(f'{title} - Box Plot')
        ax2.set_ylabel('Value')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"""
        Count: {len(data):,}
        Mean: {data.mean():.3f}
        Std: {data.std():.3f}
        Min: {data.min():.3f}
        Q25: {data.quantile(0.25):.3f}
        Median: {data.median():.3f}
        Q75: {data.quantile(0.75):.3f}
        Max: {data.max():.3f}
        """
        
        fig.text(0.02, 0.98, stats_text, transform=fig.transFigure, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def create_metric_comparison(self, summary_df: pd.DataFrame, 
                               metrics: List[str] = None) -> plt.Figure:
        """
        Create comparison plot of multiple metrics
        
        Args:
            summary_df: Summary statistics DataFrame
            metrics: List of metrics to compare
            
        Returns:
            Matplotlib figure
        """
        if metrics is None:
            metrics = ['mean_geometric_return', 'mean_max_drawdown', 'ruin_probability']
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            if metric in summary_df.columns:
                matrix = summary_df.pivot(index='win_prob', 
                                        columns='payout_multiplier', 
                                        values=metric)
                
                sns.heatmap(matrix, annot=True, fmt='.3f', ax=axes[i],
                           cmap='RdYlGn' if 'return' in metric else 'RdYlGn_r')
                axes[i].set_title(metric.replace('_', ' ').title())
        
        plt.tight_layout()
        return fig
    
    def create_kelly_fraction_surface(self, p_range: tuple = (0.1, 0.9),
                                    b_range: tuple = (1.0, 5.0),
                                    resolution: int = 50) -> go.Figure:
        """
        Create 3D surface plot of Kelly fractions
        
        Args:
            p_range: Range of win probabilities
            b_range: Range of payout multipliers
            resolution: Grid resolution
            
        Returns:
            Plotly 3D figure
        """
        p_vals = np.linspace(p_range[0], p_range[1], resolution)
        b_vals = np.linspace(b_range[0], b_range[1], resolution)
        
        P, B = np.meshgrid(p_vals, b_vals)
        
        # Calculate Kelly fractions
        Kelly = np.zeros_like(P)
        for i in range(len(p_vals)):
            for j in range(len(b_vals)):
                p, b = p_vals[i], b_vals[j]
                kelly_frac = max(0, (b * p - (1-p)) / b)
                Kelly[j, i] = min(kelly_frac, 0.5)  # Cap at 50%
        
        fig = go.Figure(data=[go.Surface(z=Kelly, x=p_vals, y=b_vals,
                                       colorscale='Viridis')])
        
        fig.update_layout(
            title='Kelly Fraction Surface<br><sub>Optimal Bet Size as Function of p and b</sub>',
            scene=dict(
                xaxis_title='Win Probability (p)',
                yaxis_title='Payout Multiplier (b)',
                zaxis_title='Kelly Fraction'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def create_performance_dashboard(self, summary_df: pd.DataFrame) -> Dict:
        """
        Create comprehensive performance dashboard
        
        Args:
            summary_df: Summary statistics DataFrame
            
        Returns:
            Dictionary of figures
        """
        figures = {}
        
        # Key metrics heatmaps
        key_metrics = ['mean_geometric_return', 'ruin_probability', 'mean_max_drawdown']
        
        for metric in key_metrics:
            if metric in summary_df.columns:
                matrix = summary_df.pivot(index='win_prob', 
                                        columns='payout_multiplier', 
                                        values=metric)
                figures[metric] = self.create_interactive_heatmap(matrix, metric.replace('_', ' ').title())
        
        # Kelly fraction surface
        figures['kelly_surface'] = self.create_kelly_fraction_surface()
        
        return figures
    
    def plot_risk_return_scatter(self, summary_df: pd.DataFrame) -> plt.Figure:
        """
        Create risk-return scatter plot
        
        Args:
            summary_df: Summary statistics DataFrame
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(summary_df['mean_max_drawdown'], 
                           summary_df['mean_geometric_return'],
                           c=summary_df['ruin_probability'],
                           s=summary_df['kelly_fraction'] * 500,  # Size by Kelly fraction
                           alpha=0.7,
                           cmap='RdYlGn_r')
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Ruin Probability')
        
        ax.set_xlabel('Mean Maximum Drawdown')
        ax.set_ylabel('Mean Geometric Return')
        ax.set_title('Risk-Return Analysis\n(Point size = Kelly Fraction, Color = Ruin Probability)')
        ax.grid(True, alpha=0.3)
        
        # Add annotations for extreme points
        best_return_idx = summary_df['mean_geometric_return'].idxmax()
        lowest_risk_idx = summary_df['mean_max_drawdown'].idxmin()
        
        ax.annotate('Best Return', 
                   xy=(summary_df.loc[best_return_idx, 'mean_max_drawdown'],
                       summary_df.loc[best_return_idx, 'mean_geometric_return']),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.annotate('Lowest Risk', 
                   xy=(summary_df.loc[lowest_risk_idx, 'mean_max_drawdown'],
                       summary_df.loc[lowest_risk_idx, 'mean_geometric_return']),
                   xytext=(10, -10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        return fig