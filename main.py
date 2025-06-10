"""
Kelly Criterion Simulation - Main Application
============================================
Entry point for the Kelly Criterion Monte Carlo simulation platform
"""

import sys
import os
import argparse
import pandas as pd
from kelly_monte_carlo import KellyEngine, SimulationParams
from parameter_sweep import ParameterSweep
from visualization_utils import KellyVisualizer

def run_command_line_simulation():
    """Run simulation from command line with preset parameters"""
    
    parser = argparse.ArgumentParser(description='Kelly Criterion Monte Carlo Simulation')
    parser.add_argument('--p-min', type=float, default=0.3, help='Minimum win probability')
    parser.add_argument('--p-max', type=float, default=0.7, help='Maximum win probability')
    parser.add_argument('--b-min', type=float, default=1.5, help='Minimum payout multiplier')
    parser.add_argument('--b-max', type=float, default=3.0, help='Maximum payout multiplier')
    parser.add_argument('--p-steps', type=int, default=5, help='Number of probability steps')
    parser.add_argument('--b-steps', type=int, default=5, help='Number of payout steps')
    parser.add_argument('--num-sims', type=int, default=1000, help='Simulations per scenario')
    parser.add_argument('--num-rounds', type=int, default=500, help='Betting rounds per simulation')
    parser.add_argument('--output-dir', type=str, default='./results', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🎯 Kelly Criterion Monte Carlo Simulation")
    print("=" * 50)
    print(f"Parameter Grid: {args.p_steps}×{args.b_steps} = {args.p_steps * args.b_steps} scenarios")
    print(f"Simulations per scenario: {args.num_sims:,}")
    print(f"Total simulations: {args.p_steps * args.b_steps * args.num_sims:,}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Initialize sweep engine
    sweep_engine = ParameterSweep()
    
    # Run parameter sweep
    print("🚀 Starting parameter sweep...")
    sweep_results = sweep_engine.run_parameter_sweep(
        p_range=(args.p_min, args.p_max),
        b_range=(args.b_min, args.b_max),
        p_steps=args.p_steps,
        b_steps=args.b_steps,
        num_rounds=args.num_rounds,
        num_simulations=args.num_sims
    )
    
    if sweep_results.empty:
        print("❌ No valid results generated!")
        return
    
    # Export results
    detailed_file = os.path.join(args.output_dir, "kelly_detailed_results.csv")
    summary_file = os.path.join(args.output_dir, "kelly_summary_results.csv")
    
    sweep_engine.export_results(detailed_file, summary_file)
    
    # Generate recommendations
    print("\n🎯 Strategy Recommendations:")
    print("-" * 30)
    
    recommendations = sweep_engine.get_parameter_recommendations()
    
    if 'best_growth' in recommendations:
        best = recommendations['best_growth']
        print(f"🚀 Maximum Growth Strategy:")
        print(f"   Win Probability: {best['win_prob']:.3f}")
        print(f"   Payout Multiplier: {best['payout_multiplier']:.2f}")
        print(f"   Kelly Fraction: {best['kelly_fraction']:.3f}")
        print(f"   Expected Growth: {best['metric_value']:.4f}")
        print(f"   Ruin Risk: {best['ruin_probability']:.1%}")
    
    if 'safest_profitable' in recommendations:
        safe = recommendations['safest_profitable']
        print(f"\n🛡️ Safest Profitable Strategy:")
        print(f"   Win Probability: {safe['win_prob']:.3f}")
        print(f"   Payout Multiplier: {safe['payout_multiplier']:.2f}")
        print(f"   Ruin Risk: {safe['ruin_probability']:.1%}")
        print(f"   Expected Growth: {safe['mean_geometric_return']:.4f}")
    
    print(f"\n✅ Simulation completed! Results saved to {args.output_dir}")

def run_single_scenario_demo():
    """Run a single scenario demonstration"""
    
    print("🎯 Kelly Criterion - Single Scenario Demo")
    print("=" * 45)
    
    # Demo parameters
    win_prob = 0.55
    payout_multiplier = 2.0
    num_simulations = 5000
    num_rounds = 300
    
    print(f"Win Probability: {win_prob}")
    print(f"Payout Multiplier: {payout_multiplier}x")
    print(f"Simulations: {num_simulations:,}")
    print(f"Rounds per simulation: {num_rounds}")
    print()
    
    # Initialize engine
    kelly_engine = KellyEngine()
    
    # Calculate Kelly fraction
    kelly_fraction = kelly_engine.calculate_kelly_fraction(win_prob, payout_multiplier)
    print(f"📊 Optimal Kelly Fraction: {kelly_fraction:.3f} ({kelly_fraction:.1%} of bankroll)")
    print()
    
    # Create simulation parameters
    params = SimulationParams(
        win_prob=win_prob,
        payout_multiplier=payout_multiplier,
        num_simulations=num_simulations,
        num_rounds=num_rounds
    )
    
    # Run simulation
    print("🚀 Running Monte Carlo simulation...")
    results_df = kelly_engine.run_monte_carlo_batch(params)
    
    # Calculate statistics
    stats = kelly_engine.calculate_batch_statistics(results_df)
    
    # Display results
    print("\n📈 Results Summary:")
    print("-" * 20)
    print(f"Mean Final Bankroll: ${stats['mean_final_bankroll']:.2f}")
    print(f"Median Final Bankroll: ${stats['median_final_bankroll']:.2f}")
    print(f"Mean Geometric Return: {stats['mean_geometric_return']:.4f}")
    print(f"Mean Max Drawdown: {stats['mean_max_drawdown']:.1%}")
    print(f"Ruin Probability: {stats['ruin_probability']:.1%}")
    print(f"Positive Return Rate: {stats['positive_return_rate']:.1%}")
    
    print(f"\n📊 Percentiles:")
    print(f"5th percentile: ${stats['percentile_5']:.2f}")
    print(f"25th percentile: ${stats['percentile_25']:.2f}")
    print(f"75th percentile: ${stats['percentile_75']:.2f}")
    print(f"95th percentile: ${stats['percentile_95']:.2f}")
    
    # Export single scenario results
    results_df.to_csv("single_scenario_results.csv", index=False)
    print(f"\n💾 Results exported to 'single_scenario_results.csv'")

def main():
    """Main entry point"""
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'demo':
            run_single_scenario_demo()
        elif sys.argv[1] == 'streamlit':
            print("🚀 Starting Streamlit dashboard...")
            print("Run: streamlit run streamlit_dashboard.py")
        else:
            run_command_line_simulation()
    else:
        print("🎯 Kelly Criterion Monte Carlo Simulator")
        print("=" * 40)
        print()
        print("Usage options:")
        print("  python main_app.py demo              # Run single scenario demo")
        print("  python main_app.py streamlit         # Instructions for Streamlit")
        print("  python main_app.py [options]         # Run parameter sweep")
        print()
        print("Parameter sweep options:")
        print("  --p-min FLOAT      Minimum win probability (default: 0.3)")
        print("  --p-max FLOAT      Maximum win probability (default: 0.7)")
        print("  --b-min FLOAT      Minimum payout multiplier (default: 1.5)")
        print("  --b-max FLOAT      Maximum payout multiplier (default: 3.0)")
        print("  --p-steps INT      Number of probability steps (default: 5)")
        print("  --b-steps INT      Number of payout steps (default: 5)")
        print("  --num-sims INT     Simulations per scenario (default: 1000)")
        print("  --num-rounds INT   Betting rounds per simulation (default: 500)")
        print("  --output-dir STR   Output directory (default: ./results)")
        print()
        print("Examples:")
        print("  python main_app.py demo")
        print("  python main_app.py --p-steps 3 --b-steps 3 --num-sims 500")
        print("  python main_app.py --p-min 0.4 --p-max 0.6 --output-dir ./my_results")

if __name__ == "__main__":
    main()