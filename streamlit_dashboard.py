"""
Kelly Criterion Simulation Dashboard
===================================
Interactive Streamlit dashboard for Kelly betting strategy analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from kelly_monte_carlo import KellyEngine, SimulationParams
from parameter_sweep import ParameterSweep
from visualization_utils import KellyVisualizer
import time
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Kelly Criterion Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #f0f2f6, #ffffff);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'sweep_results' not in st.session_state:
    st.session_state.sweep_results = pd.DataFrame()
if 'summary_stats' not in st.session_state:
    st.session_state.summary_stats = pd.DataFrame()
if 'simulation_complete' not in st.session_state:
    st.session_state.simulation_complete = False

# Initialize engines
kelly_engine = KellyEngine()
sweep_engine = ParameterSweep()
visualizer = KellyVisualizer()

# Header
st.markdown('<h1 class="main-header">🎯 Kelly Criterion Monte Carlo Simulator</h1>', 
           unsafe_allow_html=True)

st.markdown("""
**Objective:** Empirically evaluate the Kelly Criterion betting strategy across different 
probability and payout scenarios using Monte Carlo simulation.
""")

# Sidebar Configuration
st.sidebar.header("⚙️ Simulation Parameters")

# Parameter ranges
st.sidebar.subheader("Parameter Ranges")
p_min, p_max = st.sidebar.slider(
    "Win Probability Range", 
    min_value=0.1, max_value=0.9, 
    value=(0.3, 0.7), step=0.05,
    help="Range of win probabilities to test"
)

b_min, b_max = st.sidebar.slider(
    "Payout Multiplier Range", 
    min_value=1.0, max_value=5.0, 
    value=(1.5, 3.0), step=0.1,
    help="Range of payout multipliers (odds) to test"
)

# Grid resolution
st.sidebar.subheader("Grid Resolution")
p_steps = st.sidebar.slider("Probability Steps", 3, 10, 5)
b_steps = st.sidebar.slider("Payout Steps", 3, 10, 5)

# Simulation settings
st.sidebar.subheader("Simulation Settings")
num_simulations = st.sidebar.selectbox(
    "Number of Simulations per Scenario",
    [100, 500, 1000, 2000, 5000],
    index=2,
    help="More simulations = better accuracy but longer runtime"
)

num_rounds = st.sidebar.slider(
    "Betting Rounds per Simulation",
    100, 1000, 500, step=50,
    help="Number of betting rounds in each simulation path"
)

initial_bankroll = st.sidebar.number_input(
    "Initial Bankroll",
    min_value=100.0, max_value=10000.0,
    value=1000.0, step=100.0
)

# Run simulation button
run_simulation = st.sidebar.button(
    "🚀 Run Parameter Sweep",
    type="primary",
    help="Start Monte Carlo simulation across parameter grid"
)

# Display current settings
with st.sidebar.expander("📋 Current Settings Summary"):
    st.write(f"**Grid Size:** {p_steps} × {b_steps} = {p_steps * b_steps} combinations")
    st.write(f"**Total Simulations:** {p_steps * b_steps * num_simulations:,}")
    st.write(f"**Estimated Runtime:** ~{(p_steps * b_steps * num_simulations) / 1000:.1f} minutes")

# Main content area
if run_simulation or st.session_state.simulation_complete:
    if run_simulation:
        # Run parameter sweep
        with st.spinner("🔄 Running Monte Carlo simulations..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            start_time = time.time()
            
            # Update progress display
            status_text.text("Initializing parameter grid...")
            progress_bar.progress(0.1)
            
            # Run sweep
            status_text.text("Running simulations...")
            sweep_results = sweep_engine.run_parameter_sweep(
                p_range=(p_min, p_max),
                b_range=(b_min, b_max),
                p_steps=p_steps,
                b_steps=b_steps,
                num_rounds=num_rounds,
                num_simulations=num_simulations,
                initial_bankroll=initial_bankroll
            )
            
            progress_bar.progress(0.8)
            status_text.text("Processing results...")
            
            # Store results in session state
            st.session_state.sweep_results = sweep_results
            st.session_state.summary_stats = sweep_engine.summary_stats
            st.session_state.simulation_complete = True
            
            progress_bar.progress(1.0)
            runtime = time.time() - start_time
            status_text.text(f"✅ Simulation completed in {runtime:.1f} seconds!")
            
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
    
    # Display results if available
    if not st.session_state.summary_stats.empty:
        summary_df = st.session_state.summary_stats
        
        # Key metrics overview
        st.header("📊 Simulation Results Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Scenarios Tested",
                f"{len(summary_df):,}",
                help="Number of valid (p,b) combinations tested"
            )
        
        with col2:
            best_growth = summary_df['mean_geometric_return'].max()
            st.metric(
                "Best Geometric Return",
                f"{best_growth:.3f}",
                help="Highest mean geometric return achieved"
            )
        
        with col3:
            min_ruin = summary_df['ruin_probability'].min()
            st.metric(
                "Lowest Ruin Risk",
                f"{min_ruin:.1%}",
                help="Lowest probability of bankroll ruin"
            )
        
        with col4:
            avg_kelly = summary_df['kelly_fraction'].mean()
            st.metric(
                "Average Kelly Fraction",
                f"{avg_kelly:.3f}",
                help="Average optimal bet fraction across scenarios"
            )
        
        # Recommendations
        st.header("🎯 Strategy Recommendations")
        recommendations = sweep_engine.get_parameter_recommendations()
        
        if recommendations:
            rec_col1, rec_col2 = st.columns(2)
            
            with rec_col1:
                if 'best_growth' in recommendations:
                    best = recommendations['best_growth']
                    st.success(f"""
                    **🚀 Maximum Growth Strategy**
                    - Win Probability: {best['win_prob']:.3f}
                    - Payout Multiplier: {best['payout_multiplier']:.2f}
                    - Kelly Fraction: {best['kelly_fraction']:.3f}
                    - Expected Growth: {best['metric_value']:.3f}
                    """)
            
            with rec_col2:
                if 'safest_profitable' in recommendations:
                    safe = recommendations['safest_profitable']
                    st.info(f"""
                    **🛡️ Safest Profitable Strategy**
                    - Win Probability: {safe['win_prob']:.3f}
                    - Payout Multiplier: {safe['payout_multiplier']:.2f}
                    - Ruin Risk: {safe['ruin_probability']:.1%}
                    - Expected Growth: {safe['mean_geometric_return']:.3f}
                    """)
        
        # Interactive visualizations
        st.header("📈 Interactive Analysis")
        
        # Metric selection
        viz_col1, viz_col2 = st.columns([1, 3])
        
        with viz_col1:
            available_metrics = [
                'mean_geometric_return',
                'ruin_probability', 
                'mean_max_drawdown',
                'median_final_bankroll',
                'positive_return_rate'
            ]
            
            selected_metric = st.selectbox(
                "Select Metric for Heatmap",
                available_metrics,
                format_func=lambda x: x.replace('_', ' ').title()
            )
        
        with viz_col2:
            # Create heatmap
            if selected_metric in summary_df.columns:
                matrix = summary_df.pivot_table(
                    index='win_prob',
                    columns='payout_multiplier', 
                    values=selected_metric,
                    aggfunc='mean'
                )
                
                fig = visualizer.create_interactive_heatmap(
                    matrix, 
                    selected_metric.replace('_', ' ').title()
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Detailed analysis tabs
        st.header("🔍 Detailed Analysis")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Performance Heatmaps",
            "📉 Risk-Return Analysis", 
            "🎲 Kelly Surface",
            "📋 Data Tables"
        ])
        
        with tab1:
            # Multiple heatmaps
            heatmap_col1, heatmap_col2 = st.columns(2)
            
            key_metrics = ['mean_geometric_return', 'ruin_probability']
            
            for i, metric in enumerate(key_metrics):
                if metric in summary_df.columns:
                    matrix = summary_df.pivot_table(
                        index='win_prob',
                        columns='payout_multiplier',
                        values=metric,
                        aggfunc='mean'
                    )
                    
                    fig = visualizer.create_interactive_heatmap(
                        matrix,
                        metric.replace('_', ' ').title()
                    )
                    
                    if i == 0:
                        heatmap_col1.plotly_chart(fig, use_container_width=True, key=f"heatmap_{metric}")
                    else:
                        heatmap_col2.plotly_chart(fig, use_container_width=True, key=f"heatmap_{metric}")
        
        with tab2:
            # Risk-return scatter plot
            if len(summary_df) > 1:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=summary_df['mean_max_drawdown'],
                    y=summary_df['mean_geometric_return'],
                    mode='markers',
                    marker=dict(
                        size=summary_df['kelly_fraction'] * 50,
                        color=summary_df['ruin_probability'],
                        colorscale='RdYlGn_r',
                        showscale=True,
                        colorbar=dict(title="Ruin Probability")
                    ),
                    text=[f"p={row['win_prob']:.2f}, b={row['payout_multiplier']:.2f}" 
                          for _, row in summary_df.iterrows()],
                    hovertemplate="<b>%{text}</b><br>" +
                                "Drawdown: %{x:.3f}<br>" +
                                "Return: %{y:.3f}<br>" +
                                "<extra></extra>"
                ))
                
                fig.update_layout(
                    title="Risk-Return Analysis<br><sub>Point size = Kelly Fraction</sub>",
                    xaxis_title="Mean Maximum Drawdown",
                    yaxis_title="Mean Geometric Return",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Kelly fraction surface
            st.subheader("Kelly Fraction Surface")
            kelly_surface = visualizer.create_kelly_fraction_surface(
                p_range=(p_min, p_max),
                b_range=(b_min, b_max)
            )
            st.plotly_chart(kelly_surface, use_container_width=True)
            
            st.info("""
            **Kelly Formula:** f* = (bp - q) / b
            - f* = optimal fraction of bankroll to bet
            - b = payout multiplier (odds)  
            - p = probability of winning
            - q = probability of losing (1-p)
            """)
        
        with tab4:
            # Data tables
            st.subheader("Summary Statistics")
            
            # Format the dataframe for display
            display_df = summary_df.copy()
            
            # Round numeric columns
            numeric_columns = display_df.select_dtypes(include=[np.number]).columns
            display_df[numeric_columns] = display_df[numeric_columns].round(4)
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )
            
            # Export functionality
            st.subheader("📥 Export Results")
            
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                # Summary export
                csv_summary = summary_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Summary CSV",
                    data=csv_summary,
                    file_name=f"kelly_summary_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with export_col2:
                # Detailed results export (if available)
                if not st.session_state.sweep_results.empty:
                    detailed_df = st.session_state.sweep_results.drop(
                        columns=['sample_histories'], errors='ignore'
                    )
                    csv_detailed = detailed_df.to_csv(index=False)
                    st.download_button(
                        label="📋 Download Detailed CSV",
                        data=csv_detailed,
                        file_name=f"kelly_detailed_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

else:
    # Welcome screen
    st.header("🎯 Welcome to Kelly Criterion Monte Carlo Simulator")
    
    st.markdown("""
    ### What is the Kelly Criterion?
    
    The **Kelly Criterion** is a mathematical formula used to determine the optimal size of a series of bets. 
    It maximizes the logarithm of wealth over time, balancing growth potential with risk management.
    
    **Formula:** f* = (bp - q) / b
    
    Where:
    - **f*** = fraction of bankroll to wager
    - **b** = payout multiplier (odds received)
    - **p** = probability of winning
    - **q** = probability of losing (1-p)
    
    ### This Simulator
    
    This tool runs **Monte Carlo simulations** across different probability and payout scenarios to:
    
    - 🎲 **Simulate** thousands of betting sequences using Kelly sizing
    - 📊 **Analyze** performance metrics like growth rates and drawdowns  
    - 🗺️ **Visualize** optimal parameter regions via interactive heatmaps
    - ⚡ **Compare** risk-return profiles across betting scenarios
    
    ### Get Started
    
    1. **Configure** your parameter ranges in the sidebar
    2. **Set** simulation settings (more sims = better accuracy)
    3. **Click** "Run Parameter Sweep" to start the analysis
    4. **Explore** the interactive results and recommendations
    
    ---
    
    **💡 Pro Tips:**
    - Start with smaller grids (3×3) for quick exploration
    - Higher win probabilities and payout multipliers generally perform better
    - Watch out for high drawdown scenarios even with good average returns
    - The Kelly fraction caps at 50% of bankroll for safety
    """)
    
    # Quick start examples
    st.subheader("🚀 Quick Start Examples")
    
    example_col1, example_col2, example_col3 = st.columns(3)
    
    with example_col1:
        st.info("""
        **🎯 Conservative Sweep**
        - Win Prob: 0.4 - 0.6
        - Payout: 1.5x - 2.5x
        - Grid: 3×3 (9 scenarios)
        - Sims: 500 each
        """)
    
    with example_col2:
        st.warning("""
        **⚡ Balanced Analysis**
        - Win Prob: 0.3 - 0.7  
        - Payout: 1.5x - 3.5x
        - Grid: 5×5 (25 scenarios)
        - Sims: 1,000 each
        """)
    
    with example_col3:
        st.error("""
        **🔥 Comprehensive Study**
        - Win Prob: 0.2 - 0.8
        - Payout: 1.0x - 5.0x  
        - Grid: 7×7 (49 scenarios)
        - Sims: 2,000 each
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🎲 <strong>Kelly Criterion Monte Carlo Simulator</strong> | 
    Built with Streamlit & Python | 
    <em>For educational and research purposes</em></p>
</div>
""", unsafe_allow_html=True)