# Kelly Criterion Monte Carlo Simulator

An interactive Streamlit dashboard for analyzing and visualizing the Kelly betting strategy using Monte Carlo simulations.

## Features
- Parameter sweep across win probabilities and payout multipliers
- Monte Carlo simulation of betting sequences
- Interactive heatmaps and risk-return visualizations
- Strategy recommendations and summary statistics
- Exportable results (CSV)

## Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Aarush-Dubey/kelly-criterion-simulator.git
   cd kelly-criterion-simulator
   ```
2. **Create a virtual environment (optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Run the Streamlit app:**
   ```bash
   streamlit run streamlit_dashboard.py
   ```
2. **Open your browser** to the provided local URL (usually http://localhost:8501).
3. **Configure parameters** in the sidebar and run simulations.
4. **Explore** the results, visualizations, and export options.


## Dependencies
- Python 3.8+
- streamlit
- pandas
- numpy
- plotly


## License
MIT License

---
For educational and research purposes only. 