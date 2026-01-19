import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

# --- 1. Load Data (Simulated for this example, replace with your file) ---
file_path = "1statecleaned.parquet"
df = pd.read_parquet(file_path)


if not pd.api.types.is_datetime64_any_dtype(df['date']):
    df['date'] = pd.to_datetime(df['date'])

# --- 2. Core Analysis Functions ---

def analyze_district_performance(state, district, month_number=9):
    """
    Returns a dataframe with enrollment stats and 'Status' (Underserved/Overserved/Normal)
    """
    # Filter Data
    mask = (df['state'] == state) & (df['district'] == district) & (df['date'].dt.month == month_number)
    local_df = df[mask].copy()
    
    if local_df.empty:
        return pd.DataFrame()

    # Aggregate by Pincode
    stats = local_df.groupby('pincode')['age_5_17'].sum().reset_index()
    stats.rename(columns={'age_5_17': 'enrollment'}, inplace=True)
    
    # Calculate Stats
    mu = stats['enrollment'].mean()
    sigma = stats['enrollment'].std()
    
    # thresholds
    low_thresh = max(0, mu - 1.0 * sigma) # Don't go below 0
    high_thresh = mu + 2.0 * sigma
    
    # Assign Status
    def get_status(val):
        if val < low_thresh: return 'Underserved'
        elif val > high_thresh: return 'Overserved'
        else: return 'Normal'
        
    stats['status'] = stats['enrollment'].apply(get_status)
    return stats, mu, sigma, low_thresh, high_thresh

# --- 3. Visualization Functions ---

def plot_map_proxy(stats_df, district):
    """
    Simulates a map using a scatter plot since we don't have Lat/Lon in this dataset.
    If you have Lat/Lon columns, change x='lon', y='lat'.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create colors
    colors = stats_df['status'].map({
        'Underserved': 'red', 
        'Overserved': 'green', 
        'Normal': 'lightgray'
    })
    
    # Plot "Map" (Using Pincode as a proxy for location if real lat/lon missing)
    # We sort by pincode to approximate "clustering"
    x_vals = range(len(stats_df)) 
    
    sc = ax.scatter(x_vals, stats_df['enrollment'], c=colors, s=100, alpha=0.7, edgecolors='black')
    
    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Underserved (Low)', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Overserved (High)', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', label='Normal', markersize=10)
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_title(f"Enrollment Distribution in {district} (Map Proxy)", fontsize=14)
    ax.set_xlabel("Pincode Index (Sorted)")
    ax.set_ylabel("Enrollment Count")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def show_styled_table(stats_df):
    """
    Displays a color-coded table of interesting pincodes only.
    """
    # Filter for interesting ones
    interesting = stats_df[stats_df['status'] != 'Normal'].copy()
    
    if interesting.empty:
        print("No critical Underserved or Overserved areas found.")
        return

    # Style the dataframe
    def color_status(val):
        color = 'red' if val == 'Underserved' else 'green'
        return f'color: {color}; font-weight: bold'

    styler = interesting.style.map(color_status, subset=['status'])\
                        .background_gradient(subset=['enrollment'], cmap='Reds')
    
    display(styler)

# --- 4. The Dashboard UI (Fixed) ---

class Dashboard:
    def __init__(self):
        # Widgets
        self.w_state = widgets.Dropdown(options=sorted(df['state'].unique()), description='State:')
        self.w_district = widgets.Dropdown(description='District:', disabled=True)
        self.w_btn = widgets.Button(description='Analyze', button_style='info')
        self.out = widgets.Output()
        
        # Observers
        self.w_state.observe(self.on_state_change, names='value')
        self.w_btn.on_click(self.run_analysis)
        
        # Layout
        self.ui = widgets.VBox([
            widgets.HBox([self.w_state, self.w_district, self.w_btn]),
            self.out
        ])
        
        # Initialize District for first state
        self.on_state_change(None)

    def on_state_change(self, change):
        # Update district options based on selected state
        state = self.w_state.value
        districts = sorted(df[df['state'] == state]['district'].unique())
        self.w_district.options = districts
        self.w_district.disabled = False
        self.w_district.value = districts[0] if districts else None

    def run_analysis(self, b):
        self.out.clear_output()
        with self.out:
            state = self.w_state.value
            district = self.w_district.value
            
            # Run Logic
            stats, mu, sigma, low, high = analyze_district_performance(state, district)
            
            if stats.empty:
                print("No data found.")
                return
            
            # Create Tabs for View (Table vs Map)
            tab = widgets.Tab()
            out_table = widgets.Output()
            out_map = widgets.Output()
            
            tab.children = [out_table, out_map]
            tab.set_title(0, 'List View')
            tab.set_title(1, 'Map View')
            
            display(HTML(f"<h3>District: {district}</h3>"))
            display(HTML(f"<b>Avg Enrollment:</b> {int(mu)} | <b>Low Threshold:</b> <{int(low)} | <b>High Threshold:</b> >{int(high)}"))
            
            with out_table:
                show_styled_table(stats)
                
            with out_map:
                plot_map_proxy(stats, district)
                
            display(tab)

# Run the Dashboard
d = Dashboard()
display(d.ui)
