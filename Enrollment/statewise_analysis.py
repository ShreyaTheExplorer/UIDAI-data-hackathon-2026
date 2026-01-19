import pandas as pd
import matplotlib.pyplot as plt

# 1. Load Data
f = r'1statecleaned.parquet'
df2 = pd.read_parquet(f)
# 2. Filter the DataFrame for month 9
df = df2[df2['date'].dt.month == 9]

# 2. Group by State and Sum ALL age columns
# We select all 3 age columns here
age_cols = ['age_0_5', 'age_5_17', 'age_18_greater']
state_data = df.groupby('state')[age_cols].sum()

# 3. Sort by TOTAL enrollment
# This ensures states are ordered by size (e.g., MP first, then Maharashtra...)
# We create a temporary 'total' column just for sorting
state_data['total'] = state_data.sum(axis=1)
state_data_sorted = state_data.sort_values('total', ascending=False)

# Remove the 'total' column so it doesn't get plotted as a bar
state_data_sorted = state_data_sorted.drop(columns=['total'])
print(state_data_sorted)


def plot_aadhar_clusters(df):
    """
    Plots a clustered bar chart of Aadhar enrollments by age group for each state.
    
    Args:
    df (pd.DataFrame): DataFrame containing 'age_0_5', 'age_5_17', 'age_18_greater'.
                       If 'state' is a column, it is set as the index.
    """
    # Create a copy to avoid modifying the original dataframe
    plot_df = df.copy()
    
    # 1. Ensure 'state' is the index so it appears on the X-axis
    if 'state' in plot_df.columns:
        plot_df = plot_df.set_index('state')
    
    # 2. Plotting
    # kind='bar' automatically clusters the columns (age groups) for each index (state)
    ax = plot_df.plot(
        kind='bar', 
        figsize=(15, 8),     # Large width to accommodate all Indian states
        width=0.8,           # Width of the cluster
        colormap='viridis'   # Optional: nice color scheme
    )
    
    # 3. Customization
    plt.title('Aadhar Enrollment Counts by State and Age Group (September 2025)', fontsize=16)
    plt.xlabel('State', fontsize=12)
    plt.ylabel('Enrollment Count', fontsize=12)
    plt.xticks(rotation=90)  # Rotate state names 90 degrees to fit all of them
    plt.legend(title='Age Group')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()       # Adjust layout to prevent clipping of labels
    
    plt.show()

# Call the function
plot_aadhar_clusters(state_data_sorted)
