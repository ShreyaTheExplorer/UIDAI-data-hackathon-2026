# ============================================================
# Future Load Prediction (Granular: District & Pincode)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import os
from prophet import Prophet

# -----------------------------
# 1. Configuration & File Paths
# -----------------------------
BIO_FILES = [
    'c:/Users/khatr/UIDAI_Hackathon/Biometric/api_data_aadhar_biometric_0_500000.csv',
    'c:/Users/khatr/UIDAI_Hackathon/Biometric/api_data_aadhar_biometric_500000_1000000.csv',
    'c:/Users/khatr/UIDAI_Hackathon/Biometric/api_data_aadhar_biometric_1000000_1500000.csv',
    'c:/Users/khatr/UIDAI_Hackathon/Biometric/api_data_aadhar_biometric_1500000_1861108.csv'
]

# -----------------------------
# 2. Data Loading and Aggregation
# -----------------------------
print("Loading and aggregating ONLY Biometric data...")

# Normalization mapping for inconsistent district names
DISTRICT_MAP = {
    'Bangalore': 'Bengaluru Urban',
    'Bangalore Urban': 'Bengaluru Urban',
    'Bengaluru': 'Bengaluru Urban',
    'Allahabad': 'Prayagraj',
    'Gurgaon': 'Gurugram',
    'Mysore': 'Mysuru',
    'Tumkur': 'Tumakuru',
    'Gulbarga': 'Kalaburagi',
    'Shimoga': 'Shivamogga',
    'Belgaum': 'Belagavi',
    'Bellary': 'Ballari',
    'Hasan': 'Hassan',
    'Vijayapura': 'Vijayapura', # Consistency
    'Bijapur': 'Vijayapura' # Karnataka Bijapur is Vijayapura
}

def normalize_district(name):
    if not isinstance(name, str):
        return name
    name = name.strip().title()
    return DISTRICT_MAP.get(name, name)

dfs = []

# Load Biometric Data
for f in BIO_FILES:
    if os.path.exists(f):
        print(f"Reading {os.path.basename(f)}...")
        temp = pd.read_csv(f)
        # Sum age groups for total biometric load
        temp['load'] = temp['bio_age_5_17'] + temp['bio_age_17_']
        # Normalize district names
        temp['district'] = temp['district'].apply(normalize_district)
        dfs.append(temp[['date', 'district', 'pincode', 'load']])

if not dfs:
    print("Error: No biometric data files found.")
    exit()

df = pd.concat(dfs, ignore_index=True)
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
print(f"Total historical biometric records: {len(df)}")

# -----------------------------
# 3. Global Trend Forecasting
# -----------------------------
print("\nTraining Global Prophet Model on Biometric data...")

# Group by date for global daily load
daily_global = df.groupby('date')['load'].sum().reset_index()
daily_global.columns = ['ds', 'y']

# Using a VERY conservative changepoint_prior_scale to avoid aggressive linear growth
# 0.001 forces the model to stick closer to the average growth rate rather than recent spikes.
m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.001 
)
m.fit(daily_global)

# Forecast until the end of 2026
last_hist_date = daily_global['ds'].max()
end_of_2026 = pd.to_datetime('2026-12-31')
days_to_forecast = (end_of_2026 - last_hist_date).days

print(f"Forecasting {days_to_forecast} days until the end of 2026...")
future = m.make_future_dataframe(periods=days_to_forecast, freq='D')
forecast = m.predict(future)

# Identify the maximum predicted DAILY load in the future period
future_preds = forecast[forecast['ds'] > last_hist_date]
max_future_daily_load = future_preds['yhat'].max()

print(f"Maximum Predicted Global Daily Biometric Load: {int(max_future_daily_load):,}")

# -----------------------------
# 4. Granular Distribution Logic (Peak Load)
# -----------------------------
print("\nCalculating Granular Future Biometric Load (District & Pincode share of PEAK)...")

# Calculate historical volume share for each (District, Pincode)
granular_stats = df.groupby(['district', 'pincode'])['load'].sum().reset_index()
total_hist_load = granular_stats['load'].sum()
granular_stats['share'] = granular_stats['load'] / total_hist_load

# Distribute the global peak load based on historical share
granular_stats['future_biometric_load'] = (granular_stats['share'] * max_future_daily_load).round().astype(int)

# -----------------------------
# 5. Save and Export
# -----------------------------
output_csv = "future_load.csv"
granular_stats[['district', 'pincode', 'future_biometric_load']].to_csv(output_csv, index=False)
print(f"Future biometric load predictions saved to: {output_csv}")

# -----------------------------
# 6. Visualization 1: Top 20 Districts (Annual Peak)
# -----------------------------
print("\nGenerating Top 20 Districts Plot...")

district_future_load = granular_stats.groupby('district')['future_biometric_load'].sum().sort_values(ascending=False).head(20)

plt.figure(figsize=(12, 8))
district_future_load.plot(kind='bar', color='salmon', edgecolor='darkred')

plt.title("Top 20 Districts by Predicted High Future Biometric Load (2026 Peak)", fontsize=14, fontweight='bold')
plt.xlabel("District", fontsize=12)
plt.ylabel("Future Daily Biometric Load", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Adding value labels
for i, v in enumerate(district_future_load):
    plt.text(i, v + (district_future_load.max() * 0.01), f"{v:,}", ha='left', fontsize=9, fontweight='bold', rotation=45)

plt.tight_layout()
plt.show()

# -----------------------------
# 7. 2026 Monthly Analysis (Reference ChatGPT Logic: Seasonal Average)
# -----------------------------
print("\nAnalyzing Monthly Trends for 2026 (using Seasonal Average)...")

import calendar
import numpy as np

# Prepare historical monthly data
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

# Aggregate by month, year, and spot
monthly_hist = df.groupby(['year', 'month', 'district', 'pincode'])['load'].sum().reset_index()

# Calculate Seasonal Average for 2026 (Mean across past years for each month/spot)
# This captures local seasonality for every pincode.
seasonal_preds_2026 = (
    monthly_hist.groupby(['month', 'district', 'pincode'])['load']
    .mean()
    .reset_index()
)
seasonal_preds_2026.columns = ['month', 'district', 'pincode', 'predicted_monthly_load']

# Identify the Top 20 Spots by TOTAL predicted 2026 load (Sum of their 12 monthly averages)
# This ensures the heatmap has 20 consistent, high-impact rows (Uniformity)
annual_top_20 = (
    seasonal_preds_2026.groupby(['district', 'pincode'])['predicted_monthly_load']
    .sum()
    .sort_values(ascending=False)
    .head(20)
    .reset_index()
)

# Label for heatmap rows
annual_top_20['spot_label'] = annual_top_20['district'] + " (" + annual_top_20['pincode'].astype(str) + ")"

# Filter the prediction data to only include these top 20 spots
heatmap_data = seasonal_preds_2026.merge(annual_top_20[['district', 'pincode', 'spot_label']], on=['district', 'pincode'])

# Pivot for Heatmap: Rows = Spot, Columns = Month
pivot_heatmap = heatmap_data.pivot(index='spot_label', columns='month', values='predicted_monthly_load')

# Ensure all months are present initially, then drop January, February, and August (1, 2, 8)
pivot_heatmap = pivot_heatmap.reindex(columns=range(1, 13), fill_value=0)
pivot_heatmap = pivot_heatmap.drop(columns=[1, 2, 8])

# Reorder rows to match annual ranking
pivot_heatmap = pivot_heatmap.reindex(annual_top_20['spot_label']).fillna(0)

# Heatmap Plot
plt.figure(figsize=(25, 15))

# Debug: Print the first few rows to console to verify data presence
print("\nTop 5 Pincode Spot Projections (Sample - Selected Months):")
print(pivot_heatmap.head())

try:
    import seaborn as sns
    print("Generating forced-annotation Heatmap (Excluding Jan, Feb, Aug)...")
    # Using annot=False in sns.heatmap and then manually adding text for 100% control
    ax = sns.heatmap(pivot_heatmap, annot=False, cmap="YlOrRd", 
                     cbar_kws={'label': 'Predicted Monthly Load (Transactions)'}, 
                     linewidths=0.5, linecolor='gray')
except ImportError:
    print("Seaborn not found, using Matplotlib.")
    ax = plt.gca()
    im = ax.imshow(pivot_heatmap, cmap="YlOrRd", aspect='auto')
    plt.colorbar(im, label='Predicted Monthly Load')

# MANUAL ANNOTATION: This ensures EVERY box has a number
for i in range(len(pivot_heatmap.index)):
    for j in range(len(pivot_heatmap.columns)):
        val = pivot_heatmap.iloc[i, j]
        # Only display if value is >= 0 (all should be >= 0 due to fillna)
        text_color = "white" if val > (pivot_heatmap.max().max() * 0.6) else "black"
        plt.text(j + 0.5, i + 0.5, f"{int(val):,}", 
                 ha="center", va="center", color=text_color, fontsize=9, fontweight='bold')

# Formatting
month_cols = pivot_heatmap.columns
month_names = [calendar.month_name[m] for m in month_cols]

plt.title("Predicted Top 20 High-Load Biometric Localities (Selected Months 2026)", fontsize=18, fontweight='bold', pad=25)
plt.xlabel("Month of 2026", fontsize=14)
plt.ylabel("District (Pincode)", fontsize=14)

# Set ticks at center of each cell
plt.xticks(np.arange(len(month_names)) + 0.5, month_names, rotation=45)
plt.yticks(np.arange(len(pivot_heatmap.index)) + 0.5, pivot_heatmap.index, rotation=0)

plt.tight_layout()
plt.show()

print("\nDone.")
