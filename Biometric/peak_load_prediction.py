
# ============================================================
# Peak Load Prediction (Prophet Model) - 180 Day Forecast
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from prophet import Prophet

# -----------------------------
# Configuration
# -----------------------------
PEAK_TO_DAILY_RATIO = 0.18    # 18% of daily traffic in peak hour
CONCURRENCY_WINDOW_SEC = 5    # Max concurrent users window
AVG_REQ_DURATION_SEC = 2      # Avg transaction time (implied)

# -----------------------------
# 1. Load and Aggregate Data
# -----------------------------
files = [
    'c:/Users/khatr/UIDAI_Hackathon/Biometric/api_data_aadhar_biometric_0_500000.csv',
    'c:/Users/khatr/UIDAI_Hackathon/Biometric/api_data_aadhar_biometric_500000_1000000.csv',
    'c:/Users/khatr/UIDAI_Hackathon/Biometric/api_data_aadhar_biometric_1000000_1500000.csv',
    'c:/Users/khatr/UIDAI_Hackathon/Biometric/api_data_aadhar_biometric_1500000_1861108.csv'
]

print("Aggregating data files...")
dfs = []
for f in files:
    try:
        temp_df = pd.read_csv(f)
        dfs.append(temp_df)
    except Exception as e:
        print(f"Error reading {f}: {e}")

if not dfs:
    print("No data loaded.")
    exit()

df = pd.concat(dfs, ignore_index=True)
df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y")
print(f"Total historical records: {len(df)}")

# -----------------------------
# 2. Prepare Data for Prophet
# -----------------------------
daily_data = df.groupby('date')[['bio_age_5_17', 'bio_age_17_']].sum().sort_index()
daily_data['total'] = daily_data['bio_age_5_17'] + daily_data['bio_age_17_']

# Prophet requires 'ds' (date) and 'y' (value) columns
df_prophet = daily_data.reset_index()[['date', 'total']].rename(columns={'date': 'ds', 'total': 'y'})

# -----------------------------
# 3. Train Prophet Model
# -----------------------------
print("Training Prophet model...")
m = Prophet(
    yearly_seasonality=True,   # Enabled as we likely have enough data to see year-over-year patterns or at least full year structure
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.5
)
m.fit(df_prophet)

# -----------------------------
# 4. Forecast (180 Days)
# -----------------------------
print("Generating 180-day forecast...")
future = m.make_future_dataframe(periods=180, freq='D')
forecast = m.predict(future)

# Extract Future Part
forecast_future = forecast[forecast['ds'] > df_prophet['ds'].max()][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()

# Fix negative predictions: Clamp to 0
forecast_future['yhat'] = forecast_future['yhat'].clip(lower=0)
forecast_future['yhat_lower'] = forecast_future['yhat_lower'].clip(lower=0)
forecast_future['yhat_upper'] = forecast_future['yhat_upper'].clip(lower=0)

forecast_future.rename(columns={'ds': 'date', 'yhat': 'pred_total'}, inplace=True)

# -----------------------------
# 5. Add Capacity Metrics
# -----------------------------
def add_capacity_metrics(forecast_df):
    df_metrics = forecast_df.copy()
    
    # 1) Predicted peak-hour volume
    df_metrics['pred_peak_hour'] = df_metrics['pred_total'] * PEAK_TO_DAILY_RATIO
    
    # 2) Peak-hour average RPS (requests per second)
    df_metrics['pred_peak_rps'] = df_metrics['pred_peak_hour'] / 3600.0
    
    # 3) Estimated max concurrent requests in window
    # Formula: RPS * Window
    df_metrics['est_max_concurrent'] = df_metrics['pred_peak_rps'] * CONCURRENCY_WINDOW_SEC
    
    # Rounding
    df_metrics['pred_total'] = df_metrics['pred_total'].round().astype(int)
    df_metrics['pred_peak_hour'] = df_metrics['pred_peak_hour'].round().astype(int)
    df_metrics['est_max_concurrent'] = df_metrics['est_max_concurrent'].round(1)
    
    return df_metrics

capacity_forecast = add_capacity_metrics(forecast_future)

# -----------------------------
# 6. Output & Visualization
# -----------------------------
output_file = "aadhar_capacity_forecast_next_180_days.csv"
capacity_forecast.to_csv(output_file, index=False)
print(f"\nForecast saved to: {output_file}")

# Display Top High Load Days
print("\n" + "="*80)
print(f"TOP PREDICTED PEAK LOAD DAYS (NEXT 180 DAYS)")
print("="*80)
print(f"{'Date':<15} | {'Daily Vol':<15} | {'Peak Hr Vol':<15} | {'Max Concurrent':<15}")
print("-" * 80)
top_days = capacity_forecast.sort_values('est_max_concurrent', ascending=False).head(10)
for _, row in top_days.iterrows():
    print(f"{row['date'].strftime('%Y-%m-%d'):<15} | {row['pred_total']:<15,d} | {row['pred_peak_hour']:<15,d} | {row['est_max_concurrent']:<15,.1f}")
print("-" * 80)


# Plotting
plt.figure(figsize=(14, 8))

# Historical Data
plt.plot(daily_data.index, daily_data['total'], label='Historical Actuals', color='black', alpha=0.6, linewidth=1)

# Prophet Forecast
plt.plot(capacity_forecast['date'], capacity_forecast['pred_total'], label='Forecast (180 Days)', color='#1f77b4', linewidth=2)
plt.fill_between(capacity_forecast['date'], 
                 forecast_future['yhat_lower'], 
                 forecast_future['yhat_upper'], 
                 color='#1f77b4', alpha=0.2, label='Confidence Interval')

# Highlight Peaks
plt.scatter(top_days['date'], top_days['pred_total'], color='red', s=50, zorder=5, label='Projected Critical Peaks')

# Annotate Top Peak
if not top_days.empty:
    top_peak = top_days.iloc[0]
    plt.annotate(
        f"Max Load\n{top_peak['date'].strftime('%Y-%m-%d')}\nConcur: {top_peak['est_max_concurrent']:.1f}",
        xy=(top_peak['date'], top_peak['pred_total']),
        xytext=(top_peak['date'], top_peak['pred_total'] * 1.15),
        ha='center',
        arrowprops=dict(arrowstyle="->", color='red'),
        fontsize=9, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8)
    )

plt.title("Biometric Traffic Forecast (Prophet Model) - 180 Days", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Daily Transactions")
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.ylim(0, max(daily_data['total'].max(), capacity_forecast['pred_total'].max()) * 1.4) 
plt.tight_layout()
plt.show()

# Secondary Plot for Capacity Planning
plt.figure(figsize=(14, 6))
plt.plot(capacity_forecast['date'], capacity_forecast['est_max_concurrent'], color='darkorange', linewidth=2, label='Est. Max Concurrent Users')
plt.title("Capacity Planning: Forecasted Concurrent Users (5s Window)", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Concurrent Users")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
