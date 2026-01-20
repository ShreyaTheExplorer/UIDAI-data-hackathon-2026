
# ============================================================
# Trend Analysis - Aggregated Data
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry.base import BaseGeometry
import geopandas as gpd
import os

# -----------------------------
# 1. Load and Aggregate Data
# -----------------------------
files = [
    'c:/Users/khatr/UIDAI_Hackathon/Biometric/api_data_aadhar_biometric_0_500000.csv',
    'c:/Users/khatr/UIDAI_Hackathon/Biometric/api_data_aadhar_biometric_500000_1000000.csv',
    'c:/Users/khatr/UIDAI_Hackathon/Biometric/api_data_aadhar_biometric_1000000_1500000.csv',
    'c:/Users/khatr/UIDAI_Hackathon/Biometric/api_data_aadhar_biometric_1500000_1861108.csv'
]

dfs = []
for f in files:
    print(f"Reading {os.path.basename(f)}...")
    try:
        temp_df = pd.read_csv(f)
        dfs.append(temp_df)
    except Exception as e:
        print(f"Error reading {f}: {e}")

if not dfs:
    print("No data loaded.")
    exit()

df = pd.concat(dfs, ignore_index=True)
print(f"Total records aggregated: {len(df)}")

df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y")

# -----------------------------
# 2. Data Cleaning / Preparation
# (Re-applying critical logic from adhar_bio.py for plotting consistency)
# -----------------------------

# Ensure state names are title case (they should be normalized already, but safety first)
df['state_clean'] = df['state'].astype(str).str.strip().str.title()

# District normalization (already done in source files mostly, but need column unified)
# We need to find the specific district column if it varies, but normalize_states.py 
# updated the column in place. We assume 'District' or 'district'.
district_col = None
for col in df.columns:
    if 'district' in col.lower():
        district_col = col
        break

if district_col:
    df['district_clean'] = df[district_col].astype(str).str.strip().str.title()
else:
    df['district_clean'] = 'Unknown'

# Resolve multi-state districts (Logic from adhar_bio.py)
multi_state_district_map = {
    'Balrampur': 'Uttar Pradesh',
    'Bijapur': 'Karnataka',
    'Bilaspur': 'Chhattisgarh',
    'Hamirpur': 'Uttar Pradesh',
    'Karimnagar': 'Telangana',
    'Khammam': 'Telangana',
    'Leh': 'Ladakh',
    'Mahabubnagar': 'Telangana',
    'Medak': 'Telangana',
    'Nalgonda': 'Telangana',
    'Adilabad': 'Telangana',
    'Hyderabad': 'Telangana',
    'K.V. Rangareddy': 'Telangana',
    'Cuddalore': 'Tamil Nadu',
    'Kargil': 'Ladakh',
    'Nizamabad': 'Telangana',
    'Pratapgarh': 'Rajasthan',
    'Raigarh': 'Chhattisgarh',
    'Rupnagar': 'Punjab',
    'Viluppuram': 'Tamil Nadu',
    'Warangal': 'Telangana'
}

for dist, state in multi_state_district_map.items():
    df.loc[df['district_clean'] == dist, 'state_clean'] = state

# -----------------------------
# 2.5. Analysis: 18+ Biometric Contribution
# -----------------------------
print("\n--- 18+ Group Contribution to Biometric Updates (Month-Year) ---")
df['month_year'] = df['date'].dt.to_period('M')
monthly_data = df.groupby('month_year')[['bio_age_17_', 'bio_age_5_17']].sum()
monthly_data['total_updates'] = monthly_data['bio_age_17_'] + monthly_data['bio_age_5_17']
monthly_data['18+_contribution_percent'] = (monthly_data['bio_age_17_'] / monthly_data['total_updates']) * 100

# Display the table with nice formatting
print(monthly_data[['18+_contribution_percent']].rename(columns={'18+_contribution_percent': '18+ Contribution (%)'}).round(2))
print("---------------------------------------------------------------")

# -----------------------------
# 3. Visualization 1: Time-wise Line Graph
# -----------------------------
print("\nGenerating Line Graph...")
daily_data = df.groupby('date')[['bio_age_5_17', 'bio_age_17_']].sum().sort_index()
daily_data['total'] = daily_data['bio_age_5_17'] + daily_data['bio_age_17_']

spike_date = daily_data['total'].idxmax()
spike_value = daily_data.loc[spike_date, 'total']
drop_date = daily_data['total'].idxmin()
drop_value = daily_data.loc[drop_date, 'total']

plt.figure(figsize=(12,6))
plt.plot(daily_data.index, daily_data['bio_age_17_'], label='Age 18+', linewidth=2)
plt.plot(daily_data.index, daily_data['bio_age_5_17'], label='Age 5–17', linewidth=2)
plt.plot(daily_data.index, daily_data['total'], linestyle='--', linewidth=2, label='Total Usage')

plt.annotate("Spike observed", xy=(spike_date, spike_value),
             xytext=(spike_date, spike_value*1.05), arrowprops=dict(arrowstyle="->"), fontsize=9, fontweight='bold')
plt.annotate("Sudden drop", xy=(drop_date, drop_value),
             xytext=(drop_date + pd.Timedelta(days=2), drop_value*1.5 if drop_value > 0 else 10), # Adjusted y-offset
             arrowprops=dict(arrowstyle="->"), fontsize=9, fontweight='bold')

plt.title("Time-wise Aadhaar Biometric Authentication Trends (Aggregated)")
plt.xlabel("Date")
plt.ylabel("Number of Biometric Authentications")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# -----------------------------
# 4. Visualization 2: Choropleth Map
# -----------------------------
print("\nGenerating Map...")
statewise_bio = (
    df.groupby('state_clean')[['bio_age_5_17', 'bio_age_17_']]
      .sum()
      .reset_index()
)
statewise_bio['total_biometric'] = statewise_bio['bio_age_5_17'] + statewise_bio['bio_age_17_']

# Load Shapefile (User specified locally in Biometric/Shapefiles)
shapefile_path = "Shapefiles/india_states.shp"

try:
    india_map = gpd.read_file(shapefile_path)
    india_map = india_map.to_crs(epsg=4326)
    india_map = india_map.rename(columns={'ST_NM': 'state'})
    
    # Normalize shapefile state names
    india_map['state'] = (
        india_map['state']
        .astype(str)
        .str.upper()
        .str.strip()
        .replace({
            'JAMMU & KASHMIR': 'JAMMU AND KASHMIR',
            'JAMMU AND KASMIR': 'JAMMU AND KASHMIR'
        })
    )
    
    # Prepare data for merge
    state_data = statewise_bio.copy()
    state_data['state'] = (
        state_data['state_clean']
        .astype(str)
        .str.upper()
        .str.strip()
        .replace({
            'JAMMU & KASHMIR': 'JAMMU AND KASHMIR',
            'JAMMU AND KASMIR': 'JAMMU AND KASHMIR'
        })
    )
    
    merged = india_map.merge(state_data, on='state', how='left')
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 14))
    
    merged.plot(
        column='total_biometric',
        cmap='OrRd',
        linewidth=0.8,
        ax=ax,
        edgecolor='0.7',
        legend=True,
        legend_kwds={
            'label': "Total Aadhaar Biometric Transactions",
            'orientation': "vertical"
        }
    )
    
    # Set bounds
    xmin, ymin, xmax, ymax = merged.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(
        "State-wise Aadhaar Biometric Transactions (Aggregated)",
        fontsize=14,
        pad=15
    )
    
    # Label offsets
    label_offsets = {
        'GOA': (0.12, 0),
        'ASSAM': (1.75, 0.2),
        'CHANDIGARH': (0.15, -0.16),
        'PUNJAB': (-0.2, 0),
        'MEGHALAYA': (0, -0.15),
        'WEST BENGAL': (0, -1.2),
        'MADHYA PRADESH':(0,-0.2)
    }
    
    for _, row in merged.iterrows():
        if not isinstance(row.geometry, BaseGeometry) or row.geometry.is_empty:
            continue
            
        if pd.notna(row['total_biometric']) and row['state'] != 'PUDUCHERRY':
            point = row.geometry.representative_point()
            x, y = point.x, point.y
            
            if row['state'] in label_offsets:
                dx, dy = label_offsets[row['state']]
                x += dx
                y += dy
            
            # State name
            ax.text(x, y, row['state'], fontsize=7, color='darkblue', ha='center', va='bottom', fontweight='bold', clip_on=True)
            # Count
            ax.text(x, y, str(int(row['total_biometric'])), fontsize=6.5, color='black', ha='center', va='top', fontweight='bold', clip_on=True)
            
    # Puducherry special
    pudu = merged[merged['state'] == 'PUDUCHERRY']
    if not pudu.empty:
        label_ax_x, label_ax_y = 0.84, 0.24
        
        ax.text(
            label_ax_x, label_ax_y,
            f"PUDUCHERRY\n{int(pudu.iloc[0]['total_biometric'])}",
            fontsize=7, color='darkblue', ha='left', va='center', fontweight='bold',
            transform=ax.transAxes, zorder=5
        )
        
        for geom in pudu.geometry:
            if isinstance(geom, BaseGeometry) and not geom.is_empty:
                point = geom.representative_point()
                ax.plot(
                    [point.x, label_ax_x * (merged.total_bounds[2] - merged.total_bounds[0]) + merged.total_bounds[0]],
                    [point.y, label_ax_y * (merged.total_bounds[3] - merged.total_bounds[1]) + merged.total_bounds[1]],
                    linestyle='--', linewidth=0.9, color='black', zorder=4
                )
    
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Error generating map: {e}") 
    print("Ensure 'Shapefiles/india_states.shp' exists in the Biometric folder.")


# -----------------------------
# 5. Visualization 3: Top 20 Districts Bar Graph
# -----------------------------
print("\nGenerating District Bar Graph...")
districtwise_bio = (
    df.groupby('district_clean')[['bio_age_5_17', 'bio_age_17_']]
      .sum()
      .reset_index()
)
districtwise_bio['total_biometric'] = districtwise_bio['bio_age_5_17'] + districtwise_bio['bio_age_17_']
districtwise_bio = districtwise_bio.sort_values('total_biometric', ascending=False).reset_index(drop=True)

top_n = 20
top20 = districtwise_bio.head(top_n).copy()

fig, ax = plt.subplots(figsize=(12,8))
bars = ax.bar(range(top_n), top20['total_biometric'], color='teal', label='District Total')

ax.set_xticks(range(top_n))
ax.set_xticklabels(top20['district_clean'], rotation=45, ha='right', fontsize=10)

ax.set_xlabel("Top 20 Districts", fontsize=12)
ax.set_ylabel("Total Biometric Transactions", fontsize=12)
ax.set_title("Top 20 Districts by Aadhaar Biometric Usage (Aggregated)", fontsize=14)

for i, v in enumerate(top20['total_biometric']):
    ax.text(i+0.5, v + max(top20['total_biometric'])*0.01, str(v),
            ha='right', va='bottom', rotation=45, fontsize=9, color='black',fontweight='bold')

ax.legend()
plt.tight_layout()
plt.show()

print("\nDone.")
