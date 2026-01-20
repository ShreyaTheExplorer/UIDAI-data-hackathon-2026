import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv(
    r'D:\uidia hackathon\enrollment\api_data_aadhar_enrolment_0_500000.csv'
)

# Convert date
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')

# ---- FILTER: Maharashtra + Pune district ----
pune_df = df[
    (df['state'].str.lower() == 'maharashtra') &
    (df['district'].str.lower() == 'pune')
]

# Aggregate future burden (Age 0–5) by pincode
future_burden = (
    pune_df.groupby('pincode')['age_0_5']
    .sum()
    .reset_index()
    .sort_values('age_0_5', ascending=False)
)

# Take top 30 critical pincodes (better readability)
top_pune_burden = future_burden.head(30)

# Prepare heatmap data
heatmap_data = top_pune_burden.set_index('pincode')

# Plot heatmap
plt.figure(figsize=(8, 10))
sns.heatmap(
    heatmap_data,
    cmap='Reds',
    linewidths=0.5,
    annot=True,
    fmt='d'
)

plt.title(
    'Pune District: Heatmap of Future Education & Childcare Demand\n(Age 0–5 Population)',
    fontsize=12
)
plt.xlabel('Future Burden Index')
plt.ylabel('Pincode')
plt.tight_layout()
plt.show()
