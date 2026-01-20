import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv(
    r'D:\uidia hackathon\enrollment\api_data_aadhar_enrolment_0_500000.csv'
)

# Convert date column (DD-MM-YYYY)
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')

# Aggregate daily totals by age group
daily_totals = (
    df.groupby('date')[['age_0_5', 'age_5_17', 'age_18_greater']]
    .sum()
    .reset_index()
)

# Convert to long format for seaborn
daily_long = daily_totals.melt(
    id_vars='date',
    var_name='Age Group',
    value_name='Enrollments'
)

# Plot
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=daily_long,
    x='date',
    y='Enrollments',
    hue='Age Group',
    marker='o'
)

plt.title('Daily Aadhaar Enrollment Trends by Age Group')
plt.xlabel('Date')
plt.ylabel('Total Enrollments')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Save and show
plt.savefig('daily_trends.png', dpi=300)
plt.show()
