import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv(
    r'D:\uidia hackathon\enrollment\api_data_aadhar_enrolment_0_500000.csv'
)

# Aggregate total enrollments per pincode (sum across dates/age groups for simplicity)
df['total_enroll'] = df['age_0_5'] + df['age_5_17'] + df['age_18_greater']
pincode_totals = df.groupby(['state', 'pincode'])['total_enroll'].sum().reset_index()

# Plot boxplot by state
plt.figure(figsize=(14, 8))
sns.boxplot(data=pincode_totals, x='state', y='total_enroll', palette='Set2')
plt.title('Intra-State Variability: Total Enrollments per Pincode')
plt.xlabel('State')
plt.ylabel('Total Enrollments per Pincode')
plt.xticks(rotation=90)
plt.ylim(0, pincode_totals['total_enroll'].quantile(0.95))  # Trim outliers for visibility
plt.grid(True)
plt.tight_layout()
plt.savefig('pincode_variability.png')
plt.show()