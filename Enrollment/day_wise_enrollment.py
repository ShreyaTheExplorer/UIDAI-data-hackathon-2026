import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv(
    r'D:\uidia hackathon\enrollment\api_data_aadhar_enrolment_0_500000.csv'
)

# Convert date to datetime and extract day of week
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
df['day_of_week'] = df['date'].dt.day_name()

# Aggregate average enrollments by day of week and age group
avg_by_day = df.groupby('day_of_week')[['age_0_5', 'age_5_17', 'age_18_greater']].mean().reset_index()
avg_by_day = avg_by_day.melt(id_vars='day_of_week', var_name='Age Group', value_name='Average Enrollments')

# Order days
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
avg_by_day['day_of_week'] = pd.Categorical(avg_by_day['day_of_week'], categories=days_order, ordered=True)

# Plot grouped bar chart
plt.figure(figsize=(12, 6))
sns.barplot(data=avg_by_day, x='day_of_week', y='Average Enrollments', hue='Age Group')
plt.title('Average Aadhaar Enrollments by Day of Week and Age Group')
plt.xlabel('Day of Week')
plt.ylabel('Average Enrollments')
plt.legend(title='Age Group')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('day_of_week_comparison.png')
plt.show()