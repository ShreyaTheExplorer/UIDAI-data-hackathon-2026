import pandas as pd
import os

files = [
    'c:/Users/khatr/UIDAI_Hackathon/Biometric/api_data_aadhar_biometric_0_500000.csv',
    'c:/Users/khatr/UIDAI_Hackathon/Biometric/api_data_aadhar_biometric_500000_1000000.csv',
    'c:/Users/khatr/UIDAI_Hackathon/Biometric/api_data_aadhar_biometric_1000000_1500000.csv',
    'c:/Users/khatr/UIDAI_Hackathon/Biometric/api_data_aadhar_biometric_1500000_1861108.csv'
]

dfs = []
for f in files:
    try:
        temp_df = pd.read_csv(f)
        dfs.append(temp_df)
    except Exception as e:
        print(f"Error reading {f}: {e}")

df = pd.concat(dfs, ignore_index=True)
df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y")
df['year'] = df['date'].dt.year
df['total'] = df['bio_age_5_17'] + df['bio_age_17_']

print("Yearly Totals:")
print(df.groupby('year')['total'].sum())

print("\nLast 20 records:")
print(df.sort_values('date').tail(20)[['date', 'total']])
