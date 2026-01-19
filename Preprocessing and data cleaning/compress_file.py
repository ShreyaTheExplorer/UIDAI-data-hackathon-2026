import pandas as pd
import os

# 1. Setup File Paths
input_csv = r'c:\Users\VICTUS\OneDrive\Desktop\Aadhar dada\api_data_aadhar_enrolment\api_data_aadhar_enrolment_1000000_1006029.csv'
output_parquet = 'UIDAI/3optimized_file.parquet'
def compress_file(input_csv,output_parquet):
    print("Loading CSV...")
    
    # 2. Load CSV (Default types first)
    # We use low_memory=False to ensure pandas guesses types accurately before we convert
    df = pd.read_csv(input_csv, low_memory=False)
    
    
    
    # 3. Convert Types
    # A. Convert Date
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    
    # B. Convert Numeric Columns to int32
    # This cuts memory for these specific columns by 50%
    cols_to_int32 = ['age_0_5', 'age_5_17', 'age_18_greater']
    for col in cols_to_int32:
        df[col] = df[col].astype('int32')
    
    # C. (Implicit) All other columns like state/district stay as 'object' (string) by default
    
    # 4. Save to Parquet
    print(f"Saving to {output_parquet}...")
    df.to_parquet(output_parquet, index=False)
    
