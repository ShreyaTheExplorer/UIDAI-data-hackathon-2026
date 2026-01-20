import pandas as pd

import pandas as pd
import os
import re

files = [
    'c:/Users/khatr/UIDAI_Hackathon/Biometric/api_data_aadhar_biometric_0_500000.csv',
    'c:/Users/khatr/UIDAI_Hackathon/Biometric/api_data_aadhar_biometric_500000_1000000.csv',
    'c:/Users/khatr/UIDAI_Hackathon/Biometric/api_data_aadhar_biometric_1000000_1500000.csv',
    'c:/Users/khatr/UIDAI_Hackathon/Biometric/api_data_aadhar_biometric_1500000_1861108.csv'
]

# Mapping dictionary for States
state_mapping = {
    'Orissa': 'Odisha',
    'Westbangal': 'West Bengal',
    'West Bangal': 'West Bengal',
    'Westbengal': 'West Bengal',
    'West  Bengal': 'West Bengal',
    'Pondicherry': 'Puducherry',
    'Uttaranchal': 'Uttarakhand',
    'Andaman & Nicobar Islands': 'Andaman and Nicobar Islands',
    'Andaman And Nicobar Islands': 'Andaman and Nicobar Islands',
    'Jammu & Kashmir': 'Jammu and Kashmir',
    'Jammu And Kashmir': 'Jammu and Kashmir',
    'Daman & Diu': 'Daman and Diu',
    'Daman And Diu': 'Daman and Diu',
    'Dadra & Nagar Haveli And Daman & Diu': 'Dadra and Nagar Haveli and Daman and Diu',
    'Dadra And Nagar Haveli And Daman And Diu': 'Dadra and Nagar Haveli and Daman and Diu',
    'Dadra And Nagar Haveli': 'Dadra and Nagar Haveli',
    'Chhatisgarh': 'Chhattisgarh',
    'Nct Of Delhi': 'Delhi'
}

# Manual district corrections (State, District_Lower) -> Correct_District_Lower
district_manual_map = {
    # Odisha
    ('Odisha', 'balesore'): 'balasore',
    ('Odisha', 'jajpur road'): 'jajpur',
    ('Odisha', 'kendrapara'): 'kendrapada',
    # West Bengal
    ('West Bengal', 'north 24 parganas'): 'north twenty four parganas',
    ('West Bengal', 'south 24 parganas'): 'south twenty four parganas',
    ('West Bengal', 'west medinipur'): 'paschim medinipur',
    ('West Bengal', 'east medinipur'): 'purba medinipur',
    # Chhattisgarh
    ('Chhattisgarh', 'kanker'): 'uttar bastar kanker',
    # Tamil Nadu
    ('Tamil Nadu', 'thiruvallur'): 'tiruvallur',
    ('Tamil Nadu', 'thiruvannamalai'): 'tiruvannamalai',
    # Maharashtra official renames
    ('Maharashtra', 'aurangabad'): 'chhatrapati sambhajinagar',
    ('Maharashtra', 'chhatrapti sambhajinagar'): 'chhatrapati sambhajinagar',
    ('Maharashtra', 'osmanabad'): 'dharashiv',
    ('Maharashtra', 'dharashiv'): 'dharashiv',
    ('Maharashtra', 'ahmednagar'): 'ahilyanagar',
    ('Maharashtra', 'ahmadnagar'): 'ahilyanagar',
    ('Maharashtra', 'ahilyanagar'): 'ahilyanagar',
    ('Maharashtra', 'raigad'): 'raigarh',
}

def normalize_state(state):
    if pd.isna(state):
        return state
    
    # Strip and Title Case
    state = str(state).strip().title()
    
    # Specific Mapping
    return state_mapping.get(state, state)

def normalize_district_text(text):
    if pd.isna(text):
        return text
    text = str(text).lower().strip()
    text = text.replace('&', 'and')
    text = re.sub(r'\s+', ' ', text)
    return text

def get_clean_district(row):
    state = row['state'] # This is already normalized state
    if pd.isna(row['district']):
        return row['district']
    
    dist_norm = normalize_district_text(row['district'])
    
    # Check specific map
    key = (state, dist_norm)
    if key in district_manual_map:
        return district_manual_map[key].title()
    
    return dist_norm.title()

def process_file(file_path):
    print(f"\n{'='*50}")
    print(f"Processing {os.path.basename(file_path)}...")
    
    try:
        df = pd.read_csv(file_path)
        
        # 1. Normalize States
        if 'state' in df.columns:
            print("Normalizing 'state' column...")
            df['state'] = df['state'].apply(normalize_state)
        else:
            print("'state' column not found, skipping state norm.")

        # 2. Normalize Districts
        # Find district column (case insensitive)
        district_col = None
        for col in df.columns:
            if 'district' in col.lower():
                district_col = col
                break
        
        if district_col and 'state' in df.columns:
            print(f"Normalizing '{district_col}' column...")
            # We need to pass the row to access both state and district
            # Renaming the district column temporarily to standard 'district' to reuse function easily or just use variable
            
            # Helper wrapper
            def clean_row_district(row):
                return get_clean_district({'state': row['state'], 'district': row[district_col]})

            df[district_col] = df.apply(clean_row_district, axis=1)
            print("Districts normalized.")
            
        else:
             print("Skipping district normalization (missing 'state' or 'district' column).")

        # Save back to CSV
        print("Saving changes...")
        df.to_csv(file_path, index=False)
        print("Saved.")
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()

# Main execution loop
for f in files:
    process_file(f)

print(f"\n{'='*50}")
print("All files processed.")

