import pandas as pd
def clean_maharashtra_districts(input_filename, output_filename):
    """
    Loads a parquet file (Maharashtra data only), cleans district names 
    to match the official 36 districts, aggregates duplicates, and saves to new parquet.
    """
    print(f"Loading {input_filename}...")
    df = pd.read_parquet(input_filename)

    # 1. Define the Maharashtra-Specific Map
        # 2. Define the Expanded Maharashtra-Specific Map
      # 2. Define the Expanded Maharashtra-Specific Map
    maha_district_map = {
        # --- MAJOR RENAMES ---
        'ahmednagar': 'Ahilyanagar',
        'ahmed nagar': 'Ahilyanagar',
        'ahmadnagar': 'Ahilyanagar',
        'ahilyanagar': 'Ahilyanagar',
        
        'aurangabad': 'Chhatrapati Sambhajinagar',
        'chhatrapati sambhajinagar': 'Chhatrapati Sambhajinagar',
        'chatrapati sambhaji nagar': 'Chhatrapati Sambhajinagar', # Fix variation
        
        'osmanabad': 'Dharashiv',
        'dharashiv': 'Dharashiv',
        
        # --- SPELLING FIXES & TYPOS ---
        'bid': 'Beed',
        'beed': 'Beed',
        
        'buldana': 'Buldhana',
        'buldhana': 'Buldhana',
        
        'gondiya': 'Gondia',
        'gondia': 'Gondia',
        'gondiya *': 'Gondia', # Fix asterisk
        
        'hingoli': 'Hingoli',
        'hingoli *': 'Hingoli', # Fix asterisk
        
        'washim': 'Washim',
        'washim *': 'Washim',   # Fix asterisk
        
        'nandurbar': 'Nandurbar',
        'nandurbar *': 'Nandurbar', # Fix asterisk
        
        'raigarh': 'Raigarh',
        'raigarh(mh)': 'Raigarh',
        'raigad': 'Raigarh',
        
        # --- MUMBAI NORMALIZATION ---
       
        'mumbai( sub urban )': 'Mumbai Suburban',
        'mumbai suburban': 'Mumbai Suburban',
        
        # --- STANDARD VARIATIONS ---
        'nashik': 'Nashik',
        'nasik': 'Nashik',
        
        # Self-mapping standard names just to be safe (optional but good)
        'akola': 'Akola', 'amravati': 'Amravati', 'bhandara': 'Bhandara',
        'chandrapur': 'Chandrapur', 'dhule': 'Dhule', 'gadchiroli': 'Gadchiroli',
        'jalgaon': 'Jalgaon', 'jalna': 'Jalna', 'kolhapur': 'Kolhapur',
        'latur': 'Latur', 'nagpur': 'Nagpur', 'nanded': 'Nanded',
        'palghar': 'Palghar', 'parbhani': 'Parbhani', 'pune': 'Pune',
        'ratnagiri': 'Ratnagiri', 'sangli': 'Sangli', 'satara': 'Satara',
        'sindhudurg': 'Sindhudurg', 'solapur': 'Solapur', 'thane': 'Thane',
        'wardha': 'Wardha', 'yavatmal': 'Yavatmal'
    }

    # 2. Clean District Names
    print("Cleaning district names...")
    
    # Convert to lowercase and strip whitespace for matching
    df['district'] = df['district'].astype(str).str.strip().str.lower()
    
    # Map the values. If a name isn't in the map, fallback to Title Case.
    df['district'] = df['district'].map(maha_district_map).fillna(df['district'].str.title())

    # 3. Aggregate Duplicates (Critical)
    # Because 'Bid' and 'Beed' are now both 'Beed', we must sum their rows
    print("Merging duplicate district rows...")
    
    # Identify numeric columns to sum
    numeric_cols = ['age_0_5', 'age_5_17', 'age_18_greater']
    
    # Group by all ID columns to collapse duplicates
    # We include 'date', 'state', 'district', 'pincode' to keep granularity
    groupby_cols = ['date', 'state', 'district', 'pincode']
    
    df_clean = df.groupby(groupby_cols, as_index=False, observed=True)[numeric_cols].sum()

    # 4. Save
    print(f"Saving cleaned data to {output_filename}...")
    df_clean.to_parquet(output_filename, index=False)
    
    # Validation Print
    print("\n--- Validation ---")
    print(f"Unique Districts Count: {len(df_clean['district'].unique())}")
    print("Districts List:", sorted(df_clean['district'].unique()))
