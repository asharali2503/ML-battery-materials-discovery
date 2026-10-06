import pandas as pd
import numpy as np
from src.clean_data import clean_obelix_data

def test_clean_obelix_data():
    # Mock data setup
    data = {
        'True Composition': [
            'Material A', 'Material A', 'Material A',  # Low variance
            'Material B', 'Material B',                # High variance (>1.0 diff)
            'Material C',                              # Single measurement
            'Material D'                               # Should be dropped by Temp
        ],
        'Ionic conductivity (S cm-1)': [
            1e-4, 1e-4, 3.16e-4, # log10: -4.0, -4.0, ~ -3.5 -> Diff is 0.5 (< 1.0)
            1e-6, 1e-4,          # log10: -6.0, -4.0 -> Diff is 2.0 (> 1.0)
            1e-5,                # log10: -5.0
            1e-3                 # log10: -3.0
        ],
        'Temperature': [
            298, 298, 298,       # RT
            298, 298,            # RT
            298,                 # RT
            400                  # High Temp
        ]
    }
    raw_df = pd.DataFrame(data)
    
    # Run cleaning function
    cleaned_df = clean_obelix_data(raw_df)
    
    # Assertions
    comps = cleaned_df['True Composition'].tolist()
    
    # 1. Material B should be completely dropped due to >1.0 diff
    assert 'Material B' not in comps
    
    # 2. Material D should be dropped due to Temperature
    assert 'Material D' not in comps
    
    # 3. Material A and C should remain
    assert 'Material A' in comps
    assert 'Material C' in comps
    
    # 4. Check correct columns
    expected_cols = ['True Composition', 'log10_conductivity', 'measurement_count']
    assert list(cleaned_df.columns) == expected_cols
    
    # 5. Check median aggregation for Material A
    mat_a_row = cleaned_df[cleaned_df['True Composition'] == 'Material A'].iloc[0]
    expected_median = -4.0 # median of [-4.0, -4.0, -3.5]
    assert np.isclose(mat_a_row['log10_conductivity'], expected_median)
    assert mat_a_row['measurement_count'] == 3
    
    # 6. Check Material C
    mat_c_row = cleaned_df[cleaned_df['True Composition'] == 'Material C'].iloc[0]
    assert np.isclose(mat_c_row['log10_conductivity'], -5.0)
    assert mat_c_row['measurement_count'] == 1
