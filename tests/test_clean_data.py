import pandas as pd
import numpy as np
from src.clean_data import clean_obelix_data

def test_clean_obelix_data():
    # Mock data setup
    data = {
        'True Composition': [
            'Material A', 'Material A', 'Material A',  # Low variance
            'Material B', 'Material B',                # High variance
            'Material C'                               # Single measurement
        ],
        'Ionic conductivity (S cm-1)': [
            1e-4, 2e-4, 1.5e-4,  # log10: -4.0, ~ -3.69, ~ -3.82 -> Diff < 2.0
            1e-6, 1e-2,          # log10: -6.0, -2.0 -> Diff is 4.0
            1e-5                 # log10: -5.0
        ]
    }
    raw_df = pd.DataFrame(data)
    
    # Run cleaning function
    cleaned_df = clean_obelix_data(raw_df)
    
    # Assertions
    comps = cleaned_df['True Composition'].tolist()
    
    # 1. Material B should be completely dropped due to >2.0 diff
    assert 'Material B' not in comps
    
    # 2. Material A and C should remain
    assert 'Material A' in comps
    assert 'Material C' in comps
    
    # 3. Check correct columns
    expected_cols = ['True Composition', 'log10_conductivity', 'measurement_count']
    assert list(cleaned_df.columns) == expected_cols
    
    # 4. Check median aggregation for Material A
    mat_a_row = cleaned_df[cleaned_df['True Composition'] == 'Material A'].iloc[0]
    expected_median = np.log10(1.5e-4)
    assert np.isclose(mat_a_row['log10_conductivity'], expected_median)
    assert mat_a_row['measurement_count'] == 3
    
    # 5. Check Material C
    mat_c_row = cleaned_df[cleaned_df['True Composition'] == 'Material C'].iloc[0]
    assert np.isclose(mat_c_row['log10_conductivity'], -5.0)
    assert mat_c_row['measurement_count'] == 1
