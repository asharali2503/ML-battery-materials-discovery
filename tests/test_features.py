import pandas as pd
import numpy as np
from src.build_features import generate_features

def test_generate_features():
    # Mock data setup
    data = {
        'True Composition': ["Li2O", "NaCl", "FakeElement99"],
        'log10_conductivity': [-3.0, -5.0, -1.0],
        'measurement_count': [1, 2, 1]
    }
    df = pd.DataFrame(data)
    
    # Run featurization
    feat_df = generate_features(df, 'True Composition')
    
    # Assertions
    # 1. "FakeElement99" should be dropped
    assert len(feat_df) == 2
    
    # 2. Check that string columns are dropped
    assert 'True Composition' not in feat_df.columns
    assert 'composition' not in feat_df.columns
    
    # 3. Check for Magpie features
    assert any('Electronegativity' in col for col in feat_df.columns)
    
    # 4. Check all columns are numeric
    for col in feat_df.columns:
        assert pd.api.types.is_numeric_dtype(feat_df[col])
