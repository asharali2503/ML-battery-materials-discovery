import pandas as pd
import os

def test_feature_alignment():
    train_path = os.path.join("data", "processed", "featurized_dataset.csv")
    candidate_path = os.path.join("data", "processed", "mp_candidates_featurized.csv")
    
    # Load data
    df_train = pd.read_csv(train_path)
    df_cand = pd.read_csv(candidate_path)
    
    # Identify training features by dropping target/metadata
    cols_to_drop_train = ["log10_conductivity", "measurement_count"]
    if "True Composition" in df_train.columns:
        cols_to_drop_train.append("True Composition")
        
    X_train = df_train.drop(columns=cols_to_drop_train)
    
    # Identify candidate features by dropping MP metadata
    meta_cols = ["material_id", "formula_pretty", "energy_above_hull", "formation_energy_per_atom"]
    X_candidate = df_cand.drop(columns=meta_cols)
    
    # Assert feature sets have the exact same number of columns
    assert len(X_train.columns) == len(X_candidate.columns), \
        f"Feature mismatch: Train has {len(X_train.columns)}, Candidates have {len(X_candidate.columns)}"
        
    # Assert column names and order are perfectly identical
    assert list(X_train.columns) == list(X_candidate.columns), \
        "Feature columns or order do not strictly match between training and inference datasets!"
