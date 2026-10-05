import pandas as pd
import numpy as np
import logging
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from pymatgen.core import Composition
from src.models.metrics import calculate_metrics

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_primary_anion(formula: str) -> str:
    """
    Parses a chemical formula and identifies the primary anion 
    by finding the element with the highest Pauling electronegativity.
    """
    try:
        comp = Composition(formula)
        elements = comp.elements
        max_x = -1
        primary = None
        for el in elements:
            try:
                # Some elements might lack electronegativity data in pymatgen
                x_val = float(el.X)
            except (ValueError, TypeError):
                x_val = 0
                
            if x_val > max_x:
                max_x = x_val
                primary = el.symbol
        return primary
    except Exception as e:
        logging.warning(f"Failed to parse {formula}: {e}")
        return "Unknown"

if __name__ == "__main__":
    cleaned_path = os.path.join("data", "interim", "cleaned_dataset.csv")
    feat_path = os.path.join("data", "processed", "featurized_dataset.csv")
    
    logging.info("Loading datasets...")
    df_clean = pd.read_csv(cleaned_path)
    df_feat = pd.read_csv(feat_path)
    
    # 1. Generate Groups
    logging.info("Generating anion groups to evaluate true domain extrapolation...")
    df_clean['primary_anion'] = df_clean['True Composition'].apply(get_primary_anion)
    groups = df_clean['primary_anion'].values
    
    group_counts = df_clean['primary_anion'].value_counts()
    logging.info(f"Unique anion families breakdown:\n{group_counts.to_string()}")
    
    # 2. Setup Data
    y = df_feat['log10_conductivity']
    X = df_feat.drop(columns=['log10_conductivity', 'measurement_count'])
    
    # 3. Setup Model
    rf_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    gkf = GroupKFold(n_splits=5)
    
    # 4. Evaluation Loop
    logging.info("Starting GroupKFold (Out-of-Domain) evaluation...")
    metrics_list = []
    
    # Ensure there are enough groups
    num_groups = len(np.unique(groups))
    if num_groups < 5:
        logging.warning(f"Only {num_groups} unique anion groups found. GroupKFold will reduce n_splits.")
        
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        # Extract train and test sets strictly isolating the anion families
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Fit on training domain
        rf_pipe.fit(X_train, y_train)
        
        # Predict on hold-out domain
        y_pred = rf_pipe.predict(X_test)
        
        fold_metrics = calculate_metrics(y_test, y_pred)
        metrics_list.append(fold_metrics)
        logging.info(f"Fold {fold+1} metrics (Hold-out Size: {len(y_test)}): {fold_metrics}")
        
    avg_metrics = {
        'MAE': round(np.mean([m['MAE'] for m in metrics_list]), 4),
        'RMSE': round(np.mean([m['RMSE'] for m in metrics_list]), 4),
        'R2': round(np.mean([m['R2'] for m in metrics_list]), 4)
    }
    
    logging.info(f"Final Grouped Domain Extrapolation Metrics: {avg_metrics}")
