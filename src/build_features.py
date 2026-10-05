import pandas as pd
import logging
import os
import sys
import subprocess

# Ensure matminer is installed
try:
    import matminer
except ImportError:
    print("Installing matminer...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matminer"])

from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def safe_parse_composition(formula: str):
    try:
        return Composition(formula)
    except Exception:
        return None

def generate_features(df: pd.DataFrame, formula_col: str) -> pd.DataFrame:
    df = df.copy()
    initial_len = len(df)
    
    # 1. Parse Composition
    df['composition'] = df[formula_col].apply(safe_parse_composition)
    
    # Drop parsing failures
    parsed_df = df.dropna(subset=['composition']).copy()
    dropped_parse = initial_len - len(parsed_df)
    if dropped_parse > 0:
        logging.info(f"Dropped {dropped_parse} rows due to pymatgen parsing failures.")
        
    # 2. Featurize
    featurizer = ElementProperty.from_preset("magpie")
    logging.info("Generating Magpie features...")
    # ignore_errors=True to prevent crashes on missing element properties
    feat_df = featurizer.featurize_dataframe(parsed_df, col_id='composition', ignore_errors=True)
    
    # Drop rows that failed featurization (NaNs in feature columns)
    before_feat_drop = len(feat_df)
    feat_cols = featurizer.feature_labels()
    feat_df = feat_df.dropna(subset=feat_cols)
    dropped_feat = before_feat_drop - len(feat_df)
    if dropped_feat > 0:
        logging.info(f"Dropped {dropped_feat} rows due to Magpie featurization failures.")
        
    # 3. Clean up columns
    cols_to_drop = [formula_col, 'composition']
    final_df = feat_df.drop(columns=cols_to_drop)
    
    logging.info(f"Final featurized dataset shape: {final_df.shape}")
    return final_df

if __name__ == "__main__":
    input_path = os.path.join("data", "interim", "cleaned_dataset.csv")
    output_path = os.path.join("data", "processed", "featurized_dataset.csv")
    
    logging.info(f"Loading cleaned data from {input_path}")
    df = pd.read_csv(input_path)
    
    featurized_df = generate_features(df, "True Composition")
    
    logging.info(f"Saving to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    featurized_df.to_csv(output_path, index=False)
