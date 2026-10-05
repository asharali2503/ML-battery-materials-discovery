import os
import pandas as pd
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty

def featurize_mp_candidates():
    input_path = os.path.join("data", "raw", "mp_candidates.csv")
    output_path = os.path.join("data", "processed", "mp_candidates_featurized.csv")
    
    print(f"Loading candidates from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Initial shape: {df.shape}")
    
    print("Converting formulas to Composition objects...")
    stc = StrToComposition()
    # Use ignore_errors=True to prevent crashes on any unparseable strings
    df = stc.featurize_dataframe(df, "formula_pretty", ignore_errors=True)
    
    print("Generating Magpie features...")
    ep = ElementProperty.from_preset("magpie")
    df = ep.featurize_dataframe(df, "composition", ignore_errors=True)
    
    print("Cleaning up columns...")
    # Drop the intermediate 'composition' object column which cannot be saved to CSV properly
    df = df.drop(columns=["composition"])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Saved featurized candidates to {output_path}")
    print(f"Final Featurized DataFrame Shape: {df.shape}")

if __name__ == "__main__":
    featurize_mp_candidates()
