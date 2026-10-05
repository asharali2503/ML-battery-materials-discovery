import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def run_final_inference():
    train_path = os.path.join("data", "processed", "featurized_dataset.csv")
    candidate_path = os.path.join("data", "processed", "mp_candidates_featurized.csv")
    output_path = os.path.join("data", "processed", "top_candidates_ranked.csv")
    
    print("Loading training data...")
    df_train = pd.read_csv(train_path)
    y_train = df_train["log10_conductivity"]
    
    # Drop non-feature columns
    cols_to_drop_train = ["log10_conductivity", "measurement_count"]
    X_train = df_train.drop(columns=cols_to_drop_train)
    
    print("Building and fitting Random Forest Pipeline...")
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    
    print("Loading candidate data...")
    df_cand = pd.read_csv(candidate_path)
    
    # Drop metadata to isolate features
    meta_cols = ["material_id", "formula_pretty", "energy_above_hull", "formation_energy_per_atom"]
    X_candidate = df_cand.drop(columns=meta_cols)
    
    # Ensure exact column match in case order differs
    X_candidate = X_candidate[X_train.columns]
    
    print("Running inference on candidates...")
    y_pred = pipeline.predict(X_candidate)
    
    df_cand["predicted_log10_conductivity"] = y_pred
    
    # Sort descending
    df_cand = df_cand.sort_values(by="predicted_log10_conductivity", ascending=False)
    
    print("Saving ranked candidates...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_cand.to_csv(output_path, index=False)
    
    print("\n--- TOP 5 CANDIDATES ---")
    print(df_cand[["formula_pretty", "energy_above_hull", "predicted_log10_conductivity"]].head(5).to_string(index=False))

if __name__ == "__main__":
    run_final_inference()
