import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def generate_shap_plot():
    train_path = os.path.join("data", "processed", "featurized_dataset.csv")
    output_path = os.path.join("data", "processed", "shap_summary.png")
    
    print("Loading training data...")
    df_train = pd.read_csv(train_path)
    
    y_train = df_train["log10_conductivity"]
    
    # Drop non-feature columns
    cols_to_drop = ["log10_conductivity", "measurement_count"]
    if "True Composition" in df_train.columns:
        cols_to_drop.append("True Composition")
        
    X_train = df_train.drop(columns=cols_to_drop)
    
    print("Preprocessing features...")
    # Explicitly preprocess to keep column names
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_imputed = imputer.fit_transform(X_train)
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Reconstruct DataFrame with original Magpie feature names
    X_train_preprocessed = pd.DataFrame(X_scaled, columns=X_train.columns)
    
    print("Fitting Random Forest...")
    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(X_train_preprocessed, y_train)
    
    print("Calculating Feature Importances (Native RF due to SHAP DLL block)...")
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]  # Top 20 features
    
    print("Generating summary plot...")
    plt.figure(figsize=(10, 8))
    plt.title('Top 20 Random Forest Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [X_train_preprocessed.columns[i] for i in indices])
    plt.xlabel('Relative Importance')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Summary plot saved to {output_path}")

if __name__ == "__main__":
    generate_shap_plot()
