import numpy as np
import pandas as pd
import logging
import os
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from src.models.validation import get_cv_strategy
from src.models.metrics import calculate_metrics

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def evaluate_model(X, y, model, cv_strategy) -> dict:
    """
    Evaluates a model using the provided cross-validation strategy.
    Ensures absolute separation between train and test sets to prevent leakage.
    Returns the averaged MAE, RMSE, and R2 across all folds.
    """
    metrics_list = []
    
    # Ensure X and y are Pandas objects to safely use .iloc
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    for train_idx, test_idx in cv_strategy.split(X, y):
        # Strict slicing
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Fit strictly on train
        model.fit(X_train, y_train)
        
        # Predict purely on test
        y_pred = model.predict(X_test)
        
        # Calculate fold metrics
        fold_metrics = calculate_metrics(y_test, y_pred)
        metrics_list.append(fold_metrics)
        
    # Average the metrics
    avg_metrics = {
        'MAE': round(np.mean([m['MAE'] for m in metrics_list]), 4),
        'RMSE': round(np.mean([m['RMSE'] for m in metrics_list]), 4),
        'R2': round(np.mean([m['R2'] for m in metrics_list]), 4)
    }
    
    return avg_metrics

if __name__ == "__main__":
    data_path = os.path.join("data", "processed", "featurized_dataset.csv")
    logging.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    y = df['log10_conductivity']
    X = df.drop(columns=['log10_conductivity', 'measurement_count'])
    
    cv = get_cv_strategy(n_splits=5, random_state=42)
    
    # 1. Baseline
    model = DummyRegressor(strategy="mean")
    logging.info("Evaluating Baseline DummyRegressor...")
    baseline_metrics = evaluate_model(X, y, model, cv)
    logging.info(f"Baseline Metrics: {baseline_metrics}")
    
    # 2. Ridge Pipeline
    ridge_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0))
    ])
    logging.info("Evaluating Ridge Regression Pipeline...")
    ridge_metrics = evaluate_model(X, y, ridge_pipe, cv)
    logging.info(f"Ridge Metrics: {ridge_metrics}")
    
    # 3. Random Forest Pipeline
    rf_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    logging.info("Evaluating Random Forest Pipeline...")
    rf_metrics = evaluate_model(X, y, rf_pipe, cv)
    logging.info(f"Random Forest Metrics: {rf_metrics}")

    # 4. XGBoost Pipeline
    xgb_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror'))
    ])
    logging.info("Evaluating XGBoost Pipeline...")
    xgb_metrics = evaluate_model(X, y, xgb_pipe, cv)
    logging.info(f"XGBoost Metrics: {xgb_metrics}")
