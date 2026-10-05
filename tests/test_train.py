import pandas as pd
import numpy as np
from sklearn.dummy import DummyRegressor
from src.models.train import evaluate_model
from src.models.validation import get_cv_strategy

def test_evaluate_model():
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(20, 3), columns=['f1', 'f2', 'f3'])
    y = pd.Series(np.random.rand(20), name='target')
    
    model = DummyRegressor(strategy="mean")
    cv = get_cv_strategy(n_splits=2, random_state=42)
    
    metrics = evaluate_model(X, y, model, cv)
    
    assert isinstance(metrics, dict)
    assert 'MAE' in metrics
    assert 'RMSE' in metrics
    assert 'R2' in metrics
