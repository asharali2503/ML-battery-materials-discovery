from src.models.metrics import calculate_metrics

def test_perfect_prediction():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 3.0]
    
    metrics = calculate_metrics(y_true, y_pred)
    
    assert metrics['MAE'] == 0.0
    assert metrics['RMSE'] == 0.0
    assert metrics['R2'] == 1.0

def test_known_error():
    y_true = [1.0, 2.0]
    y_pred = [2.0, 3.0]
    
    metrics = calculate_metrics(y_true, y_pred)
    
    assert metrics['MAE'] == 1.0
    assert metrics['RMSE'] == 1.0
