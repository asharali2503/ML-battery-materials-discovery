import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred) -> dict:
    """
    Calculates standard regression metrics: MAE, RMSE, and R2.
    Rounds all values to 4 decimal places.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': round(float(mae), 4),
        'RMSE': round(float(rmse), 4),
        'R2': round(float(r2), 4)
    }
