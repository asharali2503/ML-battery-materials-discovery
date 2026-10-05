import logging
from sklearn.model_selection import KFold

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_cv_strategy(n_splits=5, random_state=42) -> KFold:
    """
    Returns a deterministic KFold cross-validation strategy 
    to ensure reproducible train/test splits.
    """
    logging.info(f"Initializing {n_splits}-Fold CV with random_state={random_state}")
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
