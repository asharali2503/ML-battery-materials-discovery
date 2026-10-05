import numpy as np
from src.models.validation import get_cv_strategy

def test_get_cv_strategy():
    # 1. Create a dummy array of 100 rows
    X = np.zeros((100, 5))
    
    cv = get_cv_strategy(n_splits=5, random_state=42)
    splits1 = list(cv.split(X))
    
    # 2. Assert exactly 5 folds are generated
    assert len(splits1) == 5
    
    # 3. Assert first fold is 80 train / 20 test
    train_idx, test_idx = splits1[0]
    assert len(train_idx) == 80
    assert len(test_idx) == 20
    
    # 4. Assert exact determinism by calling again with the same seed
    cv2 = get_cv_strategy(n_splits=5, random_state=42)
    splits2 = list(cv2.split(X))
    
    for (train1, test1), (train2, test2) in zip(splits1, splits2):
        assert np.array_equal(train1, train2)
        assert np.array_equal(test1, test2)
