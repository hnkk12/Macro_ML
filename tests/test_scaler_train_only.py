import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

def test_scaler_fit_on_train_only():
    # Make dummy training and test sets
    train_data = np.array([[1.0], [2.0], [3.0]])
    test_data = np.array([[10.0], [20.0]])
    
    scaler = StandardScaler()
    scaler.fit(train_data)
    
    # Check that scaler mean and variance are based ONLY on training data
    assert scaler.mean_[0] == 2.0
    assert np.allclose(scaler.var_[0], np.var([1.0, 2.0, 3.0]))
    
    # Transform test data using the train-fitted scaler
    scaled_test = scaler.transform(test_data)
    expected_scaled_test = (test_data - 2.0) / np.sqrt(scaler.var_[0])
    assert np.allclose(scaled_test, expected_scaled_test)
