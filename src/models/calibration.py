import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

class RecessionCalibrator:
    """
    Calibrator to map model predictions to calibrated probabilities.
    method: "none" | "sigmoid" | "isotonic"
    
    IMPORTANT:
    - Calibrator must only be fit on training/inner-validation data.
    - Never fit on test data.
    """
    def __init__(self, method: str = "none"):
        self.method = method.lower()
        self.calibrator = None
        
    def fit(self, y_prob: np.ndarray, y_true: np.ndarray):
        """Fit calibrator on validation probabilities and true targets.
        y_prob is 1D array of probabilities (P(Y=1)) or 2D array.
        """
        if self.method == "none":
            return self
            
        if len(y_prob.shape) > 1:
            y_prob_1d = y_prob[:, 1]
        else:
            y_prob_1d = y_prob
            
        # Clip to avoid numerical instability
        y_prob_1d = np.clip(y_prob_1d, 1e-7, 1 - 1e-7)
        
        if self.method == "sigmoid":
            # Platt scaling: fit logistic regression on logit of probabilities
            X_logit = np.log(y_prob_1d / (1 - y_prob_1d)).reshape(-1, 1)
            self.calibrator = LogisticRegression(C=1e5, solver='lbfgs')
            self.calibrator.fit(X_logit, y_true)
            
        elif self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_prob_1d, y_true)
            
        return self
        
    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """Calibrate test probabilities."""
        if self.method == "none" or self.calibrator is None:
            return y_prob
            
        is_2d = len(y_prob.shape) > 1
        if is_2d:
            y_prob_1d = y_prob[:, 1]
        else:
            y_prob_1d = y_prob
            
        y_prob_1d = np.clip(y_prob_1d, 1e-7, 1 - 1e-7)
        
        if self.method == "sigmoid":
            X_logit = np.log(y_prob_1d / (1 - y_prob_1d)).reshape(-1, 1)
            cal_prob = self.calibrator.predict_proba(X_logit)[:, 1]
        elif self.method == "isotonic":
            cal_prob = self.calibrator.predict(y_prob_1d)
        else:
            cal_prob = y_prob_1d
            
        if is_2d:
            return np.column_stack([1 - cal_prob, cal_prob])
        return cal_prob
