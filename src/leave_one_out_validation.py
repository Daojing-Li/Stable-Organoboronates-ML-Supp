from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def leave_one_out_validation(best_model, scaler_X, X_model, y):
    """
    Perform leave-one-out validation.

    Parameters:
    best_model (object): The trained best model.
    scaler_X (object): The feature scaler.
    X_model (DataFrame): The feature data.
    y (Series): The target data.

    Returns:
    r2_loo (float): The R^2 score from leave-one-out validation.
    """
    loo = LeaveOneOut()
    y_pred_loo = []
    y_true_loo = []
    scaler_y = MinMaxScaler(feature_range=(0, 100))
    
    for train_index, test_index in loo.split(X_model):
        X_train_loo, X_test_loo = X_model.iloc[train_index], X_model.iloc[test_index]
        y_train_loo, y_test_loo = y.iloc[train_index], y.iloc[test_index]
        
        # Scale features and target
        X_train_loo_scaled = scaler_X.fit_transform(X_train_loo)
        X_test_loo_scaled = scaler_X.transform(X_test_loo)
        y_train_loo_scaled = scaler_y.fit_transform(y_train_loo.values.reshape(-1, 1)).ravel()
        
        # Train model with scaled data
        best_model.fit(X_train_loo_scaled, y_train_loo_scaled)
        
        # Predict and inverse transform
        y_pred_loo_scaled = best_model.predict(X_test_loo_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_loo_scaled.reshape(-1, 1)).ravel()
        
        # Ensure predictions are within valid range
        y_pred = np.clip(y_pred, 0, 100)
        
        y_pred_loo.append(y_pred[0])
        y_true_loo.append(y_test_loo.values[0])
        
    r2_loo = r2_score(y_true_loo, y_pred_loo)
    return r2_loo
