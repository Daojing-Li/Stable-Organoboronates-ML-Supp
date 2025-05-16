from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.train_and_evaluate import train_and_evaluate
import logging
logger = logging.getLogger(__name__)

def hyperparameter_optimization_and_training(
    model_class, X, y, n_trials=100, random_state=42,n_jobs=-1
):
    """
    Perform hyperparameter optimization and train the model.

    Parameters:
    model_class (class): The model class.
    X (DataFrame): The feature data.
    y (Series): The target data.

    Returns:
    best_model (object): The trained best model.
    scaler (object): The standardization scaler.
    X_train (DataFrame): The training set feature data.
    X_test (DataFrame): The test set feature data.
    y_train (Series): The training set target data.
    y_test (Series): The test set target data.
    rmse_mean (float): The average RMSE.
    rmse_test (float): The test set RMSE.
    best_params (dict): The best hyperparameters.
    """
    mae_loo_mean, mae_mean, best_params = train_and_evaluate(
        model_class, X, y, random_state=42, n_trials=n_trials,n_jobs=n_jobs
    )
    best_model = model_class(**best_params)
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # best_model.fit(scaler.fit_transform(X_train), y_train)
    return best_model, scaler, X_train, X_test, y_train, y_test, mae_mean, best_params
