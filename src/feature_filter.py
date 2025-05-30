import itertools
from sklearn.metrics import r2_score, mean_absolute_error
from src.hyperparameter_optimization_and_training import (
    hyperparameter_optimization_and_training,
)
from src.evaluate_and_plot import evaluate_and_plot
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import logging
import optuna
from src.leave_one_out_validation import leave_one_out_validation

from src.logger_config import setup_logger
logger = setup_logger("output")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def feature_filter(models, X, y, n_trials=50, min_features=5, max_features=8, mae_threshold=16):
    """
    Perform iterative optimization by iterating through all feature combinations.

    Parameters:
    models (dict): A dictionary of model classes.
    X (DataFrame): The feature data.
    y (Series): The target data.
    n_trials (int): The number of trials for hyperparameter optimization.
    min_features (int): The minimum number of features to start with.
    max_features (int): The maximum number of features to consider.
    mae_threshold (float): The threshold for mean absolute error.

    Returns:
    results (dict): The results of the optimization.
    char_change (dict): The character change information.
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler(feature_range=(0, 100))
    results = {}
    char_change = {}
    for model_name, model_class in models.items():
        X_model = X.copy()
        removed_features = []
        final_features = []
        result = []
        num_features = min(len(X_model.columns), max_features)
        min_feat = max(1, min(min_features, num_features))
        for r in range(min_feat, num_features + 1):
            feature_combinations = itertools.combinations(X_model.columns, r)
            for feature_combination in feature_combinations:
                try:
                    X_train = X_model[list(feature_combination)]
                    X_test = X_model[list(feature_combination)]
                    (
                        best_model,
                        scaler,
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        mae_mean,
                        best_params,
                    ) = hyperparameter_optimization_and_training(
                        model_class, X_train, y, n_trials=n_trials
                    )
                    X_train_scaled = scaler_X.fit_transform(X_train)
                    X_test_scaled = scaler_X.transform(X_test)
                    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

                    # Train model with scaled data
                    best_model.fit(X_train_scaled, y_train_scaled)
                    
                    y_pred_test_scaled = best_model.predict(X_test_scaled)
                    y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()
                    y_pred_test = np.clip(y_pred_test, 0, 100)

                    y_pred_train = best_model.predict(X_train_scaled)
                    y_pred_train = scaler_y.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()
                    y_pred_train = np.clip(y_pred_train, 0, 100)

                    # Evaluate model performance
                    evaluate_and_plot(
                        model_name,
                        y_train,
                        y_pred_train,
                        y_test,
                        y_pred_test,
                        X_train,
                        X_test,
                        mae_threshold=mae_threshold,
                        mae_mean=mae_mean,
                        filename=f"number_{r}"
                    )

                    # Calculate additional metrics
                    r2_test = r2_score(y_test, y_pred_test)
                    mae_test = mean_absolute_error(y_test, y_pred_test)
                    r2_loo = leave_one_out_validation(best_model, scaler_X, X_train, y)
                    logger.info(f"r2_loo: {r2_loo}")
                    result.append(
                        {
                            "mae_mean": mae_mean,
                            "r2_test_avg": r2_test,
                            "r2_loo": r2_loo,
                            "mae_test_avg": mae_test,
                            "best_params_avg": best_params,
                            "final_features": list(feature_combination),
                            "filename": f"number_{r}"
                        }
                    )

                except Exception as e:
                    print(
                        f"Error in iterative_optimization for model {model_name}: {e}"
                    )
                    break

            # Sort the results based on MAE (Test) in ascending order and keep only top 5
            result = sorted(result, key=lambda x: x["mae_mean"])[:10]
            results[model_name] = result

    # Print the top 5 models for each model type with their hyperparameters and performance
    for model_name, model_results in results.items():
        logger.info(f"Model: {model_name}")
        for i, result in enumerate(model_results):
            logger.info(f"Rank {i+1}:")
            logger.info(f'Best Params: {result["best_params_avg"]}')
            logger.info(f'Average MAE: {result["mae_mean"]}')
            logger.info(f'R^2 (Test): {result["r2_test_avg"]}')
            logger.info(f'MAE (Test): {result["mae_test_avg"]}')
            logger.info(f'Final Features: {result["final_features"]}')
            logger.info("-" * 40)

    return results, char_change
