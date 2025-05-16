import logging
import os
import joblib
from datetime import datetime
import glob
import pandas as pd
import numpy as np

# Set matplotlib backend to non-interactive
import matplotlib
matplotlib.use('Agg')  # 必须在导入pyplot之前设置
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.metrics import r2_score, mean_absolute_error
from src.hyperparameter_optimization_and_training import hyperparameter_optimization_and_training
from src.feature_importance_analysis import feature_importance_analysis
from src.feature_correlation_analysis import feature_correlation_analysis
from src.feature_selection import feature_selection
from src.evaluate_and_plot import evaluate_and_plot
from src.leave_one_out_validation import leave_one_out_validation
from src.logger_config import setup_logger
from sklearn.preprocessing import MinMaxScaler
from utils.visualization import plot_scatter

# Set default font
plt.rcParams['font.family'] = 'DejaVu Sans'


def clean_old_versions(model_dir, keep_versions=2):
    """
    Clean old versions, keep only the latest N versions
    
    Parameters:
    model_dir (str): Model directory path
    keep_versions (int): Number of versions to keep
    """
    logger = logging.getLogger(__name__)
    # Get all version timestamps
    version_timestamps = {}
    for filepath in glob.glob(os.path.join(model_dir, "*_final_*.joblib")):
        try:
            # Extract timestamp from filename
            timestamp = filepath.split("_final_")[-1].replace(".joblib", "")
            version_timestamps[filepath] = timestamp
            # Get corresponding metrics file
            metrics_file = filepath.replace(".joblib", "_metrics.txt")
            if os.path.exists(metrics_file):
                version_timestamps[metrics_file] = timestamp
        except:
            continue
    
    # If version count exceeds limit, delete oldest versions
    if len(set(version_timestamps.values())) > keep_versions:
        # Sort by timestamp
        sorted_files = sorted(version_timestamps.items(), key=lambda x: x[1])
        # Calculate versions to delete
        delete_count = len(set(version_timestamps.values())) - keep_versions
        # Delete oldest files
        for filepath, _ in sorted_files[:delete_count*2]:  # *2 because each version has .joblib and _metrics.txt
            try:
                os.remove(filepath)
                logger.info(f"Deleted old version file: {os.path.basename(filepath)}")
            except Exception as e:
                logger.info(f"Failed to delete file {filepath}: {str(e)}")

    # Clean iteration files
    for filepath in glob.glob(os.path.join(model_dir, "*_iteration_*.joblib")):
        try:
            os.remove(filepath)
            metrics_file = filepath.replace(".joblib", "_metrics.txt")
            if os.path.exists(metrics_file):
                os.remove(metrics_file)
        except Exception as e:
            logger.info(f"Failed to delete iteration file {filepath}: {str(e)}")


def iterative_optimization(models, X, y, n_trials=100, mae_threshold=20, n_jobs=-1, keep_versions=2, min_features=5):
    """
    Perform iterative optimization by removing features based on importance and correlation.

    Parameters:
    models (dict): A dictionary of model classes.
    X (DataFrame): The feature data.
    y (Series): The target data.
    n_trials (int): Number of trials for hyperparameter optimization.
    mae_threshold (float): Threshold for MAE to determine good models.
    n_jobs (int): Number of CPU cores to use (-1 for all cores).
    keep_versions (int): Number of versions to keep for each model.
    min_features (int): Minimum number of features to keep.

    Returns:
    results (dict): The results of the optimization.
    best_models (dict): Dictionary containing the best models for each algorithm.
    """
    results = {}
    best_models = {}
    
    # Create root directory for model saving
    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(models_dir)

    for model_name, model_class in models.items():
        logger.info(f"Starting to train model: {model_name}")
        
        # Create separate directory for each model
        model_dir = os.path.join(models_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Clean old versions
        clean_old_versions(model_dir, keep_versions)
        
        X_model = X.copy()
        removed_features = []
        final_features = []
        best_model = None
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler(feature_range=(0, 100))
        
        # Record performance metrics for each iteration
        performance_history = {
            'iteration': [],
            'mae': [],
            'r2': [],
            'removed_feature': [],
            'remaining_features': [],
        }
        
        iteration = 0
        while True:
            iteration += 1
            logger.info(f"Starting iteration {iteration}")
            
            # Train model with current feature set
            (
                best_model,
                _,
                X_train,
                X_test,
                y_train,
                y_test,
                current_mae,
                best_params,
            ) = hyperparameter_optimization_and_training(
                model_class, X_model, y, n_trials=n_trials, n_jobs=n_jobs,random_state=40
            )
            
            # Scale features and target
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)
            y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
            
            # Train model with scaled data
            best_model.fit(X_train_scaled, y_train_scaled)
            
            # Calculate current performance metrics
            y_pred_test_scaled = best_model.predict(X_test_scaled)
            y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()
            y_pred_test = np.clip(y_pred_test, 0, 100)
            current_r2 = r2_score(y_test, y_pred_test)
            
            # Record current iteration performance
            performance_history['iteration'].append(iteration)
            performance_history['mae'].append(current_mae)
            performance_history['r2'].append(current_r2)
            performance_history['remaining_features'].append(len(X_model.columns))
            performance_history['removed_feature'].append(
                removed_features[-1] if removed_features else "Initial"
            )
            
            # Feature importance analysis
            feature_importance_percent = feature_importance_analysis(
                model_name, best_model, scaler_X, X_train, current_mae, mae_threshold
            )
            
            # Correlation analysis
            corr_matrix = feature_correlation_analysis(X_model, current_mae, mae_threshold)
            
            # Get features to remove
            remove_feature = feature_selection(
                X_model, feature_importance_percent, corr_matrix, removed_features
            )
            
            # Check if minimum feature count limit is reached
            if len(X_model.columns) <= min_features:
                logger.info(f"Minimum feature count limit ({min_features}) reached")
                final_features = X_model.columns.tolist()
                break
                
            if not remove_feature:
                logger.info("No more features to remove")
                final_features = X_model.columns.tolist()
                break
            
            # Output removed features and their importance and correlation
            for feature in remove_feature:
                logger.info(f"\nRemoving feature: {feature}")
                feature_index = X_model.columns.get_loc(feature)
                logger.info(
                    f"Feature importance: {feature_importance_percent[feature_index]:.2f}%"
                )
                logger.info("Correlation with other features:")
                for j, other_feature in enumerate(X_model.columns):
                    if (
                        other_feature != feature
                        and corr_matrix.iloc[feature_index, j] > 0.8
                    ):
                        logger.info(
                            f"  {feature} - {other_feature}: {corr_matrix.iloc[feature_index, j]:.2f}"
                        )
                        
            # Remove selected features
            for feature in remove_feature:
                removed_features.append(feature)
                X_model = X_model.drop(columns=[feature])
                X_train = X_train.drop(columns=[feature])
                X_test = X_test.drop(columns=[feature])
            
            # Save current iteration model and related information
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{model_name}_iteration_{iteration}_{timestamp}"
            model_path = os.path.join(model_dir, model_filename)
            
            # Save all information related to the model
            model_info = {
                'model': best_model,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
                'features': X_model.columns.tolist(),
                'hyperparameters': best_params,
                'metrics': {
                    'mae_mean': current_mae,
                    'r2_test': current_r2,
                    'mae_test': mean_absolute_error(y_test, y_pred_test)
                },
                'removed_features': removed_features.copy()
            }
            joblib.dump(model_info, f"{model_path}.joblib")
            
            # Save performance metrics to separate text file
            with open(f"{model_path}_metrics.txt", 'w') as f:
                f.write(f"Model: {model_name}\n")
                f.write(f"Iteration: {iteration}\n")
                f.write(f"Features: {', '.join(X_model.columns)}\n")
                f.write(f"Removed Features: {', '.join(removed_features)}\n")
                f.write(f"MAE Mean: {current_mae:.4f}\n")
                f.write(f"R² Test: {current_r2:.4f}\n")
                f.write(f"MAE Test: {mean_absolute_error(y_test, y_pred_test):.4f}\n")
                f.write(f"Best Parameters: {best_params}\n")
            
            evaluate_and_plot(
                model_name,
                y_train,
                best_model.predict(X_train_scaled),
                y_test,
                y_pred_test,
                X_train,
                X_test,
                mae_threshold=mae_threshold,
                mae_mean=current_mae,
            )
            
        r2_loo = leave_one_out_validation(best_model, scaler_X, X_model, y)
        result = {
            "mae_mean": current_mae,
            "r2_test_avg": current_r2,
            "mae_test_avg": mean_absolute_error(y_test, y_pred_test),
            "best_params_avg": best_params,
            "r2_loo_avg": r2_loo,
            "final_features": final_features
        }
        results[model_name] = result
        best_models[model_name] = best_model
        
        # Output final performance record here
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_performance_history(
            performance_history,
            model_name,
            os.path.join(model_dir, f"performance_history_{timestamp}.png")
        )
        
        save_performance_history(
            performance_history,
            os.path.join(model_dir, f"performance_history_{timestamp}.csv")
        )
        
        # Save final best model
        final_model_info = {
            'model': best_model,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'features': final_features,
            'hyperparameters': best_params,
            'metrics': {
                'mae_mean': current_mae,
                'r2_test': current_r2,
                'mae_test': mean_absolute_error(y_test, y_pred_test),
                'r2_loo': r2_loo
            },
            'removed_features': removed_features
        }
        final_model_path = os.path.join(model_dir, f"{model_name}_final_{timestamp}")
        joblib.dump(final_model_info, f"{final_model_path}.joblib")
        
        # Save final model performance metrics
        with open(f"{final_model_path}_metrics.txt", 'w') as f:
            f.write(f"Model: {model_name} (Final)\n")
            f.write(f"Features: {', '.join(final_features)}\n")
            f.write(f"Removed Features: {', '.join(removed_features)}\n")
            f.write(f"MAE Mean: {current_mae:.4f}\n")
            f.write(f"R² Test: {current_r2:.4f}\n")
            f.write(f"MAE Test: {mean_absolute_error(y_test, y_pred_test):.4f}\n")
            f.write(f"R² LOO: {r2_loo:.4f}\n")
            f.write(f"Best Parameters: {best_params}\n")
            
        # Plot final scatter plot for this model
        # Get predictions for train and test sets
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        
        # Train model with final parameters
        best_model.fit(X_train_scaled, y_train_scaled)
        
        # Get predictions
        y_pred_train_scaled = best_model.predict(X_train_scaled)
        y_pred_test_scaled = best_model.predict(X_test_scaled)
        
        # Inverse transform predictions
        y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).ravel()
        y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()
        
        # Ensure predictions are within valid range
        y_pred_train = np.clip(y_pred_train, 0, 100)
        y_pred_test = np.clip(y_pred_test, 0, 100)
        
        # Plot scatter plot
        plot_scatter(
            y_train=y_train,
            y_pred_train=y_pred_train,
            y_test=y_test,
            y_pred_test=y_pred_test,
            model_name=model_name,
            mae_mean=current_mae,
            output_dir=model_dir + '/',
            output_name=f'final_scatter_{timestamp}.png',
            X_train=X_train,
            X_test=X_test
        )

    for model_name, result in results.items():
        logger.info(f"Model: {model_name}")
        logger.info(f'Average MAE: {result["mae_mean"]}')
        logger.info(f'R^2 (Test): {result["r2_test_avg"]}')
        logger.info(f'MAE (Test): {result["mae_test_avg"]}')
        logger.info(f'Best Params: {result["best_params_avg"]}')
        logger.info(f'R^2 (LOO): {result["r2_loo_avg"]}')
        logger.info(f'Final Features: {result["final_features"]}')
        logger.info("-" * 40)

    return results, best_models


def plot_performance_history(history, model_name, output_path):
    """
    Plot performance history curve.
    
    Parameters:
    history (dict): Dictionary containing performance metrics history
    model_name (str): Model name
    output_path (str): Output file path
    """
    plt.figure(figsize=(12, 8))
    
    # Create two y-axes
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Plot MAE and R² curves
    line1 = ax1.plot(history['iteration'], history['mae'], 'b-o', label='MAE')
    line2 = ax2.plot(history['iteration'], history['r2'], 'r-o', label='R²')
    
    # Set x-axis to integer ticks
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Reverse MAE axis (lower is better)
    ax1.set_ylim(ax1.get_ylim()[::-1])
    
    # Set labels and title
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('MAE (lower is better)', color='b')
    ax2.set_ylabel('R² (higher is better)', color='r')
    plt.title(f'{model_name} Performance History')
    
    # Merge legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Add feature names and remaining feature count at each point
    for i, (feature, remaining) in enumerate(zip(history['removed_feature'], history['remaining_features'])):
        if i > 0:  # Skip initial point
            plt.annotate(
                f"{feature}\n({remaining} features left)",
                (history['iteration'][i], history['mae'][i]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                rotation=45,
                ha='left'
            )
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_performance_history(history, output_path):
    """
    Save performance history to CSV file.
    
    Parameters:
    history (dict): Dictionary containing performance metrics history
    output_path (str): Output file path
    """
    logger = logging.getLogger(__name__)
    df = pd.DataFrame(history)
    df.to_csv(output_path, index=False)
    logger.info(f"Performance history saved to: {output_path}")
