import shap
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor

logger = logging.getLogger(__name__)

def feature_importance_analysis(
    model_name, best_model, scaler, X_train, mae_mean, mae_threshold
):
    """
    Analyze the feature importance using SHAP values.

    Parameters:
    model_name (str): The model name.
    best_model (object): The trained best model.
    scaler (object): The standardization scaler.
    X_train (DataFrame): The training set feature data.

    Returns:
    feature_importance_percent (array): The feature importance in percentage.
    """
    importances = None
    shap_values = None
    
    # Determine the number of clusters based on data size (about 15% of data size)
    n_clusters = max(10, min(15, int(len(X_train) * 0.15)))
    # Number of samples for SHAP value calculation (about 30% of data size)
    n_samples = max(20, min(int(len(X_train) * 0.3), 100))
    
    if model_name in ["LinearRegression", "Ridge", "Lasso", "ElasticNet"]:
        explainer = shap.LinearExplainer(best_model, scaler.transform(X_train))
        shap_values = explainer.shap_values(scaler.transform(X_train))
    elif model_name in ["SVR"]:
        # For SVR, use KernelExplainer
        logger.info(f"Using {n_clusters} cluster centers as background data")
        logger.info(f"Using {n_samples} samples to calculate SHAP values")
        
        # Use K-means to select background data points, number of clusters adapts to data size
        background = shap.kmeans(scaler.transform(X_train), n_clusters)
        explainer = shap.KernelExplainer(best_model.predict, background)
        
        # Randomly select samples to calculate SHAP values
        sample_indices = np.random.choice(len(X_train), n_samples, replace=False)
        X_sample = X_train.iloc[sample_indices]
        shap_values = explainer.shap_values(scaler.transform(X_sample))
        feature_importance = np.abs(shap_values).mean(0)
    elif model_name in [
        "DecisionTree",
        "RandomForest",
        "GradientBoosting",
        "XGBoost",
        "LightGBM",
    ]:
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(scaler.transform(X_train))
        importances = np.abs(shap_values).mean(0)
    elif model_name in ["AdaBoost"]:
        # For AdaBoost, use feature_importances_
        importances = best_model.feature_importances_
    elif model_name in ["MLP", "KNR"]:
        # For MLP and KNR, use the same method as SVR
        logger.info(f"Using {n_clusters} cluster centers as background data")
        logger.info(f"Using {n_samples} samples to calculate SHAP values")
        
        background = shap.kmeans(scaler.transform(X_train), n_clusters)
        explainer = shap.KernelExplainer(best_model.predict, background)
        
        sample_indices = np.random.choice(len(X_train), n_samples, replace=False)
        X_sample = X_train.iloc[sample_indices]
        shap_values = explainer.shap_values(scaler.transform(X_sample))
        feature_importance = np.abs(shap_values).mean(0)
    else:
        raise ValueError(f"Unsupported model type for SHAP analysis: {model_name}")

    if model_name in ["RandomForest", "GradientBoosting", "XGBoost", "LightGBM", "AdaBoost"]:
        # For tree-based models, directly use feature_importances_
        feature_importance = importances
    elif model_name in ["LinearRegression", "Ridge", "Lasso", "ElasticNet"]:
        # For linear models, use the absolute value of coefficients
        feature_importance = np.abs(importances) if importances is not None else np.abs(shap_values).mean(0)
    elif model_name in ["SVR", "MLP", "KNR"]:  # These models have already been calculated above
        pass  # feature_importance has already been calculated
    else:
        # Ensure shap_values is not None
        if shap_values is not None:
            feature_importance = np.mean(np.abs(shap_values), axis=0)
        else:
            logger.error(f"Unable to calculate feature importance: shap_values for {model_name} is None")
            # Return uniformly distributed feature importance
            feature_importance = np.ones(len(X_train.columns)) / len(X_train.columns)

    total_importance = np.sum(feature_importance)
    feature_importance_percent = (feature_importance / total_importance) * 100
    
    # Plot feature importance
    if mae_mean < mae_threshold and shap_values is not None:
        if model_name in ["SVR", "MLP", "KNR"]:
            # For models using partial samples, use the same samples for visualization
            shap.summary_plot(
                shap_values, 
                scaler.transform(X_sample),  # Use the same data as SHAP value calculation
                feature_names=X_train.columns, 
                plot_type="bar"
            )
        else:
            shap.summary_plot(
                shap_values, 
                X_train, 
                feature_names=X_train.columns, 
                plot_type="bar"
            )
    return feature_importance_percent
