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
    
    # 根据数据量确定聚类中心数量（约15%的数据量）
    n_clusters = max(10, min(15, int(len(X_train) * 0.15)))
    # 用于计算SHAP值的样本数量（约30%的数据量）
    n_samples = max(20, min(int(len(X_train) * 0.3), 100))
    
    if model_name in ["LinearRegression", "Ridge", "Lasso", "ElasticNet"]:
        explainer = shap.LinearExplainer(best_model, scaler.transform(X_train))
        shap_values = explainer.shap_values(scaler.transform(X_train))
    elif model_name in ["SVR"]:
        # 对于SVR，使用KernelExplainer
        logger.info(f"使用 {n_clusters} 个聚类中心作为背景数据")
        logger.info(f"使用 {n_samples} 个样本计算SHAP值")
        
        # 使用K-means选择背景数据点，聚类数量根据数据量自适应
        background = shap.kmeans(scaler.transform(X_train), n_clusters)
        explainer = shap.KernelExplainer(best_model.predict, background)
        
        # 随机选择样本计算SHAP值
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
        # 对于AdaBoost，使用feature_importances_
        importances = best_model.feature_importances_
    elif model_name in ["MLP", "KNR"]:
        # 对于MLP和KNR，使用与SVR相同的方法
        logger.info(f"使用 {n_clusters} 个聚类中心作为背景数据")
        logger.info(f"使用 {n_samples} 个样本计算SHAP值")
        
        background = shap.kmeans(scaler.transform(X_train), n_clusters)
        explainer = shap.KernelExplainer(best_model.predict, background)
        
        sample_indices = np.random.choice(len(X_train), n_samples, replace=False)
        X_sample = X_train.iloc[sample_indices]
        shap_values = explainer.shap_values(scaler.transform(X_sample))
        feature_importance = np.abs(shap_values).mean(0)
    else:
        raise ValueError(f"Unsupported model type for SHAP analysis: {model_name}")

    if model_name in ["RandomForest", "GradientBoosting", "XGBoost", "LightGBM", "AdaBoost"]:
        # 对于基于树的模型，直接使用feature_importances_
        feature_importance = importances
    elif model_name in ["LinearRegression", "Ridge", "Lasso", "ElasticNet"]:
        # 对于线性模型，使用系数的绝对值
        feature_importance = np.abs(importances) if importances is not None else np.abs(shap_values).mean(0)
    elif model_name in ["SVR", "MLP", "KNR"]:  # 这些模型已经在上面计算过了
        pass  # feature_importance已经计算好了
    else:
        # 确保shap_values不为None
        if shap_values is not None:
            feature_importance = np.mean(np.abs(shap_values), axis=0)
        else:
            logger.error(f"无法计算特征重要性: {model_name}的shap_values为None")
            # 返回均匀分布的特征重要性
            feature_importance = np.ones(len(X_train.columns)) / len(X_train.columns)

    total_importance = np.sum(feature_importance)
    feature_importance_percent = (feature_importance / total_importance) * 100
    
    # Plot feature importance
    if mae_mean < mae_threshold and shap_values is not None:
        if model_name in ["SVR", "MLP", "KNR"]:
            # 对于使用部分样本的模型，使用相同的样本进行可视化
            shap.summary_plot(
                shap_values, 
                scaler.transform(X_sample),  # 使用与SHAP值计算相同的数据
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
