import optuna
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsRegressor
import logging

logger = logging.getLogger(__name__)

def train_and_evaluate(model_class, X, y, random_state=42, n_trials=100, n_jobs=-1):
    """
    Train and evaluate a regression model using Optuna for hyperparameter tuning.

    Parameters:
    model_class (class): The regression model class to be used (e.g., LinearRegression, Ridge).
    X (DataFrame): The feature matrix.
    y (Series): The target vector.
    random_state (int): The random seed for reproducibility (default is 42).
    n_trials (int): Number of trials for Optuna optimization (default is 100).
    n_jobs (int): Number of CPU cores to use (-1 for all cores).

    Returns:
    tuple: A tuple containing the best mean absolute error (MAE) from Optuna, the average MAE from test sets, and the best hyperparameters.
    """
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    logger.info(f"n_jobs: {n_jobs}")

    def objective(trial):
        """
        Define the objective function for Optuna to optimize.

        Parameters:
        trial (Trial): A trial object for hyperparameter sampling.

        Returns:
        float: The mean absolute error (MAE) for the current trial.
        """
        params = {}
        if model_class == LinearRegression:
            params = {}
        elif model_class == Ridge:
            params = {"alpha": trial.suggest_float("alpha", 1e-2, 1e2, log=True)}
        elif model_class == Lasso:
            params = {
                "alpha": trial.suggest_float("alpha", 1e-4, 1.0, log=True),
                "max_iter": 10000,
                "tol": trial.suggest_float("tol", 1e-5, 1e-3, log=True),
                "selection": "random",
                "random_state": 42,
            }
        elif model_class == ElasticNet:
            params = {
                "alpha": trial.suggest_float("alpha", 1e-1, 100.0, log=True),  # 增大正则化强度
                "l1_ratio": trial.suggest_float("l1_ratio", 0.1, 0.9),  # 保持L1比例范围
                "max_iter": 50000,  # 显著增加最大迭代次数
                "tol": trial.suggest_float("tol", 1e-3, 1e-1, log=True),  # 显著放宽收敛容差
                "selection": "cyclic",  # 使用循环选择策略
                "random_state": 42,
                "fit_intercept": True,
                "normalize": True,  # 启用特征归一化
                "precompute": True,
                "warm_start": True,
                # "positive": True,  # 强制系数为正（如果您的问题中系数应该为正）
                "copy_X": True
            }
        elif model_class == SVR:
            params = {
                "C": trial.suggest_float("C", 1e-1, 1e3, log=True),  # 扩大C的范围
                "epsilon": trial.suggest_float("epsilon", 1e-2, 10, log=True),
                "kernel": "rbf",
                "gamma": trial.suggest_float("gamma", 1e-2, 10, log=True),  # 扩大gamma的范围
                "tol": 1e-3,
                "max_iter": 10000,  # 增加最大迭代次数
                "cache_size": 1000,
                # "shrinking": True,
            }
        elif model_class == DecisionTreeRegressor:
            params = {
                "max_depth": trial.suggest_int("max_depth", 5, 15),  # 允许更深的树
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 5),  # 减小分裂所需的最小样本数
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 3),  # 允许更小的叶节点
                "criterion": trial.suggest_categorical("criterion", ["squared_error", "friedman_mse"]),  # 专注于均方误差相关指标
                "splitter": "best",  # 固定为最佳分裂
                "max_features": None,  # 使用所有特征
                "random_state": 42,
                "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0, 0.05),  # 添加剪枝参数
            }
        elif model_class == RandomForestRegressor:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 10, 300),
                "max_depth": trial.suggest_int("max_depth", 1, 3),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                "n_jobs": n_jobs,
                "random_state": 42,
            }
        elif model_class == GradientBoostingRegressor:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 150),  # 适度的树数量
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),  # 回到更小的学习率范围
                "max_depth": trial.suggest_int("max_depth", 2, 4),  # 中等树深度
                "min_samples_split": trial.suggest_int("min_samples_split", 3, 8),  # 适度的分裂要求
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 5),  # 适度的叶节点要求
                "subsample": trial.suggest_float("subsample", 0.5, 0.9),  # 更平衡的采样率
                "max_features": trial.suggest_float("max_features", 0.5, 0.9),  # 更平衡的特征选择
                "loss": "squared_error",  # 回到标准均方误差损失函数
                "random_state": 42,
                # 早停机制
                # "validation_fraction": 0.2,  # 使用20%数据作为验证集
                "n_iter_no_change": 10,  # 10次迭代无改善则停止
                "tol": 1e-4  # 合理的容差阈值
            }
        elif model_class == XGBRegressor:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),  # 减少最大树数量
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),  # 提高学习率下限
                "max_depth": trial.suggest_int("max_depth", 2, 4),  # 限制树的深度
                "min_child_weight": trial.suggest_int("min_child_weight", 3, 10),  # 大幅增加子节点权重要求
                "subsample": trial.suggest_float("subsample", 0.5, 0.8),  # 减少每棵树使用的样本比例
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),  # 减少每棵树使用的特征比例
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 0.8),  # 在每个级别上随机选择特征
                "gamma": trial.suggest_float("gamma", 0.1, 5.0),  # 增加进行分裂所需的最小损失减少值
                "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),  # 增强L1正则化
                "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0, log=True),  # 增强L2正则化
                "scale_pos_weight": 1,  # 平衡权重
                "base_score": 0.5,  # 初始预测分数
                "max_delta_step": trial.suggest_int("max_delta_step", 0, 5),  # 限制每棵树的权重估计
                "grow_policy": "depthwise",  # 标准生长策略
                "n_jobs": n_jobs,
                "random_state": 42,
                "tree_method": "hist",
                # 移除早停机制，因为LOO交叉验证中没有提供验证集
                # "early_stopping_rounds": 10,  # 10轮无改善则停止
            }
        elif model_class == LGBMRegressor:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "num_leaves": trial.suggest_int("num_leaves", 2, 64),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                "n_jobs": n_jobs,
                "random_state": 42,
            }
        elif model_class == MLPRegressor:
            # 简化网络结构，调整参数范围
            hidden_layer_choices = [
                (20,),
                (50,),
                (20, 10),
                (50, 25)
            ]
            params = {
                "hidden_layer_sizes": trial.suggest_categorical(
                    "hidden_layer_sizes",
                    hidden_layer_choices  # 直接使用元组列表
                ),
                "activation": trial.suggest_categorical(
                    "activation", ["relu", "tanh"]
                ),
                "alpha": trial.suggest_float("alpha", 1e-3, 1e-1, log=True),  # 增大正则化范围
                "learning_rate": "adaptive",  # 使用自适应学习率
                "learning_rate_init": trial.suggest_float(
                    "learning_rate_init", 1e-2, 1e-1, log=True  # 增大初始学习率
                ),
                "max_iter": 2000,  # 增加最大迭代次数
                "early_stopping": True,
                "validation_fraction": 0.2,
                "n_iter_no_change": 20,  # 减少早停所需迭代次数
                "tol": 1e-3,  # 放宽收敛条件
                "random_state": 42,
                "solver": "adam",  # 明确使用adam优化器
                "batch_size": "auto",  # 自动选择批量大小
            }
        elif model_class == AdaBoostRegressor:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
                "loss": trial.suggest_categorical("loss", ["linear", "square", "exponential"]),
                "random_state": 42,
            }
        elif model_class == KNeighborsRegressor:
            params = {
                "n_neighbors": trial.suggest_int("n_neighbors", 10, 25),  # 增加邻居数量范围，减轻过拟合
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),  # 保留两种权重选项
                "algorithm": trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree"]),  # 添加算法选择
                "metric": trial.suggest_categorical(
                    "metric", ["minkowski", "euclidean", "manhattan", "chebyshev"]
                ),  # 扩展距离度量选择
                "p": trial.suggest_int("p", 1, 3) if trial.params.get("metric", "") == "minkowski" else 2,  # 只在minkowski时调整p
                "leaf_size": trial.suggest_int("leaf_size", 20, 60),  # 增加leaf_size下限以提高稳定性
                "n_jobs": n_jobs
            }
        else:
            raise ValueError(f"Unsupported model class: {model_class}")

        model = model_class(**params)
        mae_list = []
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler(feature_range=(0, 100))
        loo = LeaveOneOut()
        for train_index, test_index in loo.split(X_train):
            X_train_loo, X_test_loo = (
                X_train.iloc[train_index],
                X_train.iloc[test_index],
            )
            y_train_loo, y_test_loo = (
                y_train.iloc[train_index],
                y_train.iloc[test_index],
            )

            # Scale features and target
            X_train_loo_scaled = scaler_X.fit_transform(X_train_loo)
            X_test_loo_scaled = scaler_X.transform(X_test_loo)
            y_train_loo_scaled = scaler_y.fit_transform(y_train_loo.values.reshape(-1, 1)).ravel()

            # Train the model
            model.fit(X_train_loo_scaled, y_train_loo_scaled)

            # Predict and calculate MAE
            y_pred_loo_scaled = model.predict(X_test_loo_scaled)
            y_pred_loo = scaler_y.inverse_transform(y_pred_loo_scaled.reshape(-1, 1)).ravel()
            mae_loo = mean_absolute_error(y_test_loo, y_pred_loo)
            mae_list.append(mae_loo)

        # Calculate the mean MAE
        mae_mean = np.mean(mae_list)
        return mae_mean

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=5, n_ei_candidates=24, seed=42
    )
    study = optuna.create_study(
        sampler=sampler, 
        direction="minimize",
        storage=f"sqlite:///optuna_optimization_{model_class.__name__}.db"  # 自动创建文件
    )
    study.optimize(objective, n_jobs=n_jobs, n_trials=n_trials)

    # Train the model with the best hyperparameters on the entire training set
    best_params = study.best_params
    best_model = model_class(**best_params)
    mae_test_list = []
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler(feature_range=(0, 100))
    
    # 获得随机数为42情况下的性能表现
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features and target
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    
    best_model.fit(X_train_scaled, y_train_scaled)
    
    # Predict and inverse transform
    y_pred_train_scaled = best_model.predict(X_train_scaled)
    y_pred_test_scaled = best_model.predict(X_test_scaled)
    
    y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).ravel()
    y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()
    
    # Ensure predictions are within valid range
    y_pred_train = np.clip(y_pred_train, 0, 100)
    y_pred_test = np.clip(y_pred_test, 0, 100)
    
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    logger.info("random_state = 42")
    logger.info(f"Best hyperparameters: {best_params}")
    logger.info(f"Train R^2: {r2_train:.4f}")
    logger.info(f"Test R^2: {r2_test:.4f}")
    logger.info(f"Test RMSE: {rmse_test:.4f}")
    logger.info(f"Test MAE: {mae_test:.4f}")

    for i in range(100):
        random_state = i
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        
        # Scale features and target
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        
        best_model.fit(X_train_scaled, y_train_scaled)
        
        # Predict and inverse transform
        y_pred_test_scaled = best_model.predict(X_test_scaled)
        y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()
        y_pred_test = np.clip(y_pred_test, 0, 100)
        
        mae_test = mean_absolute_error(y_test, y_pred_test)
        mae_test_list.append(mae_test)
        if mae_test == min(mae_test_list):
            best_random_state = random_state
    mae_test_mean = np.mean(mae_test_list)
    logger.info(f"Average MAE on 100 times random test sets: {mae_test_mean:.4f}")
    logger.info(f"Best random state: {best_random_state} with best MAE: {min(mae_test_list)}")

    # Print current feature combination
    logger.info("Current feature combination:")
    logger.info(X.columns)
    return study.best_value, mae_test_mean, best_params
