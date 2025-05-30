# find and evaluate the best model and features
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
import optuna
from src.visualization import plot_r2_on_100_random_samples, plot_scatter

# Set optuna logging level
from src.logger_config import setup_logger
import os
os.makedirs('models', exist_ok=True)
logger = setup_logger("models")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Import iterative optimization
from src.iterative_optimization import iterative_optimization
import argparse

# Define models
models = {
    # 'LinearRegression': LinearRegression,
    # 'Ridge': Ridge,
    # 'Lasso': Lasso,
    'SVR': SVR,
    # 'DecisionTree': DecisionTreeRegressor,
    # 'RandomForest': RandomForestRegressor,
    # 'GradientBoosting': GradientBoostingRegressor,
    # 'XGBoost': XGBRegressor,
    # 'MLP': MLPRegressor,
    # 'AdaBoost': AdaBoostRegressor,
    # 'ElasticNet': ElasticNet,
    # 'KNR':KNeighborsRegressor
}

def evaluate_baseline(baseline, X, y):
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Training and prediction
    baseline.fit(scaler.fit_transform(X_train), y_train)
    y_pred_train = baseline.predict(scaler.transform(X_train))
    y_pred_test = baseline.predict(scaler.transform(X_test))
    
    # Calculate evaluation metrics
    plot_scatter(y_train, y_pred_train, y_test, y_pred_test, 
                 model_name='Baseline', mae_mean=None,
                 output_dir='models/Baseline/',
                 output_name='baseline_scatter.png',
                 X_train=X_train,
                 X_test=X_test)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Machine learning model training and evaluation')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of CPU cores to use (-1 means using all cores)')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of trials for Optuna optimization')
    parser.add_argument('--mae_threshold', type=float, default=20, help='MAE threshold')
    parser.add_argument('--keep_versions', type=int, default=2, help='Number of versions to keep for each model')
    parser.add_argument('--min_features', type=int, default=5, help='Minimum number of features')
    args = parser.parse_args()
    
    # Read data
    data = pd.read_csv('example/data_round2.csv')
    features = data.select_dtypes(include=[np.number]).columns
    X = data[features].drop('yield', axis=1)
    # X = X.drop(columns=['sub_H']) # Delete sub_H column used as sequence
    y = data['yield']
    # print(X.head())
    # print(y.head())
     
    # Evaluate baseline model
    # baseline = Baseline()
    # evaluate_baseline(baseline, X, y)
    
    # Iterative optimization
    results, best_models = iterative_optimization(
        models, 
        X, 
        y, 
        n_trials=args.n_trials, 
        mae_threshold=args.mae_threshold,
        n_jobs=args.n_jobs,
        keep_versions=args.keep_versions,
        min_features=args.min_features
    )

if __name__ == '__main__':
    main()
