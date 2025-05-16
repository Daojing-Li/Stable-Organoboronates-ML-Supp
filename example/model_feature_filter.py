import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from src.feature_filter import feature_filter
from sklearn.ensemble import GradientBoostingRegressor

models = {
    'SVR': SVR,
    # 'GradientBoosting': GradientBoostingRegressor,
}

data = pd.read_csv('data_new.csv')
# 选择data中的'pka_H', 'dipole_H', 'homo_H', 'electronegativity_H', 'APT_charge_C_H','NPA_charge_C_H', 'Mulliken_charge_H_H', 'NPA_charge_H_H', 'NICS0','NICS1', 'pka_B', 'delta_G_B', 'delta_G_B_TS', 'homo_B','electronegativity_B' 创建data1
data = data[['pka_H', 'dipole_H', 'homo_H', 'electronegativity_H', 'APT_charge_C_H','NPA_charge_C_H', 'Mulliken_charge_H_H', 'NPA_charge_H_H', 'NICS0','NICS1', 'pka_B', 'delta_G_B', 'delta_G_B_TS', 'homo_B','electronegativity_B','yield']]
# scaler = StandardScaler()
features = data.select_dtypes(include=[np.number]).columns # 选择数值列
X = data[features].drop('yield', axis=1)  # 假设目标列名为 'yield'
# X = X.drop(columns=['sub_H']) # 删除作为序列的sub_H列
y = data['yield']
results, char_change = feature_filter(models, X, y, n_trials=100,max_features=7,mae_threshold=16)