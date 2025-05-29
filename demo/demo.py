#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
演示脚本: 演示基于有机硼化合物数据的机器学习预测

该脚本展示了如何使用我们的机器学习模型来预测有机硼化合物反应的产率。
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# 确保src目录在Python路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目核心功能模块
from src.feature_importance_analysis import feature_importance_analysis
from src.feature_correlation_analysis import feature_correlation_analysis
from src.leave_one_out_validation import leave_one_out_validation
from src.logger_config import setup_logger

# 设置随机种子，确保结果可复现
np.random.seed(42)

# 设置日志
logger = setup_logger("demo")

def load_data():
    """
    加载演示数据集
    """
    logger.info("加载数据集...")
    # 优先使用演示数据集，如果不存在则使用完整数据集
    demo_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_data.csv")
    if os.path.exists(demo_data_path):
        logger.info(f"使用演示数据集: {demo_data_path}")
        data = pd.read_csv(demo_data_path)
    else:
        # 使用完整数据集作为备选
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                "example", "data_round2.csv")
        logger.info(f"使用完整数据集: {data_path}")
        data = pd.read_csv(data_path)
    
    logger.info(f"数据集大小: {data.shape}")
    return data

def prepare_data(data):
    """
    准备训练和测试数据
    """
    logger.info("准备数据...")
    # 提取数值型特征
    features = data.select_dtypes(include=[np.number]).columns
    X = data[features].drop('yield', axis=1)
    y = data['yield']
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, X, y

def train_model(X_train, y_train):
    """
    训练SVR模型
    """
    logger.info("训练模型...")
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 创建SVR模型
    model = SVR(C=10.0, epsilon=0.1, gamma='scale', kernel='rbf')
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

def analyze_features(model, X_train, X, mae, model_name="SVR"):
    """
    使用src中的功能分析特征重要性和相关性
    """
    logger.info("分析特征重要性和相关性...")
    
    # 特征重要性分析
    logger.info("执行特征重要性分析...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    feature_importance_percent = feature_importance_analysis(
        model_name, model, scaler, X_train, mae, 30.0
    )
    
    # 特征相关性分析
    logger.info("执行特征相关性分析...")
    corr_matrix = feature_correlation_analysis(X, mae, 30.0)
    
    return feature_importance_percent, corr_matrix

def evaluate_model_performance(model, scaler, X_train, X_test, y_train, y_test, X, y):
    """
    使用src中的功能评估模型性能
    """
    logger.info("评估模型性能...")
    
    # 基本评估
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # 计算性能指标
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    logger.info(f"训练集 MAE: {mae_train:.2f}, R²: {r2_train:.2f}, RMSE: {rmse_train:.2f}")
    logger.info(f"测试集 MAE: {mae_test:.2f}, R²: {r2_test:.2f}, RMSE: {rmse_test:.2f}")
    
    # 留一交叉验证
    logger.info("执行留一交叉验证...")
    r2_loo = leave_one_out_validation(model, scaler, X, y)
    logger.info(f"留一交叉验证 R²: {r2_loo:.2f}")
    
    # 分析异常值
    logger.info("分析异常值...")
    train_diff = np.abs(y_train - y_pred_train)
    test_diff = np.abs(y_test - y_pred_test)
    
    train_outliers = train_diff >= 30
    test_outliers = test_diff >= 30
    
    if np.any(train_outliers) or np.any(test_outliers):
        logger.info(f"发现异常值 (偏差 >= 30):")
        logger.info(f"训练集异常值: {np.sum(train_outliers)}")
        logger.info(f"测试集异常值: {np.sum(test_outliers)}")
        
        # 创建异常值数据框
        outliers_data = []
        
        if np.any(train_outliers):
            train_outliers_df = pd.DataFrame({
                'Set': ['Train'] * np.sum(train_outliers),
                'Actual': y_train[train_outliers],
                'Predicted': y_pred_train[train_outliers],
                'Difference': train_diff[train_outliers]
            })
            outliers_data.append(train_outliers_df)
            
        if np.any(test_outliers):
            test_outliers_df = pd.DataFrame({
                'Set': ['Test'] * np.sum(test_outliers),
                'Actual': y_test[test_outliers],
                'Predicted': y_pred_test[test_outliers],
                'Difference': test_diff[test_outliers]
            })
            outliers_data.append(test_outliers_df)
            
        if outliers_data:
            outliers_df = pd.concat(outliers_data, axis=0)
            outliers_df = outliers_df.sort_values('Difference', ascending=False)
            
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
            os.makedirs(output_dir, exist_ok=True)
            outliers_path = os.path.join(output_dir, "outliers.csv")
            outliers_df.to_csv(outliers_path, index=True)
            logger.info(f"异常值数据已保存至: {outliers_path}")
    
    return mae_test, r2_test, r2_loo, y_pred_train, y_pred_test, rmse_test

def plot_results(y_train, y_pred_train, y_test, y_pred_test, metrics, r2_loo=None):
    """
    创建结果可视化，包含完整的评估指标
    """
    logger.info("创建结果可视化...")
    plt.figure(figsize=(10, 8))
    
    # 绘制完美预测的参考线
    max_val = max(np.max(y_train), np.max(y_test))
    min_val = min(np.min(y_train), np.min(y_test))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='完美预测')
    
    # 计算偏差
    train_diff = np.abs(y_train - y_pred_train)
    test_diff = np.abs(y_test - y_pred_test)
    
    # 找出正常值和异常值的索引
    train_normal = train_diff < 30
    train_outlier = train_diff >= 30
    test_normal = test_diff < 30
    test_outlier = test_diff >= 30
    
    # 绘制正常值和异常值
    plt.scatter(y_train[train_normal], y_pred_train[train_normal], color='blue', label='训练集')
    plt.scatter(y_test[test_normal], y_pred_test[test_normal], color='green', label='测试集')
    
    # 绘制异常值
    if np.any(train_outlier) or np.any(test_outlier):
        outlier_actual = np.concatenate([y_train[train_outlier], y_test[test_outlier]])
        outlier_pred = np.concatenate([y_pred_train[train_outlier], y_pred_test[test_outlier]])
        if len(outlier_actual) > 0:
            plt.scatter(outlier_actual, outlier_pred, color='red', marker='x', s=100, label='异常值')
    
    plt.xlabel('实际产率 (%)', fontsize=12)
    plt.ylabel('预测产率 (%)', fontsize=12)
    plt.title('SVR模型: 实际产率 vs 预测产率', fontsize=14)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True)
    
    # 添加评估指标文本
    text_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    metrics_text = [
        f"训练集 R²: {metrics['r2_train']:.2f}",
        f"测试集 R²: {metrics['r2_test']:.2f}",
        f"训练集 MAE: {metrics['mae_train']:.2f}",
        f"测试集 MAE: {metrics['mae_test']:.2f}",
        f"训练集 RMSE: {metrics['rmse_train']:.2f}",
        f"测试集 RMSE: {metrics['rmse_test']:.2f}"
    ]
    
    if r2_loo is not None:
        metrics_text.append(f"留一交叉验证 R²: {r2_loo:.2f}")
    
    plt.text(0.05, 0.95, '\n'.join(metrics_text), transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', bbox=text_props)
    
    # 保存图像
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'prediction_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"结果图像已保存到 {output_dir}/prediction_results.png")

def main():
    """
    主函数
    """
    logger.info("=" * 50)
    logger.info("开始演示...")
    logger.info("=" * 50)
    
    # 加载数据
    data = load_data()
    
    # 准备数据
    X_train, X_test, y_train, y_test, X, y = prepare_data(data)
    
    # 训练模型
    model, scaler = train_model(X_train, y_train)
    
    # 评估模型
    mae_test, r2_test, r2_loo, y_pred_train, y_pred_test, rmse_test = evaluate_model_performance(
        model, scaler, X_train, X_test, y_train, y_test, X, y
    )
    
    # 分析特征
    feature_importance, corr_matrix = analyze_features(model, X_train, X, mae_test)
    
    # 计算所有指标
    metrics = {
        'r2_train': r2_score(y_train, y_pred_train),
        'r2_test': r2_test,
        'mae_train': mean_absolute_error(y_train, y_pred_train),
        'mae_test': mae_test,
        'rmse_train': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'rmse_test': rmse_test
    }
    
    # 创建可视化
    plot_results(y_train, y_pred_train, y_test, y_pred_test, metrics, r2_loo)
    
    # 输出特征重要性前五的特征
    logger.info("特征重要性排名前五的特征:")
    feature_names = X.columns
    sorted_idx = np.argsort(feature_importance)[::-1]
    for i in range(min(5, len(feature_names))):
        logger.info(f"  {i+1}. {feature_names[sorted_idx[i]]}: {feature_importance[sorted_idx[i]]:.2f}%")
    
    logger.info("=" * 50)
    logger.info("演示完成!")
    logger.info("=" * 50)
    logger.info(f"请查看 {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')} 目录获取结果图表")

if __name__ == "__main__":
    main() 