import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from src.leave_one_out_validation import leave_one_out_validation
import pandas as pd
import os

def plot_r2_on_100_random_samples(data, model,output_file=None):
    """绘制模型在100个随机测试集上的R2分数分布"""
    features = data.select_dtypes(include=[np.number]).columns
    X = data[features].drop('yield', axis=1)
    y = data['yield']
    scaler = StandardScaler()

    # 计算100次随机划分的MAE
    mae_test_list = []
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        model.fit(scaler.fit_transform(X_train), y_train)
        y_pred_test = model.predict(scaler.transform(X_test))
        mae_test = mean_absolute_error(y_test, y_pred_test)
        mae_test_list.append(mae_test)
    mae_mean = np.mean(mae_test_list)
    r2_loo = leave_one_out_validation(model, scaler, X, y)
    # 绘制散点图
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(scaler.fit_transform(X_train), y_train)
    y_pred_train = model.predict(scaler.transform(X_train))
    y_pred_test = model.predict(scaler.transform(X_test))

    # 添加原始特征数据到plot_scatter调用
    plot_scatter(y_train, y_pred_train, y_test, y_pred_test, 
                model_name=type(model).__name__, 
                mae_mean=mae_mean,
                output_dir='models/',
                output_name=f'{type(model).__name__}_scatter.png',
                X_train=X_train,
                X_test=X_test)
    if output_file is None:
        output_file = 'models/' + f'{type(model).__name__}_r2.png'
    plot_r2_distribution(model, X, y,output_file=output_file)

def plot_scatter(y_train, y_pred_train, y_test, y_pred_test, model_name, mae_mean, output_dir, output_name, 
               X_train=None, X_test=None,r2_loo=None):
    """
    绘制实际值vs预测值的散点图，并导出异常值数据
    
    Parameters:
    -----------
    y_train, y_pred_train: 训练集的实际值和预测值
    y_test, y_pred_test: 测试集的实际值和预测值
    model_name: 模型名称
    mae_mean: 平均MAE
    output_dir: 输出目录
    output_name: 输出文件名
    X_train, X_test: 原始特征数据（可选）
    """
    # 如果ouput_dir不存在，那么创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Clear the current figure
    plt.clf()
    
    # Create a new figure with a larger width to accommodate the metrics
    plt.figure(figsize=(8, 6))
    
    # Create subplot with adjusted position to leave space for metrics
    plt.subplot(111)
    
    plt.gca().set_aspect('equal', adjustable='box')
    
    # 计算偏差
    train_diff = np.abs(y_train - y_pred_train)
    test_diff = np.abs(y_test - y_pred_test)
    
    # 找出正常值和异常值的索引
    train_normal = train_diff < 30
    train_outlier = train_diff >= 30
    test_normal = test_diff < 30
    test_outlier = test_diff >= 30
    
    # 绘制正常值
    plt.scatter(y_train[train_normal], y_pred_train[train_normal], color='blue', label='Train')
    plt.scatter(y_test[test_normal], y_pred_test[test_normal], color='green', label='Test')
    
    # 合并并绘制异常值
    outlier_actual = np.concatenate([y_train[train_outlier], y_test[test_outlier]])
    outlier_pred = np.concatenate([y_pred_train[train_outlier], y_pred_test[test_outlier]])
    if len(outlier_actual) > 0:
        plt.scatter(outlier_actual, outlier_pred, color='red', label='Outliers')
        
        # 导出异常值数据
        outliers_data = []
        
        # 处理训练集异常值
        if np.any(train_outlier):
            train_outliers = {
                'Set': ['Train'] * sum(train_outlier),
                'Actual': y_train[train_outlier],
                'Predicted': y_pred_train[train_outlier],
                'Difference': train_diff[train_outlier]
            }
            # 如果提供了原始特征数据，添加特征信息
            if X_train is not None:
                for col in X_train.columns:
                    train_outliers[col] = X_train.loc[y_train[train_outlier].index, col].values
            outliers_data.append(pd.DataFrame(train_outliers))
        
        # 处理测试集异常值
        if np.any(test_outlier):
            test_outliers = {
                'Set': ['Test'] * sum(test_outlier),
                'Actual': y_test[test_outlier],
                'Predicted': y_pred_test[test_outlier],
                'Difference': test_diff[test_outlier]
            }
            # 如果提供了原始特征数据，添加特征信息
            if X_test is not None:
                for col in X_test.columns:
                    test_outliers[col] = X_test.loc[y_test[test_outlier].index, col].values
            outliers_data.append(pd.DataFrame(test_outliers))
        
        # 合并并保存异常值数据
        if outliers_data:
            outliers_df = pd.concat(outliers_data, axis=0)
            outliers_df = outliers_df.sort_values('Difference', ascending=False)  # 按偏差大小排序
            outliers_csv_path = output_dir + output_name.replace('.png', '_outliers.csv')
            outliers_df.to_csv(outliers_csv_path, index=True)
            
            # 打印异常值统计信息
            print(f"\n异常值统计 (偏差 >= 30):")
            print(f"总异常值数量: {len(outliers_df)}")
            print(f"训练集异常值: {sum(train_outlier)}")
            print(f"测试集异常值: {sum(test_outlier)}")
            print(f"异常值数据已保存至: {outliers_csv_path}\n")
    
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 
             color='red', label='Perfect Prediction')

    # 计算评估指标
    metrics = calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)
    
    # 添加标签和文本
    if mae_mean is not None:
        add_plot_labels(metrics, mae_mean,model_name,r2_loo)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_dir+output_name, dpi=300, format='png', bbox_inches='tight')
    
    # Close the figure to free memory
    plt.close()

def plot_r2_distribution(model, X, y,output_file):
    """绘制100次随机划分的R2分布"""
    r2_test_list = []
    mae_test_list = []
    scaler = StandardScaler()
    
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        model.fit(scaler.fit_transform(X_train), y_train)
        y_pred_test = model.predict(scaler.transform(X_test))
        r2_test_list.append(r2_score(y_test, y_pred_test))
        mae_test_list.append(mean_absolute_error(y_test, y_pred_test))

    plt.figure(figsize=(20, 3))
    plt.scatter(range(100), r2_test_list, s=100)
    plt.ylim(-1, 1)
    plt.yticks([-1, 0, 1])
    plt.xlabel('Random Test Set Index', fontsize=20, fontname='Arial')
    plt.ylabel(r'$R^2$', fontsize=20, fontname='Arial')
    
    params = model.get_params()
    title = fr'$R^2$ on 100 Random Test Sets (SVR: C={float(params["C"]):.4f}, ' \
            fr'epsilon={float(params["epsilon"]):.4f}, kernel={params["kernel"]}), ' \
            fr'$R^2_{{mean}}$: {float(np.mean(r2_test_list)):.4f}'
    
    plt.title(title, fontsize=20, fontname='Arial')
    plt.savefig(output_file, dpi=300, format='png', bbox_inches='tight')
    plt.close()

def calculate_metrics(y_train, y_pred_train, y_test, y_pred_test):
    """计算模型评估指标"""
    return {
        'r2_train': r2_score(y_train, y_pred_train),
        'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'r2_test': r2_score(y_test, y_pred_test),
        'mae_test': mean_absolute_error(y_test, y_pred_test)
    }

def add_plot_labels(metrics, mae_mean,model_name):


    """添加图表标签和文本"""
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted Values for {model_name}')
    plt.legend(loc='upper left')

    # 获取当前轴的范围
    ax = plt.gca()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    
    # 计算文本框的位置（在图的右侧）
    text_x = xmax + (xmax - xmin) * 0.05  # 在x轴最大值右侧留出5%的间距
    
    # 计算文本之间的垂直间距
    y_range = ymax - ymin
    y_spacing = y_range * 0.1  # 文本间距为y轴范围的10%
    
    # 添加评估指标文本
    metrics_text = [
        fr"$Pearson \; R_{{train}}: {np.sqrt(metrics['r2_train']):.4f}$",
        fr"$Pearson \; R_{{test}}: {np.sqrt(metrics['r2_test']):.4f}$",
        fr"$RMSE_{{test}}: {metrics['rmse_test']:.4f}$",
        fr"$MAE_{{test}}: {metrics['mae_test']:.4f}$",
        fr'$MAE_{{mean}}: {mae_mean:.4f}$'
    ]
    
    # 从上到下依次添加文本
    for i, text in enumerate(metrics_text):
        y_pos = ymin + i * y_spacing + y_range * 0.1
        plt.text(text_x, y_pos, text,
                fontsize=12, ha='left', va='top', fontname='Arial')
    
    # 调整图形布局以适应文本
    plt.subplots_adjust(right=0.85)  # 留出右侧空间显示文本 

def add_plot_labels_standard(metrics, mae_mean,model_name,r2_loo,fontsize=20,fontname='Times New Roman'):
    """添加图表标签和文本"""
    # plt.xlabel('Actual Values')
    # plt.ylabel('Predicted Values')
    # # plt.title(f'Actual vs Predicted Values for {model_name}',fontsize=fontsize+2)
    # plt.legend(loc='upper left',fontsize=fontsize - 5)

    plt.text(0.95, 0.05, fr"$Pearson \; R_{{train}}: {np.sqrt(metrics['r2_train']):.4f}$", fontsize=fontsize-5, ha='right', va='bottom', fontname='Arial', transform=plt.gca().transAxes)
    plt.text(0.95, 0.19, fr"$RMSE_{{test}}: {metrics['rmse_test']:.4f}$", fontsize=fontsize-5, ha='right', va='bottom', fontname='Arial', transform=plt.gca().transAxes)
    plt.text(0.95, 0.12, fr"$Pearson \; R_{{test}}: {np.sqrt(metrics['r2_test']):.4f}$", fontsize=fontsize-5, ha='right', va='bottom', fontname='Arial', transform=plt.gca().transAxes)        
    plt.text(0.95, 0.26, fr"$MAE_{{test}}: {metrics['mae_test']:.4f}$", fontsize=fontsize-5, ha='right', va='bottom', fontname='Arial', transform=plt.gca().transAxes)
    plt.text(0.95, 0.33, fr'$MAE_{{mean}}: {mae_mean:.4f}$', fontsize=fontsize-5, ha='right', va='bottom', fontname='Arial', transform=plt.gca().transAxes)
    plt.text(0.95, 0.4, fr'$R^2_{{LOO}}: {r2_loo:.4f}$', fontsize=fontsize-5, ha='right', va='bottom', fontname='Arial', transform=plt.gca().transAxes)

def plot_scatter_standard(y_train, y_pred_train, y_test, y_pred_test, model_name, mae_mean, output_dir, output_name, 
               X_train=None, X_test=None,r2_loo=None,fontsize=20,fontname='Times New Roman'):
    """
    绘制实际值vs预测值的散点图，不标记异常值
    
    Parameters:
    -----------
    y_train, y_pred_train: 训练集的实际值和预测值
    y_test, y_pred_test: 测试集的实际值和预测值
    model_name: 模型名称
    mae_mean: 平均MAE
    output_dir: 输出目录
    output_name: 输出文件名
    X_train, X_test: 原始特征数据（可选）
    fontsize: 字体大小
    fontname: 字体名称
    """

    fontsize = 20
    fontname = 'Times New Roman'
    plt.rcParams['font.family'] = fontname
    # 如果ouput_dir不存在，那么创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Clear the current figure
    plt.clf()
    
    # Create a new figure with a larger width to accommodate the metrics
    plt.figure(figsize=(8, 6))
    
    # Create subplot with adjusted position to leave space for metrics
    plt.subplot(111)
    
    plt.gca().set_aspect('equal', adjustable='box')
    
    # 绘制正常值
    plt.scatter(y_train, y_pred_train, color='blue', label='Train')
    plt.scatter(y_test, y_pred_test, color='green', label='Test')
    plt.xlabel('True Values/(%)', fontsize=fontsize, fontname=fontname)
    plt.ylabel('Predicted Values/(%)', fontsize=fontsize, fontname=fontname)
    # plt.title(f'Actual vs Predicted Values for {model_name}', fontsize=fontsize, fontname=fontname)
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 
             color='red', label='Perfect Prediction')
    plt.legend(loc='upper left', fontsize=fontsize-6)

    # 修改刻度的字体
    plt.xticks(fontsize=fontsize-3, fontname=fontname)
    plt.yticks(fontsize=fontsize-3, fontname=fontname)

    # 计算评估指标
    metrics = calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)
    
    # 添加标签和文本
    if mae_mean is not None:
        add_plot_labels_standard(metrics, mae_mean,model_name,r2_loo,fontsize,fontname)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_dir+output_name, dpi=300, format='png', bbox_inches='tight')
    
    # Close the figure to free memory
    plt.close()
