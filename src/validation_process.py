import pandas as pd
import numpy as np

# 生成验证集
def validation_data_produce(data,H_feature_cols=None,B_feature_cols=None):
    """
    根据数据集生成验证集
    
    参数:
    data: 包含sub_H和sub_B组合的pandas DataFrame
    
    返回:
    完整组合空间的DataFrame，包含所有可能的sub_H和sub_B组合
    """
    import pandas as pd
    import numpy as np
    
    # 获取所有唯一的sub_H和sub_B值
    unique_sub_H = data['sub_H'].unique()
    unique_sub_B = data['sub_B'].unique()
    
    # 创建sub_H特征字典
    sub_H_features = {}
    # H_feature_cols = ['pka_H', 'dipole_H', 'homo_H', 'lumo_H', 'electronegativity_H', 
    #                  'Mulliken_charge_C_H', 'APT_charge_C_H', 'NPA_charge_C_H', 
    #                  'Mulliken_charge_H_H', 'APT_charge_H_H', 'NPA_charge_H_H', 
    #                  'NICS0', 'NICS1']


    for h in unique_sub_H:
        # 对于每个sub_H，获取其第一次出现时的特征值
        h_data = data[data['sub_H'] == h].iloc[0]
        sub_H_features[h] = {col: h_data[col] for col in H_feature_cols}
    
    # 创建sub_B特征字典
    sub_B_features = {}
    # B_feature_cols = ['pka_B', 'delta_G_B', 'delta_G_B_TS', 'dipole_B', 
    #                  'homo_B', 'lumo_B', 'electronegativity_B']

    for b in unique_sub_B:
        # 对于每个sub_B，获取其第一次出现时的特征值
        b_data = data[data['sub_B'] == b].iloc[0]
        sub_B_features[b] = {col: b_data[col] for col in B_feature_cols}
    
    # 生成所有可能的组合
    all_combinations = []
    
    for h in unique_sub_H:
        for b in unique_sub_B:
            # 创建新行
            new_row = {'sub_H': h, 'sub_B': b}
            
            # 添加sub_H特征
            new_row.update(sub_H_features[h])
            
            # 添加sub_B特征
            new_row.update(sub_B_features[b])
            
            # 查找原始数据中是否存在此组合
            existing = data[(data['sub_H'] == h) & (data['sub_B'] == b)]
            
            if len(existing) > 0:
                # 如果存在，使用原始yield值
                new_row['yield'] = existing.iloc[0]['yield']
                # 如果有new_order列，也一并保留
                if 'new_order' in data.columns:
                    new_row['new_order'] = existing.iloc[0]['new_order']
            else:
                # 如果不存在，yield设为空
                new_row['yield'] = np.nan
                # 如果有new_order列，也设为空
                if 'new_order' in data.columns:
                    new_row['new_order'] = np.nan
            
            all_combinations.append(new_row)
    
    # 创建新的DataFrame
    validation_data = pd.DataFrame(all_combinations)
    
    # 确保列的顺序与原始数据相同
    if 'new_order' in data.columns:
        column_order = ['sub_H', 'new_order', 'sub_B'] + H_feature_cols + B_feature_cols + ['yield']
    else:
        column_order = ['sub_H', 'sub_B'] + H_feature_cols + B_feature_cols + ['yield']
    
    validation_data = validation_data[column_order]
    
    return validation_data

data = pd.read_csv('/home/ldj/Code/Projects/Stable-Organoboronates-ML-Supp/data_round2_all.csv')
H_feature_cols = ['Mulliken_charge_H_H', 'APT_charge_H_H', 'NICS1']
B_feature_cols = ['pka_B','delta_G_B']
validation_data = validation_data_produce(data, H_feature_cols, B_feature_cols)
validation_data.to_csv('validation_data.csv', index=False)



# 模型绘制
