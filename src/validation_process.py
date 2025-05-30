import pandas as pd
import numpy as np

# Generate validation dataset
def validation_data_produce(data,H_feature_cols=None,B_feature_cols=None):
    """
    Generate a validation dataset based on the input data
    
    Parameters:
    data: pandas DataFrame containing sub_H and sub_B combinations
    
    Returns:
    A DataFrame with complete combination space, including all possible sub_H and sub_B combinations
    """
    
    # Get all unique sub_H and sub_B values
    unique_sub_H = data['sub_H'].unique()
    unique_sub_B = data['sub_B'].unique()
    
    # Create sub_H features dictionary
    sub_H_features = {}
    # H_feature_cols = ['pka_H', 'dipole_H', 'homo_H', 'lumo_H', 'electronegativity_H', 
    #                  'Mulliken_charge_C_H', 'APT_charge_C_H', 'NPA_charge_C_H', 
    #                  'Mulliken_charge_H_H', 'APT_charge_H_H', 'NPA_charge_H_H', 
    #                  'NICS0', 'NICS1']


    for h in unique_sub_H:
        # For each sub_H, get its feature values from its first occurrence
        h_data = data[data['sub_H'] == h].iloc[0]
        sub_H_features[h] = {col: h_data[col] for col in H_feature_cols}
    
    # Create sub_B features dictionary
    sub_B_features = {}
    # B_feature_cols = ['pka_B', 'delta_G_B', 'delta_G_B_TS', 'dipole_B', 
    #                  'homo_B', 'lumo_B', 'electronegativity_B']

    for b in unique_sub_B:
        # For each sub_B, get its feature values from its first occurrence
        b_data = data[data['sub_B'] == b].iloc[0]
        sub_B_features[b] = {col: b_data[col] for col in B_feature_cols}
    
    # Generate all possible combinations
    all_combinations = []
    
    for h in unique_sub_H:
        for b in unique_sub_B:
            # Create new row
            new_row = {'sub_H': h, 'sub_B': b}
            
            # Add sub_H features
            new_row.update(sub_H_features[h])
            
            # Add sub_B features
            new_row.update(sub_B_features[b])
            
            # Check if this combination exists in the original data
            existing = data[(data['sub_H'] == h) & (data['sub_B'] == b)]
            
            if len(existing) > 0:
                # If exists, use the original yield value
                new_row['yield'] = existing.iloc[0]['yield']
                # If new_order column exists, keep it as well
                if 'new_order' in data.columns:
                    new_row['new_order'] = existing.iloc[0]['new_order']
            else:
                # If doesn't exist, set yield to empty
                new_row['yield'] = np.nan
                # If new_order column exists, set it to empty as well
                if 'new_order' in data.columns:
                    new_row['new_order'] = np.nan
            
            all_combinations.append(new_row)
    
    # Create new DataFrame
    validation_data = pd.DataFrame(all_combinations)
    
    # Ensure column order matches the original data
    if 'new_order' in data.columns:
        column_order = ['sub_H', 'new_order', 'sub_B'] + H_feature_cols + B_feature_cols + ['yield']
    else:
        column_order = ['sub_H', 'sub_B'] + H_feature_cols + B_feature_cols + ['yield']
    
    validation_data = validation_data[column_order]
    
    return validation_data