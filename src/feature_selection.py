import logging
logger = logging.getLogger(__name__)

def feature_selection(X, feature_importance_percent, corr_matrix, removed_features):
    """
    Select features based on importance and correlation.
    优先处理高相关性特征，在高相关性特征组中选择重要性最低的特征删除。
    如果没有高相关性特征，则考虑删除重要性最低的特征。

    Parameters:
    X (DataFrame): The feature data.
    feature_importance_percent (array): Feature importance percentages.
    corr_matrix (DataFrame): Correlation matrix of features.
    removed_features (list): List of already removed features.

    Returns:
    list: A list containing the feature to be removed (if any).
    """
    # 初始化候选删除特征列表
    high_corr_candidates = []
    low_importance_candidates = []
    
    # 首先检查高相关性特征对
    for i in range(len(X.columns)):
        feature = X.columns[i]
        if feature in removed_features:
            continue
            
        for j in range(i + 1, len(X.columns)):
            if corr_matrix.iloc[i, j] > 0.8:  # 相关性高于0.8的特征对
                other_feature = X.columns[j]
                if other_feature not in removed_features:
                    # 选择重要性较低的特征作为候选
                    if feature_importance_percent[i] < feature_importance_percent[j]:
                        high_corr_candidates.append({
                            'feature': feature,
                            'importance': feature_importance_percent[i],
                            'correlation': corr_matrix.iloc[i, j],
                            'correlated_with': other_feature
                        })
                    else:
                        high_corr_candidates.append({
                            'feature': other_feature,
                            'importance': feature_importance_percent[j],
                            'correlation': corr_matrix.iloc[i, j],
                            'correlated_with': feature
                        })
    
    # 如果存在高相关性特征，从中选择重要性最低的
    if high_corr_candidates:
        # 按重要性排序
        high_corr_candidates.sort(key=lambda x: x['importance'])
        selected = high_corr_candidates[0]
        logger.info(f"发现高相关性特征组:")
        logger.info(f"  - 特征: {selected['feature']}")
        logger.info(f"  - 重要性: {selected['importance']:.2f}%")
        logger.info(f"  - 与特征 {selected['correlated_with']} 的相关性: {selected['correlation']:.2f}")
        return [selected['feature']]
    
    # 如果没有高相关性特征，检查低重要性特征
    logger.info("未发现高相关性特征，检查低重要性特征")
    for i, importance in enumerate(feature_importance_percent):
        if importance < 5:  # 重要性低于5%的特征
            feature = X.columns[i]
            if feature not in removed_features:
                low_importance_candidates.append({
                    'feature': feature,
                    'importance': importance,
                    'max_correlation': corr_matrix.iloc[i].max()
                })
    
    if low_importance_candidates:
        # 按重要性排序
        low_importance_candidates.sort(key=lambda x: x['importance'])
        selected = low_importance_candidates[0]
        logger.info(f"选择低重要性特征:")
        logger.info(f"  - 特征: {selected['feature']}")
        logger.info(f"  - 重要性: {selected['importance']:.2f}%")
        logger.info(f"  - 最大相关性: {selected['max_correlation']:.2f}")
        return [selected['feature']]
    
    return []
