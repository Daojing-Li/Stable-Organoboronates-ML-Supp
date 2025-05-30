import logging
logger = logging.getLogger(__name__)

def feature_selection(X, feature_importance_percent, corr_matrix, removed_features):
    """
    Select features based on importance and correlation.
    Prioritize processing high-correlation features, selecting the least important feature for removal in high-correlation groups.
    If there are no high-correlation features, consider removing the least important feature.

    Parameters:
    X (DataFrame): The feature data.
    feature_importance_percent (array): Feature importance percentages.
    corr_matrix (DataFrame): Correlation matrix of features.
    removed_features (list): List of already removed features.

    Returns:
    list: A list containing the feature to be removed (if any).
    """
    # Initialize candidate features for removal
    high_corr_candidates = []
    low_importance_candidates = []
    
    # First check for high-correlation feature pairs
    for i in range(len(X.columns)):
        feature = X.columns[i]
        if feature in removed_features:
            continue
            
        for j in range(i + 1, len(X.columns)):
            if corr_matrix.iloc[i, j] > 0.8:  # Feature pairs with correlation above 0.8
                other_feature = X.columns[j]
                if other_feature not in removed_features:
                    # Select the less important feature as a candidate
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
    
    # If high-correlation features exist, select the one with lowest importance
    if high_corr_candidates:
        # Sort by importance
        high_corr_candidates.sort(key=lambda x: x['importance'])
        selected = high_corr_candidates[0]
        logger.info(f"Found high-correlation feature group:")
        logger.info(f"  - Feature: {selected['feature']}")
        logger.info(f"  - Importance: {selected['importance']:.2f}%")
        logger.info(f"  - Correlation with {selected['correlated_with']}: {selected['correlation']:.2f}")
        return [selected['feature']]
    
    # If no high-correlation features, check for low importance features
    logger.info("No high-correlation features found, checking low importance features")
    for i, importance in enumerate(feature_importance_percent):
        if importance < 5:  # Features with importance below 5%
            feature = X.columns[i]
            if feature not in removed_features:
                low_importance_candidates.append({
                    'feature': feature,
                    'importance': importance,
                    'max_correlation': corr_matrix.iloc[i].max()
                })
    
    if low_importance_candidates:
        # Sort by importance
        low_importance_candidates.sort(key=lambda x: x['importance'])
        selected = low_importance_candidates[0]
        logger.info(f"Selected low importance feature:")
        logger.info(f"  - Feature: {selected['feature']}")
        logger.info(f"  - Importance: {selected['importance']:.2f}%")
        logger.info(f"  - Max correlation: {selected['max_correlation']:.2f}")
        return [selected['feature']]
    
    return []
