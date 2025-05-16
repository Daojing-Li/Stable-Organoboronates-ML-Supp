import seaborn as sns

import matplotlib.pyplot as plt


def feature_correlation_analysis(X_model, mae_mean, mae_threshold):
    """
    Analyze the feature correlation and plot the correlation matrix.

    Parameters:
    X_model (DataFrame): The feature data.

    Returns:
    corr_matrix (DataFrame): The correlation matrix.
    """
    corr_matrix = X_model.corr().abs()
    if mae_mean < mae_threshold:
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Feature Correlation Matrix")
        plt.show()
    return corr_matrix
