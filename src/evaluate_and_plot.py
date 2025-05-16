import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from IPython.display import display

import matplotlib.pyplot as plt


def evaluate_and_plot(
    model_name,
    y_train,
    y_pred_train,
    y_test,
    y_pred_test,
    X_train,
    X_test,
    threshold=30,
    mae_threshold=20,
    mae_mean=None,
    filename=None,
):
    # Calculate performance metrics
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)

    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    if mae_mean < mae_threshold:
        plt.scatter(y_train, y_pred_train, color="blue", label="Train")
        plt.scatter(y_test, y_pred_test, color="green", label="Test")
        plt.plot(
            [min(y_train), max(y_train)],
            [min(y_train), max(y_train)],
            color="red",
            label="Perfect Prediction",
        )

        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Actual vs Predicted Values for {model_name}")
        plt.legend()

        # Display performance metrics on the plot
        plt.gca().set_aspect("equal", adjustable="box")
        plt.text(
            100,
            0,
            rf"$Pearson \; R_{{train}}: {np.sqrt(r2_train):.4f}$",
            fontsize=12,
            ha="right",
            va="bottom",
            fontname="Arial",
        )
        plt.text(
            100,
            14,
            rf"$RMSE_{{test}}: {rmse_test:.4f}$",
            fontsize=12,
            ha="right",
            va="bottom",
            fontname="Arial",
        )
        plt.text(
            100,
            7,
            rf"$Pearson \; R_{{test}}: {np.sqrt(r2_test):.4f}$",
            fontsize=12,
            ha="right",
            va="bottom",
            fontname="Arial",
        )
        plt.text(
            100,
            21,
            rf"$MAE_{{test}}: {mae_test:.4f}$",
            fontsize=12,
            ha="right",
            va="bottom",
            fontname="Arial",
        )
        plt.text(
            100,
            28,
            rf"$MAE_{{mean}}: {mae_mean:.4f}$",
            fontsize=12,
            ha="right",
            va="bottom",
            fontname="Arial",
        )
        # plt.show()
        # plt.savefig(f"/home/ldj/Code/Projects/Stable-Organoboronates-ML-Supp/output/{model_name}_{filename}.png")


def evaluate_and_plot_with_abnormal_points(
    model_name,
    y_train,
    y_pred_train,
    y_test,
    y_pred_test,
    X_train,
    X_test,
    threshold=30,
    mae_threshold=20,
    mae_mean=None,
):
    """
    Evaluate the performance of a model and plot the actual vs. predicted values.

    Parameters:
    model_name (str): Name of the model.
    y_train (array-like): Actual target values for the training set.
    y_pred_train (array-like): Predicted target values for the training set.
    y_test (array-like): Actual target values for the test set.
    y_pred_test (array-like): Predicted target values for the test set.
    X_train (DataFrame): Features for the training set. Defaults to None.
    X_test (DataFrame): Features for the test set. Defaults to None.
    threshold (float, optional): Threshold for identifying anomalies. Defaults to 30.

    Returns:
    None
    """
    # Calculate performance metrics
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)

    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    if mae_mean < mae_threshold:
        # Identify anomalies based on the threshold
        anomalies_train = np.abs(y_pred_train - y_train) > threshold
        anomalies_test = np.abs(y_pred_test - y_test) > threshold

        anomaly_indices_train = np.where(anomalies_train)[0]
        anomaly_indices_test = np.where(anomalies_test)[0]

        # Create DataFrames to store anomaly details
        results_train_df = pd.DataFrame(
            columns=["Actual", "Predicted"] + list(X_train.columns)
        )
        results_test_df = pd.DataFrame(
            columns=["Actual", "Predicted"] + list(X_train.columns)
        )

        # Populate DataFrames with anomaly details
        for idx in anomaly_indices_train:
            new_row = pd.DataFrame(
                {
                    "Actual": [y_train.iloc[idx]],
                    "Predicted": [y_pred_train[idx]],
                    **{col: [X_train.iloc[idx][col]] for col in X_train.columns},
                }
            )
            results_train_df = pd.concat([results_train_df, new_row], ignore_index=True)

        for idx in anomaly_indices_test:
            new_row = pd.DataFrame(
                {
                    "Actual": [y_test.iloc[idx]],
                    "Predicted": [y_pred_test[idx]],
                    **{col: [X_test.iloc[idx][col]] for col in X_test.columns},
                }
            )
            results_test_df = pd.concat([results_test_df, new_row], ignore_index=True)

        # Display anomaly details
        display(results_train_df)
        display(results_test_df)

        # Plot actual vs. predicted values
        # plt.figure(figsize=(10, 6))
        plt.gca().set_aspect("equal", adjustable="box")
        plt.scatter(y_train, y_pred_train, color="blue", alpha=0.6, label="Train")
        plt.scatter(y_test, y_pred_test, color="green", alpha=0.6, label="Test")
        plt.scatter(
            y_train[anomalies_train],
            y_pred_train[anomalies_train],
            color="orange",
            label="Anomalies Train",
        )
        plt.scatter(
            y_test[anomalies_test],
            y_pred_test[anomalies_test],
            color="red",
            label="Anomalies Test",
        )
        plt.plot(
            [min(y_train), max(y_train)],
            [min(y_train), max(y_train)],
            color="red",
            linewidth=2,
        )  # Draw diagonal line

        # Annotate anomalies on the plot
        for i in range(len(results_train_df)):
            plt.annotate(
                f"{i}",
                (results_train_df.iloc[i, 0], results_train_df.iloc[i, 1]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )
        for i in range(len(results_test_df)):
            plt.annotate(
                f"{i}",
                (results_test_df.iloc[i, 0], results_test_df.iloc[i, 1]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Actual vs Predicted Values for {model_name}")
        plt.legend()

        # Display performance metrics on the plot
        # plt.gca().text(1.05, 0, f'Train RMSE: {rmse_train:.4f}\nTrain R^2: {r2_train:.4f}\nTrain MAE: {mae_train:.4f}\n\nTest RMSE: {rmse_test:.4f}\nTest R^2: {r2_test:.4f}\nTest MAE: {mae_test:.4f}',
        #             transform=plt.gca().transAxes, fontsize=10, verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.text(
            100,
            -5,
            f"Train RMSE: {rmse_train:.4f}",
            fontsize=12,
            ha="right",
            va="bottom",
            fontname="Arial",
        )
        plt.text(
            100,
            0,
            rf"Train $R^2$: {r2_train:.4f}",
            fontsize=12,
            ha="right",
            va="bottom",
            fontname="Arial",
        )
        plt.text(
            100,
            5,
            f"Train MAE: {mae_train:.4f}",
            fontsize=12,
            ha="right",
            va="bottom",
            fontname="Arial",
        )
        plt.text(
            100,
            10,
            f"Test RMSE: {rmse_test:.4f}",
            fontsize=12,
            ha="right",
            va="bottom",
            fontname="Arial",
        )
        plt.text(
            100,
            15,
            rf"Test $R^2$: {r2_test:.4f}",
            fontsize=12,
            ha="right",
            va="bottom",
            fontname="Arial",
        )
        plt.text(
            100,
            20,
            f"Test MAE: {mae_test:.4f}",
            fontsize=12,
            ha="right",
            va="bottom",
            fontname="Arial",
        )

        plt.show()
