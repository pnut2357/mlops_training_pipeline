from typing import NamedTuple
from kfp import dsl


@dsl.component(
    base_image="python:3.10",
    packages_to_install=[
        "google-cloud-bigquery==3.10.0",
        "db-dtypes==1.3.1",
        "plotly==5.24.1",
        "pandas==2.2.3",
        "numpy==1.26.4",
        "matplotlib==3.9.4",
    ],
)
def compare_models(
        gcs_x_train: dsl.Dataset,
        gcs_x_valid: dsl.Dataset,
        gcs_train_set: dsl.Dataset,
        gcs_valid_set: dsl.Dataset,
        gcs_ma_valid_pred_open: dsl.Dataset,
        gcs_arima_valid_pred_open: dsl.Dataset,
        gcs_lstm_valid_pred_open: dsl.Dataset,
        gcs_metrics: dsl.Artifact,
) -> dsl.Artifact:
    from google.cloud import bigquery
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import json
    def plot_predictions(x_train, x_valid, train_data, actual_data, predicted_data_list, predicted_labels, rms_values,
                         title, xlabel, ylabel):
        """Plots training data, actual validation data, and multiple predicted validation data series with RMS values.
        Args:
            x_train: Array-like, representing the x-axis data for the training set.
            x_val: Array-like, representing the x-axis data for the validation set.
            train_data: Array-like, representing the y-axis data for the training set.
            actual_data: Array-like, representing the y-axis data for the actual values in the validation set.
            predicted_data_list: List of Array-like, representing the y-axis data for the predicted values.
            predicted_labels: List of str, labels for each predicted series.
            rms_values: List of float, RMS values corresponding to each prediction series.
            title: str, the title of the plot.
            xlabel: str, the label for the x-axis.
            ylabel: str, the label for the y-axis.
        Returns:
            None. Displays the plot using matplotlib.pyplot.show().
        """
        plt.figure(figsize=(10, 6))
        plt.plot(x_train, train_data, label="Model Training Data")
        plt.plot(x_valid, actual_data, label="Actual Data")
        for predicted_data, label, rms in zip(predicted_data_list, predicted_labels, rms_values):
            plt.plot(x_valid, predicted_data, label=f"{label} (RMS: {rms:.2f})")
        # Add labels and legend
        plt.xlabel(xlabel, size=12)
        plt.ylabel(ylabel, size=12)
        plt.title(title, size=14)
        plt.legend()
        return plt
    x_train = pd.read_parquet(gcs_x_train.path+".gzip")
    x_valid = pd.read_parquet(gcs_x_valid.path+".gzip")
    train_set = pd.read_parquet(gcs_train_set.path+".gzip")
    valid_set = pd.read_parquet(gcs_valid_set.path+".gzip")
    ma_valid_pred_open = pd.read_parquet(gcs_ma_valid_pred_open.path+".gzip")
    arima_valid_pred_open = pd.read_parquet(gcs_arima_valid_pred_open.path+".gzip")
    lstm_valid_pred_open = pd.read_parquet(gcs_lstm_valid_pred_open.path + ".gzip")
    with open(gcs_metrics.path, "r") as f:
        metrics = json.load(f)
    comparison_fig = plot_predictions(
        x_train=x_train,
        x_valid=x_valid,
        train_data=train_set,
        actual_data=valid_set,
        predicted_data_list=[ma_valid_pred_open, arima_valid_pred_open, lstm_valid_pred_open],
        predicted_labels=["Moving Averages Prediction", "ARIMA Prediction", "LSTM Prediction"],
        rms_values=[metrics["ma_rms"], metrics["arima_rms"], metrics["lstm_rms"]],
        title="Stock Open Price Prediction Comparison with RMS Values",
        xlabel="Time",
        ylabel="Open Price"
    )
    gcs_comparison = dsl.Artifact(uri=dsl.get_uri(suffix="gcs_comparison.png"))
    comparison_fig.savefig(gcs_comparison.path)
    return gcs_comparison
