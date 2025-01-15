import pytest
from unittest.mock import MagicMock, patch
from kfp import dsl
from src.components import comparison


@patch("google.cloud.storage.Client")
@patch("google.cloud.bigquery.Client")
def test_register_models(
        mock_bq_client,
        mock_storage_client,
        x_train_dataset,
        x_valid_dataset,
        train_set_dataset,
        valid_set_dataset,
        ma_pred_dataset,
        arima_pred_dataset,
        lstm_pred_dataset,
        metrics,
):
    compasiton_fig = comparison.compare_models.python_func(
        gcs_x_train=x_train_dataset,
        gcs_x_valid=x_valid_dataset,
        gcs_train_set=train_set_dataset,
        gcs_valid_set=valid_set_dataset,
        gcs_ma_valid_pred_open=ma_pred_dataset,
        gcs_arima_valid_pred_open=arima_pred_dataset,
        gcs_lstm_valid_pred_open=lstm_pred_dataset,
        gcs_metrics=metrics
    )
    # Assertions
    assert isinstance(compasiton_fig, dsl.Artifact)
