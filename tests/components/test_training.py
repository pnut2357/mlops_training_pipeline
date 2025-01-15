import pytest
from unittest.mock import MagicMock, patch
from kfp import dsl
from src.components import training


@patch("google.cloud.storage.Client")
@patch("google.cloud.bigquery.Client")
def test_train_timeseries_models(
        mock_bq_client,
        mock_storage_client,
        tmp_path,
        train_set_dataset,
        valid_set_dataset,
        data_dataset,
):

    outputs = training.train_timeseries_models.python_func(
        gcs_train_set=train_set_dataset,
        gcs_valid_set=valid_set_dataset,
        gcs_data=data_dataset,
    )
    # Assertions
    assert isinstance(outputs.gcs_ma_valid_pred_open, dsl.Dataset)
    assert isinstance(outputs.gcs_arima_valid_pred_open, dsl.Dataset)
    assert isinstance(outputs.gcs_lstm_valid_pred_open, dsl.Dataset)
    assert isinstance(outputs.eval_metrics, dsl.Metrics)
    assert isinstance(outputs.gcs_metrics, dsl.Artifact)
    assert isinstance(outputs.gcs_arima_stock_model, dsl.Model)
    assert isinstance(outputs.gcs_lstm_stock_model, dsl.Model)


