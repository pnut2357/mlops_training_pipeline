import pytest
from unittest.mock import MagicMock, patch
from kfp import dsl
from src.components import store


@pytest.mark.parametrize(
    "project_id,save_pred_table_name,time_col",
    [
        (
            "your-project-id1",
            "your-project-id1.your_dataset1.your_saving_training_table1",
            "Date",
        ),
        (
            "your-project-id2",
            "your-project-id2.your_dataset2.your_saving_training_table2",
            "Date",
        ),
    ],
)
@patch("google.cloud.storage.Client")
@patch("google.cloud.bigquery.Client")
def test_train_timeseries_models(
        mock_bq_client,
        mock_storage_client,
        project_id,
        save_pred_table_name,
        time_col,
        x_valid_dataset,
        valid_set_dataset,
        ma_pred_dataset,
        arima_pred_dataset,
        lstm_pred_dataset
):
    pred_data = store.store_pred.python_func(
        project_id=project_id,
        save_pred_table_name=save_pred_table_name,
        time_col=time_col,
        gcs_x_valid=x_valid_dataset,
        gcs_valid_set=valid_set_dataset,
        gcs_ma_valid_pred_open=ma_pred_dataset,
        gcs_arima_valid_pred_open=arima_pred_dataset,
        gcs_lstm_valid_pred_open=lstm_pred_dataset
    )
    # Assertions
    assert isinstance(pred_data, dsl.Dataset)