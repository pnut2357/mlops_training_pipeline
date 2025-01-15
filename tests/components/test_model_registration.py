import pytest
from unittest.mock import patch
from kfp import dsl
from src.components import model_registraion


@pytest.mark.parametrize(
    "project_id,pipeline_name,model_registry_table_name,model_registry_bucket_name,model_versions,model_types",
    [
        (
            "your-project-id1",
            "your-pipeline-name1",
            "your-project.your_dataset.your_training_table1",
            "gcs-bucket-name1",
            {'ma': 'v1', 'arima': 'v2', 'lstm': 'v2'},
            {'ma': 'none', 'arima': 'joblib', 'lstm': 'keras'},
        ),
        (
            "your-project-id2",
            "your-pipeline-name2",
            "your-project.your_dataset.your_training_table2",
            "gcs-bucket-name2",
            {'ma': 'v5', 'arima': 'v7', 'lstm': 'v3'},
            {'ma': 'none', 'arima': 'joblib', 'lstm': 'keras'},
        ),
    ],
)
@patch("google.cloud.storage.Client")
@patch("google.cloud.bigquery.Client")
def test_register_models(
        mock_bq_client,
        mock_storage_client,
        project_id,
        pipeline_name,
        model_registry_table_name,
        model_registry_bucket_name,
        model_versions,
        model_types,
        metrics,
        arima_model,
        lstm_model
):
    metrics = model_registraion.register_models.python_func(
        project_id=project_id,
        pipeline_name=pipeline_name,
        model_registry_table_name=model_registry_table_name,
        model_registry_bucket_name=model_registry_bucket_name,
        model_versions=model_versions,
        model_types=model_types,
        gcs_metrics=metrics,
        gcs_arima_stock_model=arima_model,
        gcs_lstm_stock_model=lstm_model,
    )
    # Assertions
    assert isinstance(metrics, dsl.Dataset)

