import pytest
from unittest.mock import MagicMock, patch
from kfp import dsl
from src.components import features, training, model_registraion, comparison, store
import pandas as pd
from unittest.mock import Mock
import random


@pytest.mark.parametrize(
    "project_id,training_table_name,time_col,is_log",
    [
        (
            "your-project-id1",
            "your-project.your_dataset.your_training_table",
            "DateTime2",
            False,
        ),
        (
            "your-project-id2",
            "your-project.your_dataset.your_training_table",
            "DateTime2",
            False,
        ),
    ],
)
@patch("google.cloud.bigquery.Client")
def test_fetch_dataset(
        mock_bq_client,
        project_id,
        training_table_name,
        time_col,
        is_log,
):
    mock_bq_client_instance = mock_bq_client.return_value
    mock_query_job = Mock()
    # Create a DataFrame with varying values for the 'Open' column
    mock_data = {'Date': pd.date_range(start="2024-01-01", periods=80, freq='D'), # must be >= 60 rata points
                 'Open': [random.randint(70, 110) for _ in range(80)]}
    mock_query_job.to_dataframe.return_value = pd.DataFrame(mock_data)
    mock_bq_client_instance.query.return_value = mock_query_job
    gcs_x_train, gcs_x_valid, gcs_train_set, gcs_valid_set, gcs_data = features.fetch_dataset.python_func(
        project_id=project_id,
        training_table_name=training_table_name,
        time_col=time_col,
        is_log=is_log,
    )
    assert isinstance(gcs_x_train, dsl.Dataset)
    assert isinstance(gcs_x_valid, dsl.Dataset)
    assert isinstance(gcs_train_set, dsl.Dataset)
    assert isinstance(gcs_valid_set, dsl.Dataset)
    assert isinstance(gcs_data, dsl.Dataset)

