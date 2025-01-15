import json
from typing import Dict
import uuid
from kfp import dsl
import pytest
import pandas as pd
import os
import yaml

dsl.get_uri = lambda suffix: '/tmp/'+suffix

class Helpers:
    @staticmethod
    def is_jsonable(v):
        try:
            json.dumps(v)
            return True
        except Exception:
            return False

@pytest.fixture
def helpers():
    return Helpers

@pytest.fixture
def mock_training_dataset() -> dsl.Dataset:
    input_data = {
        'feature_col_1': [3, 2, 1, 0, 57, 345, 76],
        'feature_col_2': ['a', 'b', 'c', 'jklasdjklf', 'as', 'lks', 'lks'],
        'target_feature_1': [7, 42, 63, 58928938923, 418, 23, 34],
        'target_feature_2': [1, 0, 0, 1, 1, 0, 1],
        'bean_dip': [1, 5, 3, 3, 3, 25, 25],
    }
    df = pd.DataFrame.from_dict(input_data)
    dataset = dsl.Dataset(uri='/tmp/'+str(uuid.uuid4())+'.parquet')
    df.to_parquet(dataset.path)
    return dataset


@pytest.fixture
def x_train_dataset():
    return dsl.Dataset(uri="./tests/components/artifacts/x_train.parquet")


@pytest.fixture
def x_valid_dataset():
    return dsl.Dataset(uri="./tests/components/artifacts/x_valid.parquet")


@pytest.fixture
def train_set_dataset():
    return dsl.Dataset(uri="./tests/components/artifacts/train_set.parquet")


@pytest.fixture
def valid_set_dataset():
    return dsl.Dataset(uri="./tests/components/artifacts/valid_set.parquet")


@pytest.fixture
def data_dataset():
    return dsl.Dataset(uri="./tests/components/artifacts/data.parquet")


@pytest.fixture
def ma_pred_dataset():
    return dsl.Dataset(uri="./tests/components/artifacts/ma_valid_pred_open.parquet")


@pytest.fixture
def arima_pred_dataset():
    return dsl.Dataset(uri="./tests/components/artifacts/arima_valid_pred_open.parquet")


@pytest.fixture
def lstm_pred_dataset():
    return dsl.Dataset(uri="./tests/components/artifacts/lstm_valid_pred_open.parquet")


@pytest.fixture
def metrics():
    return dsl.Model(uri="./tests/components/artifacts/metrics")


@pytest.fixture
def arima_model():
    return dsl.Model(uri="./tests/components/artifacts/arima_stock_model.joblib")


@pytest.fixture
def lstm_model():
    return dsl.Model(uri="./tests/components/artifacts/lstm_stock_model.keras")


@pytest.fixture(scope="session")
def test_settings_file(pytestconfig):
    """Loads settings.yml from ./tests/artifacts, or uses a default if not found."""
    settings_path = os.path.join(pytestconfig.rootpath, "tests", "artifacts", "settings.yml")
    try:
        with open(settings_path, 'r') as file:
            settings = yaml.safe_load(file)
            return settings
    except FileNotFoundError:
        print("No settings.yml found")
        return None

