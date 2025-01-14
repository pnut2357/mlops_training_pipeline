from typing import Dict, NamedTuple
import kfp
from src.components import features, training, comparison, model_registraion, store
from utils.get_config import load_config
from google.cloud import aiplatform
import datetime
from google.oauth2 import service_account


@kfp.dsl.pipeline(
    name='stock_train_pipeline',
)
def pipeline(
        project_id: str,
        pipeline_name: str,
        training_table_name: str,
        time_col: str,
        is_log: bool,
        model_registry_table_name: str,
        model_registry_bucket_name: str,
        model_versions: Dict,
        model_types: Dict,
        save_pred_table_name: str
):
    data = (
        features.fetch_dataset(
            project_id=project_id,
            training_table_name=training_table_name,
            time_col=time_col,
            is_log=is_log,
        ).set_display_name("fetch_dataset")
         .set_cpu_limit('4')
         .set_memory_limit('16G')
    )
    train = (
        training.train_timeseries_models(
            gcs_train_set=data.outputs["gcs_train_set"],
            gcs_valid_set=data.outputs["gcs_valid_set"],
            gcs_data=data.outputs["gcs_data"]
        ).set_display_name("train_timeseries_models")
         .set_cpu_limit('4')
         .set_memory_limit('16G')
    )
    compare = (
        comparison.compare_models(
            gcs_x_train=data.outputs["gcs_x_train"],
            gcs_x_valid=data.outputs["gcs_x_valid"],
            gcs_train_set=data.outputs["gcs_train_set"],
            gcs_valid_set=data.outputs["gcs_valid_set"],
            gcs_ma_valid_pred_open=train.outputs["gcs_ma_valid_pred_open"],
            gcs_arima_valid_pred_open=train.outputs["gcs_arima_valid_pred_open"],
            gcs_lstm_valid_pred_open=train.outputs["gcs_lstm_valid_pred_open"],
            gcs_metrics=train.outputs["gcs_metrics"],
        ).set_display_name("compare_models")
         .set_cpu_limit('4')
         .set_memory_limit('4G')
    )
    model_registry = (
        model_registraion.register_models(
            project_id=project_id,
            pipeline_name=pipeline_name,
            model_registry_table_name=model_registry_table_name,
            model_registry_bucket_name=model_registry_bucket_name,
            model_versions=model_versions,
            model_types=model_types,
            gcs_metrics=train.outputs["gcs_metrics"],
            gcs_arima_stock_model=train.outputs["gcs_arima_stock_model"],
            gcs_lstm_stock_model=train.outputs["gcs_lstm_stock_model"],
        ).set_display_name("register_models")
    )
    store_pred = (
        store.store_pred(
            project_id=project_id,
            save_pred_table_name=save_pred_table_name,
            time_col=time_col,
            gcs_x_valid=data.outputs["gcs_x_valid"],
            gcs_valid_set=data.outputs["gcs_valid_set"],
            gcs_ma_valid_pred_open=train.outputs["gcs_ma_valid_pred_open"],
            gcs_arima_valid_pred_open=train.outputs["gcs_arima_valid_pred_open"],
            gcs_lstm_valid_pred_open=train.outputs["gcs_lstm_valid_pred_open"],
        )
    )



if __name__ == "__main__":
    env = "local-dev"
    config = load_config("./config.yml")["envs"][env]
    config_feature_metadata = config["FEATURE_METADATA"]
    config_model_registry_metadata = config["MODEL_REGISTRY_METADATA"]
    aiplatform.init(
        project=config["PROJECT_ID"],
        location=config["LOCATION"],
        credentials=service_account.Credentials.from_service_account_file(config["SA_PATH"])
    )
    TMP_PIPELINE_JSON = f"/tmp/{config['PIPELINE_NAME']}.json"
    arguments = dict(
        project_id=config["PROJECT_ID"],
        pipeline_name=config["PIPELINE_NAME"],
        training_table_name=config_feature_metadata["training_table_name"],
        time_col=config_feature_metadata["time_col"],
        is_log=config_feature_metadata["is_log"],
        model_registry_table_name=config_model_registry_metadata["model_registry_table_name"],
        model_registry_bucket_name=config_model_registry_metadata["model_registry_bucket_name"],
        model_versions=config_model_registry_metadata["model_versions"],
        model_types=config_model_registry_metadata["model_types"],
        save_pred_table_name=config["SAVE_PRED_TABLE_NAME"],
    )
    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline,
        pipeline_parameters=arguments,
        package_path=TMP_PIPELINE_JSON,
    )
    pipeline_job = aiplatform.PipelineJob(
        display_name=config['PIPELINE_NAME']+'-'
        + datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
        project=config["PROJECT_ID"],
        parameter_values=arguments,
        template_path=TMP_PIPELINE_JSON,
        enable_caching=False,
    )
    pipeline_job.submit(
        service_account=config["SERVICE_ACCOUNT"],
    )