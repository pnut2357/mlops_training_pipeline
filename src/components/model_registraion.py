from typing import NamedTuple, Dict
from kfp import dsl


@dsl.component(
    base_image="python:3.10",
    packages_to_install=[
        "google-cloud-bigquery==3.10.0",
        "google-cloud-storage==2.19.0",
        "db-dtypes==1.3.1",
        "scikit-learn==1.6.1",
        "pandas==2.2.3",
        "numpy==1.26.4",
        "tensorflow==2.18.0",
        "pmdarima==2.0.4"
    ],
)
def register_models(
        project_id: str,
        pipeline_name: str,
        model_registry_table_name: str,
        model_registry_bucket_name: str,
        model_versions: Dict,
        model_types: Dict,
        gcs_metrics: dsl.Artifact,
        gcs_arima_stock_model: dsl.Model,
        gcs_lstm_stock_model: dsl.Model,
) -> dsl.Dataset:
    from google.cloud import bigquery, storage
    from google.cloud.exceptions import NotFound
    from collections import defaultdict
    import numpy as np
    import pandas as pd
    import datetime
    import json
    import joblib
    import tensorflow as tf
    from pmdarima.arima import auto_arima
    def save_model_to_gcs(storage_client, gcs_bucket_name, gcs_model_path, local_model_path):
        """Uploads a model file to GCS."""
        bucket = storage_client.bucket(gcs_bucket_name)
        blob = bucket.blob(gcs_model_path)
        blob.upload_from_filename(local_model_path)
        print(f"Uploaded {local_model_path} to gs://{gcs_bucket_name}/{gcs_model_path}")
        return f"gs://{gcs_bucket_name}/{gcs_model_path}"
    def save_pandas2bigquery(
            bq_client,
            pdf,
            dataset_id,
            table_id,
            schema,
            time_partition_col=None,
            is_append=True,
    ):
        """Saves a Pandas DataFrame to a BigQuery table.
        Args:
            bq_client: BigQuery client.
            pdf: Pandas DataFrame containing data.
            dataset_id: BigQuery dataset ID.
            table_id: BigQuery table ID.
            schema: BigQuery table schema (list of SchemaField).
            time_partition_col: Name of the column for time partitioning (if any).
            is_append: If True, appends to the table; otherwise, overwrites.
        Returns:
            None
        """
        dataset_ref = bq_client.dataset(dataset_id)
        table_ref = dataset_ref.table(table_id)
        # Create dataset if it doesn't exist
        try:
            bq_client.get_dataset(dataset_ref)
            print(f"Dataset {dataset_id} already exists.")
        except NotFound:
            print(f"Dataset {dataset_id} does not exist. Creating it.")
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "US"
            bq_client.create_dataset(dataset)
            print(f"Dataset {dataset_id} created.")
        # Set write disposition
        write_disposition = (
            bigquery.WriteDisposition.WRITE_APPEND if is_append else bigquery.WriteDisposition.WRITE_TRUNCATE
        )
        # Configure load job
        job_config = bigquery.LoadJobConfig(
            schema=schema,
            write_disposition=write_disposition,
        )
        # Add time partitioning if specified
        if time_partition_col:
            job_config.time_partitioning = bigquery.TimePartitioning(field=time_partition_col)
        # Load data into BigQuery
        try:
            load_job = bq_client.load_table_from_dataframe(
                pdf,
                table_ref,
                job_config=job_config,
            )
            load_job.result()  # Wait for the job to complete
            print(f"Successfully loaded {load_job.output_rows} rows into {dataset_id}.{table_id}.")
        except Exception as e:
            print(f"Failed to load data into BigQuery: {e}")
            raise
    def register_models_and_metrics(storage_client, bigquery_client, pipeline_name, models_metrics, gcs_bucket_name, dataset_id, table_id):
        """Registers models to GCS and saves metrics to BigQuery."""
        # List to collect all records for BigQuery
        bq_records = []
        for model_info in models_metrics:
            model_name = model_info["model_name"]
            model_version = model_info["model_version"]
            model_type = model_info.get("model_type", None)
            metrics = model_info["metrics"]  # hashmap of metrics
            local_model_path = model_info["local_model_path"]
            # Generate GCS model path
            gcs_model_path = f"{pipeline_name}/models/{model_name}_{model_version}"
            if model_type.lower() == "none" or model_type is None or local_model_path == "none" or local_model_path is None:
                gcs_path = "none"
            else:
                gcs_path = save_model_to_gcs(storage_client, gcs_bucket_name, gcs_model_path, local_model_path)
            # Collect metrics
            metric_names = list(metrics.keys())
            metric_values = [str(metrics[m]) for m in metric_names]
            # Prepare a record for BigQuery
            bq_records.append({
                "model_name": model_name,
                "model_version": model_version,
                "time_stamp": datetime.datetime.utcnow(),
                "model_type": model_type,
                "metric_names": metric_names,
                "metric_values": metric_values,
                "gcs_path": gcs_path,
            })
        metrics_df = pd.DataFrame(bq_records)
        bq_schema = [
            bigquery.SchemaField("model_name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("model_version", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("time_stamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("model_type", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("metric_names", "STRING", mode="REPEATED"),
            bigquery.SchemaField("metric_values", "STRING", mode="REPEATED"),
            bigquery.SchemaField("gcs_path", "STRING", mode="NULLABLE"),
        ]
        # Save to BigQuery
        save_pandas2bigquery(
            bq_client=bigquery_client,
            pdf=metrics_df,
            dataset_id=dataset_id,
            table_id=table_id,
            schema=bq_schema,
            time_partition_col="time_stamp",
        )
        return metrics_df
    def process_metrics_to_models(metrics, model_versions, model_types, local_model_paths):
        """Processes a flat metrics dictionary into the models_metrics format.
        Args:
            metrics (dict): Dictionary of metrics with keys like "[model_name]_[metric_name]".
            model_versions (dict): Dictionary mapping model names to their versions.
            model_types (dict): Dictionary mapping model names to their types (optional).
            local_model_paths (dict): List of local model paths.
        Returns:
            list: A list of dictionaries in the models_metrics format.
        """
        models_metrics = defaultdict(lambda: {"metrics": {}})

        for idx, (metric_key, metric_value) in enumerate(metrics.items()):
            # Parse model name and metric name from key
            if "_" not in metric_key:
                continue  # Skip invalid keys
            model_name, metric_name = metric_key.split("_", 1)
            # Handle NaN values by converting them to None
            if isinstance(metric_value, float) and np.isnan(metric_value):
                metric_value = None
            # Populate model details
            models_metrics[model_name]["model_name"] = model_name
            models_metrics[model_name]["model_version"] = model_versions.get(model_name, "v1")
            models_metrics[model_name]["model_type"] = model_types.get(model_name, None) if model_types else None
            models_metrics[model_name]["metrics"][metric_name] = metric_value
            models_metrics[model_name]["local_model_path"] = local_model_paths.get(model_name, None)
        return [details for details in models_metrics.values()]
    # gcs_bucket_name = "ailabs-mlops-model-registry"
    _, dataset_id, table_id = model_registry_table_name.split(".")
    local_model_paths = defaultdict()
    joblib_model = joblib.load(gcs_arima_stock_model.path)
    keras_model = tf.keras.models.load_model(gcs_lstm_stock_model.path)
    for model_types_k, model_types_v in model_types.items():
        if model_types_v.lower() == "none":
            local_model_paths[model_types_k] = None
        model_version = model_versions[model_types_k]
        if model_types_v in ["joblib", "pkl"]:
            temp_local_model_path = f"{model_types_k}{model_version}.{model_types_v}"
            local_model_paths[model_types_k] = temp_local_model_path
            joblib.dump(joblib_model, temp_local_model_path)
        elif model_types_v in ["keras", "h5"]:
            temp_local_model_path = f"{model_types_k}{model_version}.{model_types_v}"
            local_model_paths[model_types_k] = temp_local_model_path
            keras_model.save(temp_local_model_path)
        print(model_types_k, model_version, local_model_paths[model_types_k])
    with open(gcs_metrics.path, "r") as f:
        metrics = json.load(f)
    bq_client = bigquery.Client(project=project_id)
    storage_client = storage.Client(project=project_id)
    models_metrics = process_metrics_to_models(metrics, model_versions, model_types, local_model_paths)
    metrics_df = register_models_and_metrics(storage_client, bq_client, pipeline_name, models_metrics,
                                             model_registry_bucket_name, dataset_id, table_id)
    # Return outputs
    gcs_metrics_df = dsl.Dataset(uri=dsl.get_uri(suffix="gcs_metrics_df.parquet"))
    metrics_df.to_parquet(gcs_metrics_df.path + ".gzip", index=False, compression="gzip")
    return gcs_metrics_df