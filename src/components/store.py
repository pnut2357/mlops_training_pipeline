# # Filter out NaN values
# valid_metrics = {key: value for key, value in metrics.items() if not (isinstance(value, float) and np.isnan(value))}
#
# # Find the model with the lowest RMS
# best_model = min(valid_metrics, key=valid_metrics.get)
# best_rms = valid_metrics[best_model]

from typing import NamedTuple, Dict
from kfp import dsl


@dsl.component(
    base_image="python:3.10",
    packages_to_install=[
        "google-cloud-bigquery==3.10.0",
        "db-dtypes==1.3.1",
        "pandas==2.2.3",
        "numpy==1.26.4",
    ],
)
def store_pred(
        project_id: str,
        save_pred_table_name: str,
        time_col: str,
        gcs_x_valid: dsl.Dataset,
        gcs_valid_set: dsl.Dataset,
        gcs_ma_valid_pred_open: dsl.Dataset,
        gcs_arima_valid_pred_open: dsl.Dataset,
        gcs_lstm_valid_pred_open: dsl.Dataset,
) -> dsl.Dataset:
    from google.cloud import bigquery
    from google.cloud.exceptions import NotFound
    import pandas as pd
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
    _, dataset_id, table_id = save_pred_table_name.split('.')
    x_valid = pd.read_parquet(gcs_x_valid.path + ".gzip")
    valid_set = pd.read_parquet(gcs_valid_set.path + ".gzip")
    ma_valid_pred_open = pd.read_parquet(gcs_ma_valid_pred_open.path + ".gzip")
    arima_valid_pred_open = pd.read_parquet(gcs_arima_valid_pred_open.path + ".gzip")
    lstm_valid_pred_open = pd.read_parquet(gcs_lstm_valid_pred_open.path + ".gzip")
    ##
    pred_data = pd.concat([x_valid, valid_set, ma_valid_pred_open, arima_valid_pred_open, lstm_valid_pred_open], axis=1)
    print(pred_data.head(4))
    bq_schema = [
        bigquery.SchemaField("Date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("Open", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("MA_Prediction", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("ARIMA_Prediction", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("LSTM_Prediction", "FLOAT64", mode="NULLABLE"),
    ]
    bq_client = bigquery.Client(project=project_id)
    save_pandas2bigquery(
        bq_client=bq_client,
        pdf=pred_data,
        dataset_id=dataset_id,
        table_id=table_id,
        schema=bq_schema,
        time_partition_col=time_col,
    )
    # try:
    #     pred_data = pd.concat([entire_data, ma_valid_pred_open, arima_valid_pred_open, lstm_valid_pred_open], axis=1)
    #     print(pred_data.head(4))
    #     bq_schema = [
    #         bigquery.SchemaField("Date", "DATE", mode="REQUIRED"),
    #         bigquery.SchemaField("Open", "FLOAT64", mode="NULLABLE"),
    #         bigquery.SchemaField("High", "FLOAT64", mode="NULLABLE"),
    #         bigquery.SchemaField("Low", "FLOAT64", mode="NULLABLE"),
    #         bigquery.SchemaField("Last", "FLOAT64", mode="NULLABLE"),
    #         bigquery.SchemaField("Close", "FLOAT64", mode="NULLABLE"),
    #         bigquery.SchemaField("TotalTradeQuantity", "INTEGER", mode="NULLABLE"),
    #         bigquery.SchemaField("TurnoverLacs", "FLOAT64", mode="NULLABLE"),
    #         bigquery.SchemaField("MA_Prediction", "FLOAT64", mode="NULLABLE"),
    #         bigquery.SchemaField("ARIMA_Prediction", "FLOAT64", mode="NULLABLE"),
    #         bigquery.SchemaField("LSTM_Prediction", "FLOAT64", mode="NULLABLE"),
    #     ]
    #     bq_client = bigquery.Client(project=project_id)
    #     save_pandas2bigquery(
    #         bq_client=bq_client,
    #         pdf=pred_data,
    #         dataset_id=dataset_id,
    #         table_id=table_id,
    #         schema=bq_schema,
    #         time_partition_col=time_col,
    #     )
    # except:
    #     print("Failed to load data into BigQuery")
    # Return outputs
    gcs_pred_data = dsl.Dataset(uri=dsl.get_uri(suffix="gcs_pred_data.parquet"))
    pred_data.to_parquet(gcs_pred_data.path + ".gzip", index=False, compression="gzip")
    return gcs_pred_data