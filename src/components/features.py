from typing import NamedTuple
from kfp import dsl


@dsl.component(
    base_image="python:3.10",
    packages_to_install=[
        "google-cloud-bigquery==3.10.0",
        "db-dtypes==1.3.1",
        "plotly==5.24.1",
        "scikit-learn==1.6.1",
        "pandas==2.2.3",
        "numpy==2.0.2",
        "matplotlib==3.9.4",
        "statsmodels==0.14.3",
    ],
)
def fetch_dataset(
        project_id: str,
        training_table_name: str,
        time_col: str,
        is_log: bool = False,
) -> NamedTuple("outputs", [("gcs_x_train", dsl.Dataset),
                            ("gcs_x_valid", dsl.Dataset),
                            ("gcs_train_set", dsl.Dataset),
                            ("gcs_valid_set", dsl.Dataset),
                            ("gcs_data", dsl.Dataset)]):
    from google.cloud import bigquery
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import statsmodels.tsa.api as stats_api
    import statsmodels.tsa.seasonal as stat_seasonal
    import math
    def test_stationarity(x_data, y_data, log_transform=False, diff=False):
        """Tests the stationarity of a time series using rolling statistics and the Augmented Dickey-Fuller test.
        Args:
            x_data: DatetimeIndex or array-like, representing the time component.
            y_data: Series or array-like, representing the values of the time series.
            log_transform (bool, optional): Whether to apply a log transformation before testing. Defaults to False.
            diff (bool, optional): Whether to difference the data after log transformation. Defaults to False.
        Returns:
            None. Prints the rolling statistics plot and the ADF test results.
        """
        data = y_data.copy()
        if log_transform:
            data = np.log(data)
            if diff:
                data = data.diff().dropna()
                x_data = x_data[1:]
        rolmean = data.rolling(window=12).mean()
        rolstd = data.rolling(window=12).std()
        plt.figure(figsize=(6, 4))
        plt.plot(x_data, data, color="blue", label="Original")
        plt.plot(x_data, rolmean, color="red", label="Rolling Mean")
        plt.plot(x_data, rolstd, color="black", label="Rolling Std")
        plt.legend(loc="best")
        plt.title("Rolling Mean and Standard Deviation")
        plt.xlabel("Time")
        if log_transform:
            ylabel = "Log"
            if diff:
                ylabel += "-Difference"
            plt.ylabel(ylabel)
        else:
            plt.ylabel("Value")
        # plt.show(block=False)
        adft = stats_api.adfuller(data, autolag="AIC")
        output = pd.Series(adft[0:4],
                           index=["Test Statistic", "p-value", "Number of Lags Used", "Number of Observations Used"])
        for key, value in adft[4].items():
            output[f"Critical Value ({key})"] = value  # Include the critical value label
        print("Results of Dickey-Fuller Test:")
        print(output)
        return plt
    def multiple_plots(pdf_x_data, pdf_data, pdf_data_col_list, y_label):
        """Generates multiple plots for different columns in a DataFrame against a common x-axis.
        Args:
            pdf_x_data: Array-like, representing the x-axis data.
            pdf_data: DataFrame, containing the data to plot.
            pdf_data_col_list: List of str, representing the column names in `pdf_data` to plot.
            y_label: str, the label for the x-axis.
        Returns:
            None, displays plots using matplotlib.pyplot.show()
        """
        multiple_figs = []
        for col in pdf_data_col_list:
            plt.figure(figsize=(4, 2))
            plt.plot(pdf_x_data, pdf_data[col])
            plt.xticks(rotation=45)
            plt.xlabel(y_label)
            plt.ylabel(col)
            plt.title(f"{col} vs. {y_label}")
            plt.tight_layout()
            multiple_figs.append(plt)
        return multiple_figs
    def seasonal_plots(pdf_y_data, stat_seasonal_model, period):
        """Performs seasonal decomposition of a time series and plots the results.
        Args:
            pdf_y_data: Series or array-like, representing the time series data.
            stat_seasonal_model: str, type of seasonal model. Either 'additive' or 'multiplicative'.
            period: int, the period of the seasonality.
        Returns:
            None, displays plots using matplotlib.pyplot.show()
        """
        result = stat_seasonal.seasonal_decompose(pdf_y_data, model=stat_seasonal_model, period=period)
        fig = plt.figure()
        fig = result.plot()
        fig.set_size_inches(16, 6)
        return fig
    def train_val_split(data, is_log=False):
        """Splits a DataFrame into training and validation sets.
        Args:
            data: DataFrame, the input data containing 'Date' and 'Open' columns.
            is_log (bool, optional): Whether to apply log transformation to the 'Open' column. Defaults to False.
        Returns:
            x_train: DataFrame, x-axis data for training set (Date column).
            x_valid_set: DataFrame, x-axis data for validation set (Date column).
            train_set: DataFrame, y-axis data for training set (Open column).
            valid_set: DataFrame, y-axis data for validation set (Open column).
        """
        shape = data.shape[0]
        x_df_new = data[["Date"]].copy()
        df_new = data[["Open"]].copy() # Create a copy to avoid modifying the original
        if is_log:
            df_new = np.log(df_new["Open"]).copy().to_frame(name="Open")
        train_set = df_new.iloc[:math.ceil(shape * 0.9)]
        valid_set = df_new.iloc[math.ceil(shape * 0.9):]
        x_train = x_df_new.iloc[:math.ceil(shape * 0.9)]
        x_valid = x_df_new.iloc[math.ceil(shape * 0.9):]
        print("Shape of Training Set:", train_set.shape)
        print("Shape of Validation Set:", valid_set.shape)
        return x_train, x_valid, train_set, valid_set

    bq_client = bigquery.Client(
        project=project_id,
        default_query_job_config=bigquery.QueryJobConfig(
            allow_large_results=False,
        )
    )
    query = f"""
        SELECT * 
        FROM {training_table_name}
        ORDER BY {time_col} ASC
    """
    data: pd.DataFrame = bq_client.query(query).to_dataframe()
    data = data.fillna(method="ffill").fillna(method="bfill")
    if is_log:
        temp_data = data["Open"].copy()
        data["Open"] = np.log(temp_data)
    multiple_figs = multiple_plots(
        pdf_x_data=data["Date"],
        pdf_data=data,
        pdf_data_col_list=data.columns[2:],
        y_label="Date"
    )
    for col, fig in zip(data.columns[2:], multiple_figs):
        gcs_fig = dsl.Artifact(uri=dsl.get_uri(suffix=col+".png"))
        fig.savefig(gcs_fig.path)
    nonlog_fig = test_stationarity(data["Date"], data["Open"])
    log_fig = test_stationarity(data["Date"], data["Open"],
                                log_transform=True, diff=True)
    gcs_nonlog_stationary = dsl.Artifact(uri=dsl.get_uri(suffix="gcs_nonlog_stationary.png"))
    gcs_log_stationary = dsl.Artifact(uri=dsl.get_uri(suffix="gcs_log_stationary.png"))
    nonlog_fig.savefig(gcs_nonlog_stationary.path)
    log_fig.savefig(gcs_log_stationary.path)

    seasonal_fig = seasonal_plots(pdf_y_data=pd.to_datetime(data["Date"]),
                                  stat_seasonal_model="multiplicative",
                                  period=30)
    gcs_seasonal = dsl.Artifact(uri=dsl.get_uri(suffix="gcs_seasonal.png"))
    seasonal_fig.savefig(gcs_seasonal.path)
    data = data.dropna()
    x_train, x_valid, train_set, valid_set = train_val_split(data, is_log=is_log)
    gcs_x_train = dsl.Dataset(uri=dsl.get_uri(suffix="x_train.parquet"))
    gcs_x_valid = dsl.Dataset(uri=dsl.get_uri(suffix="x_valid.parquet"))
    gcs_train_set = dsl.Dataset(uri=dsl.get_uri(suffix="train_set.parquet"))
    gcs_valid_set = dsl.Dataset(uri=dsl.get_uri(suffix="valid_set.parquet"))
    gcs_data = dsl.Dataset(uri=dsl.get_uri(suffix="data.parquet"))
    x_train.to_parquet(gcs_x_train.path + ".gzip", index=False, compression="gzip")
    x_valid.to_parquet(gcs_x_valid.path + ".gzip", index=False, compression="gzip")
    train_set.to_parquet(gcs_train_set.path + ".gzip", index=False, compression="gzip")
    valid_set.to_parquet(gcs_valid_set.path + ".gzip", index=False, compression="gzip")
    data.to_parquet(gcs_data.path + ".gzip", index=False, compression="gzip")
    # Return outputs
    outputs = NamedTuple("outputs", [("gcs_x_train", dsl.Dataset),
                                     ("gcs_x_valid", dsl.Dataset),
                                     ("gcs_train_set", dsl.Dataset),
                                     ("gcs_valid_set", dsl.Dataset),
                                     ("gcs_data", dsl.Dataset)])
    return outputs(gcs_x_train, gcs_x_valid, gcs_train_set, gcs_valid_set, gcs_data)