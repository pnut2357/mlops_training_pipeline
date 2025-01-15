from typing import NamedTuple
from kfp import dsl


@dsl.component(
    base_image="python:3.10",
    packages_to_install=[
        "google-cloud-bigquery==3.10.0",
        "db-dtypes==1.3.1",
        "scikit-learn==1.6.1",
        "pandas==2.2.3",
        "numpy==1.26.4",
        "tensorflow==2.18.0",
        "pmdarima==2.0.4",
        "statsmodels==0.14.3",
    ],
)
def train_timeseries_models(
        gcs_train_set: dsl.Dataset,
        gcs_valid_set: dsl.Dataset,
        gcs_data: dsl.Dataset,
) -> NamedTuple("outputs", [("gcs_ma_valid_pred_open", dsl.Dataset),
                            ("gcs_arima_valid_pred_open", dsl.Dataset),
                            ("gcs_lstm_valid_pred_open", dsl.Dataset),
                            ("eval_metrics", dsl.Metrics),
                            ("gcs_metrics", dsl.Artifact),
                            ("gcs_arima_stock_model", dsl.Model),
                            ("gcs_lstm_stock_model", dsl.Model)]):
    from google.cloud import bigquery
    import numpy as np
    import pandas as pd
    from pmdarima.arima import auto_arima
    import statsmodels.tsa.api as stats_api
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn import preprocessing
    from sklearn.metrics import mean_squared_error
    import tensorflow as tf
    import joblib
    import json
    def moving_average_prediction(train_set, valid_set):
        """Makes predictions using the moving average method.
        Args:
            train_set: DataFrame, training data with 'Open' column.
            valid_set: DataFrame, validation data with 'Open' column.
        Returns:
            train_data: Series, training 'Open' data.
            actual_data: Series, validation 'Open' data.
            predicted_data: Series, predicted 'Open' data for the validation set.
        """
        print("-----------STOCK PRICE PREDICTION BY MOVING AVERAGE--------")
        preds = []
        for i in range(0, valid_set.shape[0]):
            a = train_set["Open"][len(train_set) - valid_set.shape[0] + i:].sum() + sum(preds)
            b = a / (valid_set.shape[0])
            preds.append(b)
        rms = np.sqrt(np.mean(np.power((np.array(valid_set["Open"]) - preds), 2)))
        print("RMSE value on validation set:", rms)
        print("-----------------------------------------------------------")
        valid_set["MA_Predictions"] = preds
        return train_set["Open"], valid_set["Open"], valid_set['MA_Predictions'], rms

    def arima_prediction(train_set, valid_set):
        """Makes predictions using the ARIMA method.
        Args:
            train_set: DataFrame, training data with 'Open' column.
            valid_set: DataFrame, validation data with 'Open' column.
        Returns:
            train_data: Series, training 'Open' data.
            actual_data: Series, validation 'Open' data.
            predicted_data: Series, predicted 'Open' data for the validation set.
            model: Fitted ARIMA model object.
            auto_arima(training, start_p=0, start_q=0, max_p=3, max_q=3, start_P=0, max_P=2, max_Q=2,
                           seasonal=True, d=None, D=None, trace=True, error_action="ignore", maxiter=500,
                           max_order=10, suppress_warnings=True, stepwise=True)
        """
        print("-----------STOCK PRICE PREDICTION BY AUTO ARIMA-----------")
        training = train_set["Open"].bfill()
        validation = valid_set["Open"].bfill()
        result = stats_api.adfuller(training)
        if result[1] > 0.05:
            print("Data is not stationary. Applying differencing...")
            train_set["Open_diff"] = train_set["Open"].diff().bfill()
            valid_set["Open_diff"] = valid_set["Open"].diff().bfill()
            training = train_set["Open_diff"]
            validation = valid_set["Open_diff"]
        assert not training.isnull().any(), "Training data contains NaN values after preprocessing."
        model = auto_arima(training, start_p=0, start_q=0, max_p=5, max_q=5, start_P=0, max_P=2, max_Q=2,
                           seasonal=True, m=12, d=None, D=None, trace=True, error_action="ignore", maxiter=500,
                           suppress_warnings=True, stepwise=True)
        model.fit(training)
        forecast = model.predict(len(validation))
        # model = ARIMA(training, order=(1, 1, 1))
        # model = model.fit(method_kwargs={"maxiter": 500})
        # forecast = results.forecast(steps=len(validation))
        # forecast = pd.Series(forecast name="Prediction")  # Use a Series for easier plotting
        forecast = pd.DataFrame(forecast, columns=["ARIMA_Prediction"])
        print("ARIMA forecast:", len(forecast))
        print(forecast)
        rms = np.sqrt(np.mean(np.power((np.array(validation) - np.array(forecast)), 2)))
        print("RMSE value on validation set:", rms)
        print("-----------------------------------------------------------")
        return train_set['Open'], valid_set['Open'], forecast["ARIMA_Prediction"], model, rms

    def lstm_prediction(train, valid, entire_data):
        """Makes predictions using the LSTM method.
        Args:
            train: DataFrame, training data with 'Open' column.
            valid: DataFrame, validation data with 'Open' column.
            entire_data: DataFrame, the entire dataset with 'Open' column.
        Returns:
            train_data: Series, training 'Open' data.
            actual_data: Series, validation 'Open' data.
            predicted_data: Array, predicted 'Open' data for the validation set.
            model: Trained LSTM model object.
        """
        print("-----------STOCK PRICE PREDICTION BY LONG SHORT TERM MEMORY (LSTM)-----------")
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(entire_data)  # Fit on the whole dataset
        x_train, y_train = [], []
        for i in range(40, len(train)):
            x_train.append(scaled_data[i - 40:i, 0])
            y_train.append(scaled_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        model = tf.keras.Sequential()
        # units = 50
        model.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(tf.keras.layers.LSTM(units=10))
        model.add(tf.keras.layers.Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        # epochs = 15
        model.fit(x_train, y_train, epochs=10, batch_size=1, verbose=2)
        inputs = entire_data[len(entire_data) - len(valid) - 40:]
        inputs.values.reshape(-1, 1)
        inputs = scaler.transform(inputs)
        X_test = []
        for i in range(40, inputs.shape[0]):
            X_test.append(inputs[i - 40:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        opening_price = model.predict(X_test)
        opening_price = scaler.inverse_transform(opening_price)
        rms = np.sqrt(np.mean(np.power((valid - opening_price), 2)))
        print('RMSE value on validation set:', rms)
        return train['Open'], valid['Open'], opening_price, model, rms

    def check_value(value):
        if (isinstance(value, float) and np.isnan(value)) or (value is None) or (isinstance(value, str)):
            return -1
        else:
            return value
    train_set = pd.read_parquet(gcs_train_set.path+".gzip")
    valid_set = pd.read_parquet(gcs_valid_set.path+".gzip")
    entire_data = pd.read_parquet(gcs_data.path+".gzip")

    pred_train_open, pred_valid_open, ma_valid_pred_open, ma_rms = moving_average_prediction(train_set, valid_set)
    pred_train_open, pred_valid_open, arima_valid_pred_open, arima_stock_model, arima_rms = arima_prediction(train_set,
                                                                                                             valid_set)
    pred_train_open, pred_valid_open, lstm_valid_pred_open, lstm_stock_model, lstm_rms = lstm_prediction(train_set,
                                                                                                        valid_set,
                                                                                               entire_data[['Open']])

    gcs_arima_stock_model = dsl.Model(uri=dsl.get_uri(suffix="arima_stock_model.joblib"))
    gcs_lstm_stock_model = dsl.Model(uri=dsl.get_uri(suffix="lstm_stock_model.keras"))
    joblib.dump(arima_rms, gcs_arima_stock_model.path)
    lstm_stock_model.save(gcs_lstm_stock_model.path)

    gcs_ma_valid_pred_open = dsl.Dataset(uri=dsl.get_uri(suffix="ma_valid_pred_open.parquet"))
    gcs_arima_valid_pred_open = dsl.Dataset(uri=dsl.get_uri(suffix="arima_valid_pred_open.parquet"))
    gcs_lstm_valid_pred_open = dsl.Dataset(uri=dsl.get_uri(suffix="lstm_valid_pred_open.parquet"))

    ma_valid_pred_open.to_frame(name="MA_Prediction").to_parquet(gcs_ma_valid_pred_open.path + ".gzip", index=False, compression="gzip")
    arima_valid_pred_open.to_frame(name="ARIMA_Prediction").to_parquet(gcs_arima_valid_pred_open.path + ".gzip", index=False, compression="gzip")
    lstm_valid_pred_open_df = pd.DataFrame(lstm_valid_pred_open, columns=["LSTM_Prediction"] )
    lstm_valid_pred_open_df.to_parquet(gcs_lstm_valid_pred_open.path + ".gzip", index=False, compression="gzip")

    rms_hmap = {"ma_rms": ma_rms, "arima_rms": arima_rms, "lstm_rms": lstm_rms}
    eval_metrics = dsl.Metrics()
    for metric_name, metric_val in rms_hmap.items():
        print(metric_name, metric_val)
        eval_metrics.log_metric(metric_name, round(check_value(metric_val), 6))
    gcs_metrics = dsl.Artifact(uri=dsl.get_uri(suffix="metrics"))
    with open(gcs_metrics.path, "w") as f:
        f.write(json.dumps(rms_hmap))
    outputs = NamedTuple("outputs", [("gcs_ma_valid_pred_open", dsl.Dataset),
                                     ("gcs_arima_valid_pred_open", dsl.Dataset),
                                     ("gcs_lstm_valid_pred_open", dsl.Dataset),
                                     ("eval_metrics", dsl.Metrics),
                                     ("gcs_metrics", dsl.Artifact),
                                     ("gcs_arima_stock_model", dsl.Model),
                                     ("gcs_lstm_stock_model", dsl.Model)])
    return outputs(gcs_ma_valid_pred_open, gcs_arima_valid_pred_open, gcs_lstm_valid_pred_open, eval_metrics,
                   gcs_metrics, gcs_arima_stock_model, gcs_lstm_stock_model)