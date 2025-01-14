
import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima
from sklearn import preprocessing
import tensorflow as tf


class TimeSeriesModel:
    def __init__(self):
        pass

    def moving_average_prediction(self, train_set, valid_set):
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
        valid_set["Predictions"] = preds
        return train_set["Open"], valid_set["Open"], valid_set['Predictions'], rms


    def arima_prediction(self, train_set, valid_set):
        """Makes predictions using the ARIMA method.
        Args:
            train_set: DataFrame, training data with 'Open' column.
            valid_set: DataFrame, validation data with 'Open' column.
        Returns:
            train_data: Series, training 'Open' data.
            actual_data: Series, validation 'Open' data.
            predicted_data: Series, predicted 'Open' data for the validation set.
            model: Fitted ARIMA model object.
        """
        print("-----------STOCK PRICE PREDICTION BY AUTO ARIMA-----------")
        training = train_set["Open"]
        validation = valid_set["Open"]
        model = auto_arima(training, start_p=0, start_q=0, max_p=3, max_q=3, start_P=0,
                           seasonal=True, d=1, D=1, trace=True, error_action="ignore",
                           suppress_warnings=True)  # m = number of observations per seasonal cycle
        model.fit(training)
        forecast = model.predict(n_periods=len(validation))
        forecast = pd.Series(forecast, index=validation.index, name="Prediction")  # Use a Series for easier plotting
        rms = np.sqrt(np.mean(np.power((np.array(validation) - np.array(forecast)), 2)))
        print("RMSE value on validation set:", rms)
        print("-----------------------------------------------------------")
        return train_set['Open'], valid_set['Open'], forecast, model, rms


    def lstm_prediction(self, train, valid, entire_data):
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
        model.add(tf.keras.layers.LSTM(units=30, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(tf.keras.layers.LSTM(units=10))
        model.add(tf.keras.layers.Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        # epochs = 15
        model.fit(x_train, y_train, epochs=3, batch_size=1, verbose=2)
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