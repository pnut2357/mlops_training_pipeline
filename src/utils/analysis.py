import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.api as stats_api
import statsmodels.tsa.seasonal as stat_seasonal
import math

class Analysis:
    def __init__(self):
        pass

    def test_stationarity(self, x_data, y_data, log_transform=False, diff=False):
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
        plt.show(block=False)
        adft = stats_api.adfuller(data, autolag="AIC")
        output = pd.Series(adft[0:4], index=["Test Statistic", "p-value", "Number of Lags Used", "Number of Observations Used"])
        for key, value in adft[4].items():
            output[f"Critical Value ({key})"] = value  # Include the critical value label
        print("Results of Dickey-Fuller Test:")
        print(output)


    def multiple_plots(self, pdf_x_data, pdf_data, pdf_data_col_list, y_label):
        """Generates multiple plots for different columns in a DataFrame against a common x-axis.
        Args:
            pdf_x_data: Array-like, representing the x-axis data.
            pdf_data: DataFrame, containing the data to plot.
            pdf_data_col_list: List of str, representing the column names in `pdf_data` to plot.
            y_label: str, the label for the x-axis.
        Returns:
            None, displays plots using matplotlib.pyplot.show()
        """
        for col in pdf_data_col_list:
            plt.figure(figsize=(4, 2))
            plt.plot(pdf_x_data, pdf_data[col])
            plt.xticks(rotation=45)
            plt.xlabel(y_label)
            plt.ylabel(col)
            plt.title(f"{col} vs. {y_label}")
            plt.tight_layout()
            plt.show()

    def seasonal_plots(self, pdf_y_data, stat_seasonal_model, period):
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
        plt.show(block=False)

    def plot_predictions(self, x_train, x_val, train_data, actual_data, predicted_data, title, xlabel, ylabel):
        """Plots training data, actual validation data, and predicted validation data.
        Args:
            x_train: Array-like, representing the x-axis data for the training set.
            x_val: Array-like, representing the x-axis data for the validation set.
            train_data: Array-like, representing the y-axis data for the training set.
            actual_data: Array-like, representing the y-axis data for the actual values in the validation set.
            predicted_data: Array-like, representing the y-axis data for the predicted values in the validation set.
            title: str, the title of the plot.
            ylabel: str, the label for the y-axis.
        Returns:
            None. Displays the plot using matplotlib.pyplot.show().
        """
        plt.plot(x_train, train_data)
        plt.plot(x_val, actual_data)
        plt.plot(x_val, predicted_data)
        plt.xlabel(xlabel, size=12)
        plt.ylabel(ylabel, size=12)
        plt.title(title, size=14)
        plt.legend(["Model Training Data", "Actual Data", "Predicted Data"])
        plt.show()

    def train_val_split(self, data, is_log=False):
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
        df_new = data[["Open"]].copy()  # Create a copy to avoid modifying the original
        if is_log:
            df_new["Open"] = np.log(df_new["Open"])
        train_set = df_new.iloc[:math.ceil(shape * 0.9)]
        valid_set = df_new.iloc[math.ceil(shape * 0.9):]
        x_train = x_df_new.iloc[:math.ceil(shape * 0.9)]
        x_valid = x_df_new.iloc[math.ceil(shape * 0.9):]
        print("Shape of Training Set:", train_set.shape)
        print("Shape of Validation Set:", valid_set.shape)
        return x_train, x_valid, train_set, valid_set