#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/27 14:58
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   helper.py
# @Desc     :   

from numpy import ndarray, array
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pandas import DataFrame, read_csv
from tensorflow.keras.callbacks import Callback
from time import perf_counter
from typing import override


class Timer(object):
    """ timing code blocks using a context manager """

    def __init__(self, description: str = None, precision: int = 5):
        """ Initialise the Timer class
        :param description: the description of a timer
        :param precision: the number of decimal places to round the elapsed time
        """
        self._description: str = description
        self._precision: int = precision
        self._start: float = 0.0
        self._end: float = 0.0
        self._elapsed: float = 0.0

    def __enter__(self):
        """ Start the timer """
        self._start = perf_counter()
        print("-" * 50)
        print(f"{self._description} has started.")
        print("-" * 50)
        return self

    def __exit__(self, *args):
        """ Stop the timer and calculate the elapsed time """
        self._end = perf_counter()
        self._elapsed = self._end - self._start

    def __repr__(self):
        """ Return a string representation of the timer """
        if self._elapsed != 0.0:
            # print("-" * 50)
            return f"{self._description} took {self._elapsed:.{self._precision}f} seconds."
        return f"{self._description} has NOT started."


class StTFKLoggerFor5Callbacks(Callback):
    """ Custom Keras Callback to log training metrics and update Streamlit placeholders.
    :param num_placeholders: a dictionary of Streamlit placeholders for metrics
    :return: None
    """

    def __init__(self, num_placeholders: dict = None):
        super().__init__()
        # The key name must match the callback logs
        self._history = {k: [] for k in [
            "loss", "accuracy", "precision", "recall", "auc",
            "val_loss", "val_accuracy", "val_precision", "val_recall", "val_auc"
        ]}
        self._placeholders = num_placeholders

    @override
    def on_epoch_end(self, epoch, logs=None):
        """ At the end of each epoch, log the metrics and update the placeholders.
        :param epoch: the current epoch number
        :param logs: the logs dictionary containing the metrics
        :return: None
        """
        logs = logs or {}
        # Save the training history per epoch
        for key in self._history.keys():
            self._history[key].append(logs.get(key, None))
        # Update the placeholders with the latest metrics
        if self._placeholders:
            for key, placeholder in self._placeholders.items():
                if key in logs and placeholder is not None:
                    placeholder.metric(
                        label=f"Epoch {epoch + 1}: {key.replace('val_', 'Valid ').capitalize()}",
                        value=f"{logs[key]:.4f}"
                    )

    def get_history(self):
        """ Get the training history."""
        return self._history


class StTFKLoggerFor2Callbacks(Callback):
    """ Custom Keras Callback to log training metrics and update Streamlit placeholders.
    :param num_placeholders: a dictionary of Streamlit placeholders for metrics
    :return: None
    """

    def __init__(self, num_placeholders: dict = None):
        super().__init__()
        # The key name must match the callback logs
        self._history = {k: [] for k in ["loss", "accuracy", "val_loss", "val_accuracy"]}
        self._placeholders = num_placeholders

    @override
    def on_epoch_end(self, epoch, logs=None):
        """ At the end of each epoch, log the metrics and update the placeholders.
        :param epoch: the current epoch number
        :param logs: the logs dictionary containing the metrics
        :return: None
        """
        logs = logs or {}
        # Save the training history per epoch
        for key in self._history.keys():
            self._history[key].append(logs.get(key, None))
        # Update the placeholders with the latest metrics
        if self._placeholders:
            for key, placeholder in self._placeholders.items():
                if key in logs and placeholder is not None:
                    placeholder.metric(
                        label=f"Epoch {epoch + 1}: {key.replace('val_', 'Valid ').capitalize()}",
                        value=f"{logs[key]:.4f}"
                    )

    def get_history(self):
        """ Get the training history."""
        return self._history


def txt_reader(filepath: str) -> DataFrame:
    """ Read a txt file structurally
    :param filepath: the path to the file
    :return: a DataFrame
    """
    return read_csv(filepath, delimiter=",", encoding="utf-8")


def useless_cols_dropper(data: DataFrame) -> DataFrame:
    """ Drop the useless cols """
    data.drop(columns=["Volume", "OpenInt"], inplace=True)
    data.drop(columns=["Date"], inplace=True)
    return data


def data_standardiser(data: DataFrame):
    """ Standardise the data
    :param data: a DataFrame
    :return: a DataFrame
    """
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def data_normaliser(data: DataFrame):
    """ Normalise the data
    :param data: a DataFrame
    :return: a DataFrame
    """
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)


def importance_analyser(arr) -> list[float]:
    """ Find the importance between different items
    :param arr: a numpy array
    :return: a list
    """
    pca = PCA()
    pca.fit(arr)
    return pca.explained_variance_ratio_.tolist()


def sequential_data_extractor(
        data: DataFrame, timesteps: int,
        train_rate: float = 0.8,
        target_col: str | None = None) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    # Single col
    if isinstance(target_col, str):
        target_index = data.columns.get_loc(target_col)
    # Multiple cols
    elif isinstance(target_col, list):
        target_index = [data.columns.get_loc(col) for col in target_col]
    # All cols
    elif target_col is None:
        target_index = list(range(data.shape[1]))
    else:
        raise TypeError("target_col must be str or None")

    train_size: int = int(len(data) * train_rate)
    train: ndarray = data[: train_size].values
    X_train: list = []
    y_train: list = []
    for i in range(len(train) - timesteps):
        X_train.append(train[i: i + timesteps])
        y_train.append(train[i + timesteps, target_index])

    test: ndarray = data[train_size:].values
    X_test: list = []
    y_test: list = []
    for i in range(len(test) - timesteps):
        X_test.append(test[i: i + timesteps])
        y_test.append(test[i + timesteps, target_index])

    X_train: ndarray = array(X_train)
    y_train: ndarray = array(y_train)
    X_test: ndarray = array(X_test)
    y_test: ndarray = array(y_test)
    print(type(X_train), type(y_train), type(X_test), type(y_test))
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    return X_train, y_train, X_test, y_test
