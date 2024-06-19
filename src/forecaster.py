import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

from typing import *

class Forecaster:
    def __init__(self, data: np.ndarray = None, history_size: int = None, prediction_size: int = None) -> None:
        self.data: np.ndarray = data
        self.history_size: int = history_size
        self.prediction_size: int = prediction_size
        self.windows_size: int = history_size + prediction_size
        self.model: tf.keras.Model = None
        
        self.X_train_data = None
        self.y_train_data = None
        self.X_test_data = None
        self.y_test_data = None
        
        pass
    
    
    def load_data(self, data: np.array) -> None:
        """Loads time series historical data in to the model. 

        Args:
            data (np.array): the historical data.
        
        Examples:
            >>> # creating the model
            >>> forecaster = Forecaster()
            >>> df = pd.read_csv('stock_prices.csv')
            >>> time_series = df['price']
            >>> # loading data
            >>> forecaster.load_data(time_series)
        """
        self.data = data
        
        return
    
    
    def prepare_data(self):
        """Prepares the data for the RNN model consumption.
        
        Examples:
            >>> # creating the model and loading data
            >>> forecaster = Forecaster()
            >>> df = pd.read_csv('stock_prices.csv')
            >>> time_series = df['price']
            >>> forecaster.load_data(time_series)
            >>> # preparing data
            >>> forecaster.prepare_data()
        """
        self._scale_data()
        
        test_size = 0.2
        X_windows, y_windows = Forecaster._generate_windows(self.data, self.history_size, self.prediction_size)
        
        X_train, X_test, y_train, y_test = train_test_split(X_windows, y_windows, test_size=test_size)
        
        self.X_train_data, self.y_train_data = X_train, y_train
        self.X_test_data, self.y_test_data = X_test, y_test
        
        return
    
    
    def _scale_data(self) -> None:
        """Normalizes the data for the model consumption, in training. The whole
        time series will be scaled in the range of [0, 1].
        
        Examples:
            >>> # creating the model and loading data
            >>> forecaster = Forecaster()
            >>> df = pd.read_csv('stock_prices.csv')
            >>> time_series = df['price']
            >>> forecaster.load_data(time_series)
            >>> # scaling data
            >>> forecaster._scale_data()
        """
        lower_bound, upper_bound = 0., 1.
        
        # scaling
        scaler = MinMaxScaler(feature_range=(lower_bound, upper_bound))
        self.data = scaler.fit_transform(self.data.reshape(-1, 1))
        
        return
    
    
    def _generate_windows(historical_data: np.ndarray, history_size: int, prediction_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generates a list of windows from a given historical data. It generates a set of sliding
        windows, with size = history_size + prediction_size.

        Args:
            historical_data (np.ndarray): the historical data of the time series.
            history_size (int): the size of the past window, used to forecast the prediction window.
            prediction_size (int): the size of the prediction window.

        Returns:
            Tuple[np.ndarray, np.ndarray]: a tuple (X_windows, y_windows) with the list of generated windows.
        
        Examples:
            >>> time_series = 2.5 + np.sin(np.arange(0, 15*2*np.pi, 0.01))
            >>> windows = Forecaster._generate_windows(time_series, history_size=50, prediction_size=10)
        """
        X_windows, y_windows = [], []
        for i in range(len(historical_data) - history_size - prediction_size + 1):
            new_x = historical_data[i:i + history_size]
            new_y = historical_data[i + history_size:i + history_size + prediction_size]
            X_windows.append(new_x), y_windows.append(new_y)
        
        X_windows, y_windows = np.array(X_windows), np.array(y_windows)
        
        return X_windows, y_windows
    
    
    def build_model(self, history_size: int, prediction_size: int) -> None:
        """Builds the model with a default architecture.

        Args:
            history_size (int): the size of the past window, used to forecast the prediction window.
            prediction_size (int): the size of the prediction window.
        """
        self.model = Sequential([
            SimpleRNN(50, activation='relu', input_shape=(history_size, 1)),
            Dense(prediction_size)
        ])
        self.model_loaded = True
        
        return
    
    
    def compile_model(self, optimizer: str = 'adam', loss: str = 'mse') -> None:
        """Compiles the tensorflow RNN model, with specified optimizer and
        loss function.

        Args:
            optimizer (str, optional): _description_. Defaults to 'adam'.
            loss (str, optional): _description_. Defaults to 'mse'.
        """
        self.model.compile(
            optimizer=optimizer,
            loss=loss
            )
        
        return
    
    
    def train(self, epochs: int, batch_size: int, initial_epoch: int = 0) -> None:
        """Trains the model with the training data.

        Args:
            epochs (int): epochs for training
            batch_size (int): size of the batches
            initial_epoch (int, optional): the initial epoch. Defaults to 0.
        
        Examples:
            >>> forecaster.train()
        """
        X = self.X_train_data
        y = self.y_train_data
        
        self.model.fit(X, y, initial_epoch=initial_epoch, epochs=epochs, batch_size=batch_size)
        
        return
    
    
    def predict(self, historical_data: np.ndarray) -> np.ndarray:
        """Predicts a future window with a past window. The size of the both windows
        is specified in the model instantiation.

        Args:
            historical_data (np.ndarray): the past window, for forecasting.

        Returns:
            np.ndarray: the predicted future window.
        
        Examples:
            >>> # creating the model, loading data and training
            >>> df = pd.read_csv('stock_prices.csv')
            >>> time_series = df['price']
            >>> forecaster = Forecaster(time_series, 50, 10)
            >>> forecaster._scale_data()
            >>> # predicting
            >>> past_window = times_series[:50]
            >>> forecasted_window = forecaster.predict(past_window)
        """
        reshaped_data = historical_data.reshape(1, self.history_size, 1)
        y_predicted = self.model.predict(reshaped_data)
        
        return y_predicted
    