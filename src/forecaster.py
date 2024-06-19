import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

from typing import *

class Forecaster:
    def __init__(self, data: np.ndarray, history_size: int, prediction_size: int) -> None:
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
        self.data = data
        
        return
    
    
    def prepare_data(self):
        self._scale_data()
        
        test_size = 0.2
        X_windows, y_windows = Forecaster._generate_windows(self.data, self.history_size, self.prediction_size)
        
        X_train, X_test, y_train, y_test = train_test_split(X_windows, y_windows, test_size=test_size)
        
        self.X_train_data, self.y_train_data = X_train, y_train
        self.X_test_data, self.y_test_data = X_test, y_test
        
        return
    
    
    def _scale_data(self) -> None:
        lower_bound, upper_bound = 0., 1.
        
        # scaling
        scaler = MinMaxScaler(feature_range=(lower_bound, upper_bound))
        self.data = scaler.fit_transform(self.data.reshape(-1, 1))
        
        return
    
    
    def _generate_windows(historical_data: np.ndarray, history_size: int, prediction_size: int):
        X_windows, y_windows = [], []
        for i in range(len(historical_data) - history_size - prediction_size + 1):
            new_x = historical_data[i:i + history_size]
            new_y = historical_data[i + history_size:i + history_size + prediction_size]
            X_windows.append(new_x), y_windows.append(new_y)
        
        X_windows, y_windows = np.array(X_windows), np.array(y_windows)
        
        return X_windows, y_windows
    
    
    def build_model(self, history_size: int, prediction_size: int) -> None:
        self.model = Sequential([
            SimpleRNN(50, activation='relu', input_shape=(history_size, 1)),
            Dense(prediction_size)
        ])
        self.model_loaded = True
        
        return
    
    
    def compile_model(self, optimizer='adam', loss='mse'):
        self.model.compile(
            optimizer=optimizer,
            loss=loss
            )
        
        return
    
    
    def train(self, epochs: int, batch_size: int, initial_epoch: int = 0):
        X = self.X_train_data
        y = self.y_train_data
        
        self.model.fit(X, y, initial_epoch=initial_epoch, epochs=epochs, batch_size=batch_size)
        
        return
    
    
    def predict(self, historical_data: np.ndarray):
        reshaped_data = historical_data.reshape(1, self.history_size, 1)
        y_predicted = self.model.predict(reshaped_data)
        
        return y_predicted
    