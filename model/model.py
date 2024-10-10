import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore

class DataPreprocessor:


    def __init__(self, data_path, features, targets):
        self.data = pd.read_csv(data_path)
        self.features = features
        self.targets = targets
        self.scaler_X = MinMaxScaler()
        self.scaler_Y = MinMaxScaler()

    def preprocess(self):
        X = self.data[self.features].values
        Y = self.data[self.targets].values
        X_scaled = self.scaler_X.fit_transform(X)
        Y_scaled = self.scaler_Y.fit_transform(Y)
        return X_scaled, Y_scaled

    def inverse_transform_X(self, X_scaled):
        return self.scaler_X.inverse_transform(X_scaled)

    def inverse_transform_Y(self, Y_scaled):
        return self.scaler_Y.inverse_transform(Y_scaled)


class LSTMModel:


    def __init__(self, input_shape):
        self.model = self.build_model(input_shape)

    def build_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(256, activation='relu', input_shape=input_shape, return_sequences=False))
        model.add(Dense(5))
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, X_train, Y_train, epochs=25, batch_size=16, validation_split=0.2):
        self.model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    def predict(self, X):
        self.model.predict(X)



class Predictor:


    def __init__(self, model, scaler_Y):
        self.model = model
        self.scaler_Y = scaler_Y

    def make_predictions(self, initial_input, steps):
        current_input = initial_input.reshape((1, 1, initial_input.shape[0]))
        predictions, pred = [], self.model.predict(current_input)
        predictions.append(self.scaler_Y.inverse_transform(pred))

        for step in range(1, steps):
            current_input = np.concatenate([initial_input[:5], pred.flatten()]).reshape((1, 1, 10))
            pred = self.model.predict(current_input)
            predictions.append(self.scaler_Y.inverse_transform(pred))
        return np.array(predictions)
