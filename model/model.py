import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore

class DataLoader:
    """Класс для загрузки и хранения данных"""

    def __init__(self, file_path, features, targets):
        self.data = pd.read_csv(file_path)
        self.features = features
        self.targets = targets

    def get_features_and_targets(self):
        X = self.data[self.features].values
        Y = self.data[self.targets].values
        return X, Y

class Normalizer:
    """Класс для нормализации данных с помощью MinMaxScaler"""

    def __init__(self):
        self.scaler_X = MinMaxScaler()
        self.scaler_Y = MinMaxScaler()

    def fit_transform(self, X, Y):
        X_scaled = self.scaler_X.fit_transform(X)
        Y_scaled = self.scaler_Y.fit_transform(Y)
        return X_scaled, Y_scaled

    def transform(self, X, Y):
        X_scaled = self.scaler_X.transform(X)
        Y_scaled = self.scaler_Y.transform(Y)
        return X_scaled, Y_scaled

    def inverse_transform_X(self, X_scaled):
        return self.scaler_X.inverse_transform(X_scaled)

    def inverse_transform_Y(self, Y_scaled):
        return self.scaler_Y.inverse_transform(Y_scaled)

class DataProcessor:
    """Класс для разделения данных на тренировочные и тестовые выборки"""

    def __init__(self, test_size=0.2, random_state=20):
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, X, Y):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=self.test_size, random_state=self.random_state)
        return X_train, X_test, Y_train, Y_test

class LSTMModelBuilder:
    """Класс для создания и конфигурации LSTM модели"""

    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build_model(self):
        model = Sequential()
        model.add(LSTM(256, activation='relu', input_shape=self.input_shape, return_sequences=False))
        model.add(Dense(5))
        model.compile(optimizer='adam', loss='mse')
        return model

class DenseModelBuilder:
    """Класс для создания и конфигурации Dense модели для одиночных предсказаний"""

    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build_model(self):
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=self.input_shape))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(5))
        model.compile(optimizer='adam', loss='mse')
        return model

class Trainer:
    """Класс для обучения модели"""

    def __init__(self, model, X_train, Y_train, batch_size=16, epochs=25, validation_split=0.2):
        self.model = model
        self.X_train = X_train
        self.Y_train = Y_train
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split

    def train(self):
        self.model.fit(self.X_train, self.Y_train, batch_size=self.batch_size, epochs=self.epochs, validation_split=self.validation_split)

class Predictor:
    """Класс для генерации предсказаний с помощью обученной модели"""

    def __init__(self, model, scaler_Y):
        self.model = model
        self.scaler_Y = scaler_Y

    def make_predictions(self, initial_input, steps):
        current_input = initial_input.reshape((1, 1, initial_input.shape[0]))
        predictions = []
        pred = self.model.predict(current_input)
        predictions.append(self.scaler_Y.inverse_transform(pred))

        for step in range(1, steps):
            current_input = np.concatenate([initial_input[:5], pred.flatten()]).reshape((1, 1, 10))
            pred = self.model.predict(current_input)
            predictions.append(self.scaler_Y.inverse_transform(pred))
        return np.array(predictions)

class ModelSaver:
    """Класс для локального сохранения моделей"""

    def __init__(self, model_1, model_2, model_dir='save/'):
        self.model_1 = model_1
        self.model_2 = model_2
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def save_models(self, epoch):
        self.model_1.save(os.path.join(self.model_dir, f"model_1_epoch_{epoch}"))
        self.model_2.save(os.path.join(self.model_dir, f"model_2_epoch_{epoch}"))
        print(f"Модели сохранены после {epoch} эпох.")

    def load_models(self, epoch):
        self.model_1 = tf.keras.models.load_model(os.path.join(self.model_dir, f"model_1_epoch_{epoch}")) # type: ignore
        self.model_2 = tf.keras.models.load_model(os.path.join(self.model_dir, f"model_2_epoch_{epoch}")) # type: ignore
        print(f"Модели загружены для {epoch} эпох.")

class InteractiveTrainer:
    """Класс для обучения моделей в несколько циклов"""

    def __init__(self, model_1, model_2, optimizer_1, optimizer_2, loss_fn, num_epochs, save_interval, model_saver):
        self.model_1 = model_1
        self.model_2 = model_2
        self.optimizer_1 = optimizer_1
        self.optimizer_2 = optimizer_2
        self.loss_fn = loss_fn # Функция потерь для обеих моделей
        self.num_epochs = num_epochs
        self.save_interval = save_interval # Интервал сохранения моделей
        self.model_saver = model_saver # Объект ModelSaver

    def train_step(self, model, optimizer, inputs, targets):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = self.loss_fn(targets, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables)) # type: ignore
        return loss

    def train(self, train_dataset):
        for epoch in range(1, self.num_epochs + 1):
            print(f"Эпоха {epoch}/{self.num_epochs}")
            total_loss_1 = 0
            total_loss_2 = 0
            for inputs, targets in train_dataset:
                loss_1 = self.train_step(self.model_1, self.optimizer_1, inputs, targets)
                total_loss_1 += loss_1
                loss_2 = self.train_step(self.model_2, self.optimizer_2, inputs, targets)
                total_loss_2 += loss_2
            print(f"Потери модели 1: {total_loss_1.numpy()}, Потери модели 2: {total_loss_2.numpy()}") # type: ignore
            if epoch % self.save_interval == 0:
                self.model_saver.save_models(epoch)
        print("Обучение завершено!")
