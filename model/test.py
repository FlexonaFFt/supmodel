import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.losses import MeanSquaredError # type: ignore
from sklearn.preprocessing import MinMaxScaler
from model import DataLoader, Normalizer, ModelManager  # type: ignore

# Путь к файлам
DATA_FILE = 'data/dataset_.csv'
LSTM_MODEL_PATH = 'sets/lstm_model.h5'
DENSE_MODEL_PATH = 'sets/dense_model.h5'

# Переменные для признаков и целевых показателей
FEATURES = ['theme_id', 'category_id', 'comp_idx', 'start_m',
            'investments_m', 'crowdfunding_m', 'team_idx',
            'tech_idx', 'social_idx', 'demand_idx']
TARGETS = ['social_idx', 'investments_m', 'crowdfunding_m', 'demand_idx', 'comp_idx']

# Загрузка данных
data_loader = DataLoader(DATA_FILE, FEATURES, TARGETS)
X, Y = data_loader.get_features_and_targets()

# Нормализация данных
normalizer = Normalizer()
X_scaled, Y_scaled = normalizer.fit_transform(X, Y)

# Загрузка моделей
lstm_model = load_model(LSTM_MODEL_PATH, custom_objects={'mse': MeanSquaredError()})
dense_model = load_model(DENSE_MODEL_PATH, custom_objects={'mse': MeanSquaredError()})

# Выполнение предсказаний на тех же данных (например, тестовых)
lstm_predictions = lstm_model.predict(X_scaled)
dense_predictions = dense_model.predict(X_scaled)

# Обратная трансформация предсказаний (если требуется)
Y_pred_lstm = normalizer.inverse_transform_Y(lstm_predictions)
Y_pred_dense = normalizer.inverse_transform_Y(dense_predictions)

# Печать предсказаний
print("LSTM Model Predictions:")
print(Y_pred_lstm)

print("Dense Model Predictions:")
print(Y_pred_dense)

def make_lstm_forecast(model, X_scaled, normalizer, steps, interval):
    """
    Выполняет пошаговые предсказания с использованием LSTM модели на основе предыдущих предсказаний.

    model: обученная модель LSTM
    X_scaled: нормализованные входные данные (последнее известное значение для временного ряда)
    normalizer: объект нормализатора, для обратной трансформации
    steps: количество шагов (например, на 3 года, шаг в полгода = 6 шагов)
    interval: количество временных единиц в одном шаге (например, полгода)

    Возвращает: список предсказанных значений на каждом шаге.
    """
    predictions = []
    current_input = X_scaled.copy()

    for step in range(steps):
        # Получение предсказаний на текущем шаге
        prediction = model.predict(current_input)

        # Обратная трансформация предсказаний (если требуется)
        predicted_values = normalizer.inverse_transform_Y(prediction)

        # Добавление предсказанных значений в результат
        predictions.append(predicted_values)

        # Обновление входа для следующего шага: используем предыдущие предсказанные значения
        # Для временных рядов возможно необходимо обновлять только часть признаков
        current_input[:, -len(TARGETS):] = prediction

    return np.array(predictions)

# Пример использования функции:
# Задаем параметры для предсказаний на 3 года с шагом в полгода
steps = 6  # 3 года, шаг полгода
interval = 6  # Полгода

# Вызов функции для предсказаний
lstm_forecast = make_lstm_forecast(lstm_model, X_scaled, normalizer, steps, interval)

# Печать предсказанных значений
print("LSTM Forecast for 3 years (6 steps with half-year intervals):")
print(lstm_forecast)
