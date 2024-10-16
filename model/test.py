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
