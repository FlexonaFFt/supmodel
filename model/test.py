import os
import numpy as np
import pandas as pd
from model import ModelManager, DataLoader, Normalizer, Predictor # type: ignore

# Задаем параметры
DATA_FILE = 'data/dataset_.csv'  # Замените на путь к вашим данным
LSTM_MODEL_PATH = 'sets/lstm_model.h5'
DENSE_MODEL_PATH = 'sets/dense_model.h5'
FEATURES = ['theme_id', 'category_id', 'comp_idx', 'start_m',
            'investments_m', 'crowdfunding_m', 'team_idx',
            'tech_idx', 'social_idx', 'demand_idx']
TARGETS = ['social_idx', 'investments_m', 'crowdfunding_m', 'demand_idx',
            'comp_idx']

def main():
    # 1. Загрузка данных
    data_loader = DataLoader(DATA_FILE, FEATURES, TARGETS)
    X, Y = data_loader.get_features_and_targets()

    # 2. Нормализация данных
    normalizer = Normalizer()
    X_scaled, Y_scaled = normalizer.fit_transform(X, Y)

    # 3. Загрузка моделей
    lstm_model = ModelManager.load_model(LSTM_MODEL_PATH, compile_model=False)
    dense_model = ModelManager.load_model(DENSE_MODEL_PATH, compile_model=False)

    # 4. Подготовка для предсказания
    predictor = Predictor(lstm_model, normalizer.scaler_Y)

    # Убедитесь, что X_scaled имеет правильную форму
    initial_input = X_scaled[0].reshape((1, 1, 10))  # Преобразуем форму в (1, 1, 10)
    print("Initial input shape:", initial_input.shape)  # Проверка формы

    # Преобразуем форму для LSTM: (1, 1, num_features)
    initial_input_reshaped = initial_input.reshape((1, 1, initial_input.shape[0]))  # Преобразуем форму
    print("Reshaped input for LSTM:", initial_input_reshaped.shape)  # Проверка новой формы

    # 5. Генерация предсказаний
    predictions = predictor.make_predictions(initial_input, steps=5)

    # 6. Вывод предсказаний
    print("Predictions:\n", predictions)

if __name__ == "__main__":
    main()
