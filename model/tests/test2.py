import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.losses import MeanSquaredError # type: ignore
from sklearn.preprocessing import MinMaxScaler
from model import DataLoader, Normalizer, ModelManager # type: ignore

# Путь к файлам
DATA_FILE = 'data/dataset_.csv'
LSTM_MODEL_PATH = 'sets/lstm_model.h5'
DENSE_MODEL_PATH = 'sets/dense_model.h5'

# Переменные для признаков и целевых показателей
FEATURES = ['theme_id', 'category_id', 'comp_idx', 'start_m',
            'investments_m', 'crowdfunding_m', 'team_idx',
            'tech_idx', 'social_idx', 'demand_idx']
TARGETS = ['social_idx', 'investments_m', 'crowdfunding_m', 'demand_idx', 'comp_idx']

def make_lstm_forecast(model, X_scaled, normalizer, steps):
    predictions = []
    current_input = X_scaled.copy()

    for step in range(steps):
        prediction = model.predict(current_input)
        predicted_values = normalizer.inverse_transform_Y(prediction)
        predictions.append(predicted_values)
        current_input[:, -len(TARGETS):] = prediction
    return np.array(predictions)

def display_model_predictions(predictions, model_name):
    """
    Функция для красивого вывода предсказаний модели.
    predictions: массив предсказанных значений
    model_name: название модели (например, 'LSTM Model' или 'Dense Model')
    """
    print(f"\n{model_name} Predictions:")
    print("=" * 40)

    for i, pred in enumerate(predictions):
        print(f"Prediction {i + 1}:")
        print(f"{'-' * 40}")
        print(f"  Social Index        : {pred[0, 0]:>10.2f}")
        print(f"  Future Investments  : {pred[0, 1]:>10.2f}")
        print(f"  Future Crowdfunding : {pred[0, 2]:>10.2f}")
        print(f"  Future Demand       : {pred[0, 3]:>10.2f}")
        print(f"  Competition Index   : {pred[0, 4]:>10.2f}")
        print(f"{'=' * 40}\n")

def get_user_input():
    """
    Функция для получения пользовательских данных через консоль.
    """
    print("Введите значения для следующих признаков:")
    user_input = []
    for feature in FEATURES:
        value = float(input(f"{feature}: "))
        user_input.append(value)

    return np.array(user_input).reshape(1, -1)

def main():
    # Загрузка данных
    data_loader = DataLoader(DATA_FILE, FEATURES, TARGETS)
    X, Y = data_loader.get_features_and_targets()

    # Нормализация данных
    normalizer = Normalizer()
    X_scaled, Y_scaled = normalizer.fit_transform(X, Y)

    # Загрузка моделей
    lstm_model = load_model(LSTM_MODEL_PATH, custom_objects={'mse': MeanSquaredError()})
    dense_model = load_model(DENSE_MODEL_PATH, custom_objects={'mse': MeanSquaredError()})

    # Выбор модели пользователем
    model_choice = input("Выберите модель (LSTM или Dense): ").strip().lower()

    if model_choice == "lstm":
        model = lstm_model
        model_name = "LSTM Model"
    elif model_choice == "dense":
        model = dense_model
        model_name = "Dense Model"
    else:
        print("Неправильный выбор модели!")
        return

    # Получение пользовательских данных
    user_input = get_user_input()

    # Нормализация пользовательских данных
    user_input_scaled = normalizer.transform_X(user_input)

    # Выполнение предсказаний
    if model_choice == "lstm":
        steps = 3
        predictions = make_lstm_forecast(model, user_input_scaled, normalizer, steps)
        display_model_predictions(predictions, f"{model_name} Forecast")
    else:
        predictions = model.predict(user_input_scaled)
        Y_pred = normalizer.inverse_transform_Y(predictions)
        display_model_predictions(Y_pred, model_name)

if __name__ == "__main__":
    main()
