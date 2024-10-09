import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore

# Загрузка данных и их подготовка
data = pd.read_csv('data/maxextend.csv')
features = ['theme_id', 'category_id', 'comp_idx', 'start_m', 
            'investments_m', 'crowdfunding_m', 'team_idx', 
            'tech_idx', 'social_idx', 'demand_idx']
targets = ['social_idx', 'investments_m', 'crowdfunding_m', 'demand_idx', 
           'comp_idx']

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X = data[features].values
y = data[targets].values
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Формирование данных для LSTM (требуется [samples, timesteps, features])
X_train_lstm = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Функция для создания LSTM-модели
def create_model(input_shape):
    model = Sequential()
    
    # Первый слой LSTM, принимающий 10 признаков на входе
    model.add(LSTM(256, activation='relu', input_shape=input_shape, return_sequences=False))
    
    # Полносвязный слой, выводящий 5 целевых признаков
    model.add(Dense(5))
    
    model.compile(optimizer='adam', loss='mse')
    return model

# Функция для генерации временных предсказаний
def make_predictions(model, initial_input, steps, scaler_y):
    current_input = initial_input.reshape((1, 1, initial_input.shape[0]))  # Входной вектор с 10 признаками
    predictions = []
    
    # Первое предсказание на основе 10 признаков
    pred = model.predict(current_input)
    predictions.append(scaler_y.inverse_transform(pred))  # Обратное преобразование
    
    # Для следующих шагов используем комбинацию старых данных и предсказанных значений
    for step in range(1, steps):
        # Обновляем входные данные: используем первые 5 признаков из исходного вектора и предсказанные данные
        current_input = np.concatenate([initial_input[:5], pred.flatten()]).reshape((1, 1, 10))
        
        # Следующее предсказание
        pred = model.predict(current_input)
        predictions.append(scaler_y.inverse_transform(pred))  # Обратное преобразование
    
    return np.array(predictions)

# Параметры модели
input_shape = (1, X_train_lstm.shape[2])  # timesteps = 1, признаки = 10

# Создаем модель
model = create_model(input_shape)

# Обучение модели
model.fit(X_train_lstm, y_train, epochs=25, batch_size=16, validation_split=0.2)

# Тестовые данные для предсказаний
initial_input = X_test[0]  # Используем первый тестовый пример для предсказаний

# Количество шагов для предсказаний
steps = 6

# Генерация временных предсказаний
future_predictions = make_predictions(model, initial_input, steps, scaler_y)

# Вывод предсказаний
X_test_rescaled = scaler_X.inverse_transform(X_test)
print(f"  Input Data:")
print(f"    Theme ID: {X_test_rescaled[0, 0]:.2f}")
print(f"    Category ID: {X_test_rescaled[0, 1]:.2f}")
print(f"    Start Month: {X_test_rescaled[0, 2]:.2f}")
print(f"    Investments (M): {X_test_rescaled[0, 3]:.2f}")
print(f"    Crowdfunding (M): {X_test_rescaled[0, 4]:.2f}")
print(f"    Team Index: {X_test_rescaled[0, 5]:.2f}")
print(f"    Tech Index: {X_test_rescaled[0, 6]:.2f}")
print(f"    Competition Index: {X_test_rescaled[0, 7]:.2f}")
print(f"    Social Index: {X_test_rescaled[0, 8]:.2f}")
print(f"    Demand Index: {X_test_rescaled[0, 9]:.2f}")
print()
for i, pred in enumerate(future_predictions):
    print(f"Prediction {i+1}:")
    print(f"  Social Index: {pred[0, 0]:.2f}")
    print(f"  Future Investments: {pred[0, 1]:.2f}")
    print(f"  Future Crowdfunding: {pred[0, 2]:.2f}")
    print(f"  Future Demand: {pred[0, 3]:.2f}")
    print(f"  Competition Index: {pred[0, 4]:.2f}")
    print()
