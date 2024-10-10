import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore

# 1. Создание LSTM-модели
def create_model(input_shape):
    model = Sequential()
    
    # LSTM слой, который принимает на вход 10 признаков для первого шага
    model.add(LSTM(256, activation='relu', input_shape=input_shape, return_sequences=False))
    
    # Полносвязный слой, выводящий 5 признаков на выходе
    model.add(Dense(5))
    
    model.compile(optimizer='adam', loss='mse')
    return model

# 2. Генерация последовательных предсказаний
def make_predictions(model, initial_input, steps):
    current_input = initial_input.reshape((1, 1, initial_input.shape[0]))  # Входной вектор с 10 признаками
    predictions = []
    
    # Первое предсказание на основе 10 признаков
    pred = model.predict(current_input)
    predictions.append(pred)
    
    # Для следующих шагов используем комбинацию старых данных и предсказанных значений
    for step in range(1, steps):
        # Обновляем входные данные: используем первые 5 признаков из исходного вектора и предсказанные данные
        current_input = np.concatenate([initial_input[:5], pred.flatten()]).reshape((1, 1, 10))
        
        # Следующее предсказание
        pred = model.predict(current_input)
        predictions.append(pred)
    
    return np.array(predictions)

# 3. Подготовка данных (пример данных)
# Исходные данные для первого предсказания (10 признаков)
initial_input = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# Параметры модели
input_shape = (1, 10)  # timesteps = 1, признаки = 10

# 4. Инициализация и обучение модели
model = create_model(input_shape)
# Задаем случайные данные для обучения (это пример; обычно используется реальный датасет)
X_train = np.random.rand(100, 1, 10)  # 100 примеров, timesteps = 1, признаки = 10
y_train = np.random.rand(100, 5)      # 100 целевых значений, каждый с 5 признаками

# Обучаем модель (эпохи и размер батча можно изменить по необходимости)
model.fit(X_train, y_train, epochs=10, batch_size=16)

# 5. Генерация предсказаний
steps = 6  # Количество шагов для предсказаний
future_predictions = make_predictions(model, initial_input, steps)

# 6. Вывод предсказаний
print("Предсказания на следующие шаги:")
for i, pred in enumerate(future_predictions):
    print(f"Шаг {i+1}: {pred}")
