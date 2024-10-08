import numpy as np
import pandas as pd
import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from sklearn.model_selection import train_test_split

# Загрузка данных и их подготовка
data = pd.read_csv('data/maxextend.csv')
features = ['theme_id', 'category_id', 'comp_idx', 'start_m', 
            'investments_m', 'crowdfunding_m', 'team_idx', 
            'tech_idx', 'social_idx', 'demand_idx']
targets = ['social_idx', 'investments_m', 'demand_idx', 
           'comp_idx']

scaler = MinMaxScaler()
X = data[features].values
y = data[targets].values

# Масштабируем признаки и цели
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
# Формирование данных для LSTM (требуется [samples, timesteps, features])
X_train_lstm = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Функция для создания модели с настраиваемым количеством слоев и нейронов
def create_lstm_model(input_shape, num_layers=2, neurons_per_layer=64):
    model = Sequential()
    model.add(LSTM(units=neurons_per_layer, return_sequences=(num_layers > 1), input_shape=input_shape))
    for _ in range(num_layers - 1):
        model.add(LSTM(units=neurons_per_layer, return_sequences=False))
    model.add(Dense(units=4, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

# Создаем модель с параметрами (например, 2 слоя, 64 нейрона на слой)
model = create_lstm_model(input_shape=(1, X_train_lstm.shape[2]), num_layers=2, neurons_per_layer=64)

# Обучение модели
history = model.fit(X_train_lstm, y_train, epochs=25, batch_size=16, validation_split=0.2, verbose=1)
test_loss = model.evaluate(X_test_lstm, y_test)
print(f'Test Loss: {test_loss}')

# Тестовое предсказание
predictions = model.predict(X_test_lstm)
predictions_rescaled = scaler.inverse_transform(predictions)
# X_test_rescaled = scaler.inverse_transform(X_test)
for i in range(5):
    print(f"Prediction {i+1}:")
    print(f"  Social Index: {predictions_rescaled[i, 0]:.2f}")
    print(f"  Future Investments: {predictions_rescaled[i, 1]:.2f}")
    print(f"  Future Demand: {predictions_rescaled[i, 2]:.2f}")
    print(f"  Competition Index: {predictions_rescaled[i, 3]:.2f}")
    print()