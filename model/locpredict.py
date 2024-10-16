import numpy as np
from model import Normalizer, DataLoader, ModelManager  # type: ignore

# Путь к моделям
LSTM_MODEL_PATH = 'sets/lstm_model.h5'
DENSE_MODEL_PATH = 'sets/dense_model.h5'

# Пример новых данных, которые ты хочешь подать на вход модели
# Здесь ты должен указать свои данные, используя те же признаки, что и в обучающей выборке
new_data = np.array([[1.00, 3.00, 3.0, 5900, 29050, 35000, 7.0, 8.30, 5.4, 6.40]])  # Замените это на свои собственные данные

# Загрузка данных, чтобы применить те же параметры нормализации, что и при обучении
DATA_FILE = 'data/dataset_.csv'
FEATURES = ['theme_id', 'category_id', 'comp_idx', 'start_m', 'investments_m', 'crowdfunding_m', 'team_idx', 'tech_idx', 'social_idx', 'demand_idx']
TARGETS = ['social_idx', 'investments_m', 'crowdfunding_m', 'demand_idx', 'comp_idx']

# Загрузка данных для нормализации
data_loader = DataLoader(DATA_FILE, FEATURES, TARGETS)
X_train, Y_train = data_loader.get_features_and_targets()

# Нормализация данных
normalizer = Normalizer()
X_scaled, Y_scaled = normalizer.fit_transform(X_train, Y_train)

# Нормализуем новые данные
new_data_scaled = normalizer.scaler_X.transform(new_data)

# Преобразование данных для LSTM
new_data_lstm = new_data_scaled.reshape((new_data_scaled.shape[0], new_data_scaled.shape[1], 1))

# Загрузка моделей
lstm_model = ModelManager.load_model(LSTM_MODEL_PATH, custom_objects={'mse': 'mean_squared_error'})
dense_model = ModelManager.load_model(DENSE_MODEL_PATH, custom_objects={'mse': 'mean_squared_error'})

# Предсказания для новых данных
lstm_prediction = lstm_model.predict(new_data_lstm)
dense_prediction = dense_model.predict(new_data_scaled)

# Обратная нормализация предсказаний (если нужно вернуть исходный масштаб)
lstm_prediction_inverse = normalizer.inverse_transform_Y(lstm_prediction)
dense_prediction_inverse = normalizer.inverse_transform_Y(dense_prediction)

# Функция для форматированного вывода предсказаний
def print_predictions(prediction, model_name):
    print(f"{model_name} Predictions:")
    print(f"  Social Index: {prediction[0][0]:.2f}")
    print(f"  Future Investments: {prediction[0][1]:.2f}")
    print(f"  Future Crowdfunding: {prediction[0][2]:.2f}")
    print(f"  Future Demand: {prediction[0][3]:.2f}")
    print(f"  Competition Index: {prediction[0][4]:.2f}\n")

# Выводим предсказания для каждой модели
print_predictions(lstm_prediction_inverse, "LSTM Model")
print_predictions(dense_prediction_inverse, "Dense Model")
