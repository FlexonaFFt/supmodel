import numpy as np
import tensorflow as tf
from model import DataLoader, Normalizer, DataProcessor, LSTMModelBuilder, Predictor, Trainer # type: ignore

def get_user_input():
    user_input = []
    user_input.append(float(input("Введите Theme ID: ")))
    user_input.append(float(input("Введите Category ID: ")))
    user_input.append(float(input("Введите Start Money: ")))
    user_input.append(float(input("Введите Investments (M): ")))
    user_input.append(float(input("Введите Crowdfunding (M): ")))
    user_input.append(float(input("Введите Team Index: ")))
    user_input.append(float(input("Введите Tech Index: ")))
    user_input.append(float(input("Введите Competition Index: ")))
    user_input.append(float(input("Введите Social Index: ")))
    user_input.append(float(input("Введите Demand Index: ")))
    return np.array(user_input)

def main():
    # Загрузка модели из файла
    model_path = 'sets/lstm_model.h5'
    custom_objects = {'mse': tf.keras.losses.MeanSquaredError()}
    lstm_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    lstm_model.summary()

    # Получение данных от пользователя
    user_input = get_user_input()

    # Нормализация данных
    normalizer = Normalizer()
    user_input_scaled = normalizer.scaler_X.fit_transform(user_input.reshape(1, -1))

    # Преобразование входных данных в 3D-формат (1 временной шаг, 10 признаков)
    user_input_scaled_reshaped = np.reshape(user_input_scaled, (1, 1, user_input_scaled.shape[1]))  # Убедись, что это (1, 1, 10)

    # Инициализация предсказателя
    predictor = Predictor(lstm_model, normalizer.scaler_Y)

    # Генерация предсказаний
    steps = int(input("Введите количество шагов для предсказания: "))
    print("Форма входных данных:", user_input_scaled_reshaped.shape)
    future_predictions = predictor.make_predictions(user_input_scaled_reshaped, steps)

    # Вывод результатов
    print("\nВвод пользователя:")
    print(f"  Theme ID: {user_input[0]:.2f}")
    print(f"  Category ID: {user_input[1]:.2f}")
    print(f"  Start Money: {user_input[2]:.2f}")
    print(f"  Investments (M): {user_input[3]:.2f}")
    print(f"  Crowdfunding (M): {user_input[4]:.2f}")
    print(f"  Team Index: {user_input[5]:.2f}")
    print(f"  Tech Index: {user_input[6]:.2f}")
    print(f"  Competition Index: {user_input[7]:.2f}")
    print(f"  Social Index: {user_input[8]:.2f}")
    print(f"  Demand Index: {user_input[9]:.2f}\n")

    for i, pred in enumerate(future_predictions):
        print(f"Prediction {i+1}:")
        print(f"  Social Index: {pred[0, 0]:.2f}")
        print(f"  Future Investments: {pred[0, 1]:.2f}")
        print(f"  Future Crowdfunding: {pred[0, 2]:.2f}")
        print(f"  Future Demand: {pred[0, 3]:.2f}")
        print(f"  Competition Index: {pred[0, 4]:.2f}\n")

if __name__ == "__main__":
    main()
