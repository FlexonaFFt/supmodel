from sklearn.model_selection import train_test_split
from model import DataLoader, Normalizer, DataProcessor, ModelSaver, IterativeTrainer # type: ignore
from model import DenseModelBuilder, LSTMModelBuilder, Trainer, Predictor # type: ignore
import numpy as np
import pandas as pd
import tensorflow as tf

def main():
    file_path = 'data/dataset_.csv'
    features = ['theme_id', 'category_id', 'comp_idx', 'start_m',
                'investments_m', 'crowdfunding_m', 'team_idx',
                'tech_idx', 'social_idx', 'demand_idx']
    targets = ['social_idx', 'investments_m', 'crowdfunding_m', 'demand_idx',
                'comp_idx']

    # Загрузка данных
    data_loader = DataLoader(file_path, features, targets)
    X, Y = data_loader.get_features_and_targets()

    # Нормализация данных
    normalizer = Normalizer()
    X_scaled, Y_scaled = normalizer.fit_transform(X, Y)

    # Разделение данных
    processor = DataProcessor()
    X_train, X_test, Y_train, Y_test = processor.split_data(X_scaled, Y_scaled)
    X_train_lstm = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Создание LSTM модели
    lstm_builder = LSTMModelBuilder(input_shape=(1, X_train_lstm.shape[2]))
    lstm_model = lstm_builder.build_model()

    # Оптимизатор и функция потерь
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # type: ignore
    loss_fn = tf.keras.losses.MeanSquaredError() # type: ignore

    # Создание класса для сохранения модели
    model_saver = ModelSaver(lstm_model)

    # Обучение модели с сохранением
    trainer = IterativeTrainer(
        model=lstm_model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=50,  # Указываешь количество эпох
        save_interval=50,  # Как часто сохранять модель
        model_saver=model_saver
    )

    # Подготовка тренировочных данных
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_lstm, Y_train)).batch(32)

    # Запуск процесса обучения
    trainer.train(train_dataset)

    # Предсказания
    predictor = Predictor(lstm_model, normalizer.scaler_Y)
    initial_input = X_test[0]
    steps = 6
    future_predictions = predictor.make_predictions(initial_input, steps)

    X_test_rescaled = normalizer.inverse_transform_X(X_test)
    print(f"Input Data:")
    print(f"Theme ID: {X_test_rescaled[0, 0]:.2f}")
    print(f"Category ID: {X_test_rescaled[0, 1]:.2f}")
    print(f"Start Month: {X_test_rescaled[0, 2]:.2f}")
    print(f"Investments (M): {X_test_rescaled[0, 3]:.2f}")
    print(f"Crowdfunding (M): {X_test_rescaled[0, 4]:.2f}")
    print(f"Team Index: {X_test_rescaled[0, 5]:.2f}")
    print(f"Tech Index: {X_test_rescaled[0, 6]:.2f}")
    print(f"Competition Index: {X_test_rescaled[0, 7]:.2f}")

if __name__ == '__main__':
    main()
