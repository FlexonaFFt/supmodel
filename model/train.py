from sklearn.model_selection import train_test_split
from model import DataLoader, Normalizer, DataProcessor, ModelManager # type: ignore
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

    data_loader = DataLoader(file_path, features, targets)
    X, Y = data_loader.get_features_and_targets()

    # Нормализация данных
    normalizer = Normalizer()
    X_scaled, Y_scaled = normalizer.fit_transform(X, Y)

    # Разделение на тренировочные и тестовые данные
    data_processor = DataProcessor()
    X_train, X_test, Y_train, Y_test = data_processor.split_data(X_scaled, Y_scaled)

    # Построение моделей
    lstm_builder = LSTMModelBuilder(input_shape=(X_train.shape[1], 1))
    lstm_model = lstm_builder.build_model()

    dense_builder = DenseModelBuilder(input_shape=(X_train.shape[1],))
    dense_model = dense_builder.build_model()

    # Обучение моделей без сохранения
    lstm_trainer = Trainer(lstm_model, X_train, Y_train)
    lstm_trainer.train()

    dense_trainer = Trainer(dense_model, X_train, Y_train)
    dense_trainer.train()

    # Сохранение обученных моделей
    ModelManager.save_model(lstm_model, 'sets/lstm_model.h5')
    ModelManager.save_model(dense_model, 'sets/dense_model.h5')

if __name__ == '__main__':
    main()