from sklearn.model_selection import train_test_split
from model import DataLoader, Normalizer, DataProcessor # type: ignore
from model import LSTMModelBuilder, Trainer, Predictor # type: ignore
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
    normalizer = Normalizer()
    X_scaled, Y_scaled = normalizer.fit_transform(X, Y)
    processor = DataProcessor()
    X_train, X_test, Y_train, Y_test = processor.split_data(X_scaled, Y_scaled)
    X_train_lstm = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    lstm_builder = LSTMModelBuilder(input_shape=(1, X_train_lstm.shape[2]))
    lstm_model = lstm_builder.build_model()
    trainer = Trainer(lstm_model, X_train_lstm, Y_train)
    trainer.train()

    predictor = Predictor(lstm_model, normalizer.scaler_Y)
    function = int(input('1, 2: '))
    if function == 1:
        # initial_input = X_test[0]
        _input = [1.00, 3.00, 3.0, 5900, 29050, 35000, 7.0, 8.30, 5.4, 6.40]
        #initial_input = np.array(_input).reshape((1, 1, len(_input)))
        initial_input = np.array(_input).reshape((1, len(_input)))
        initial_input = normalizer.scaler_X.transform(initial_input)
        steps = 3
        future_predictions = predictor.make_predictions2(initial_input, steps)
        X_test_rescaled = normalizer.inverse_transform_X(X_test)
        print(f"  Input Data:")
        print(f"    Theme ID: {X_test_rescaled[0, 0]:.2f}")
        print(f"    Category ID: {X_test_rescaled[0, 1]:.2f}")
        print(f"    Start Money: {X_test_rescaled[0, 2]:.2f}")
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
    if function == 2:
        initial_input = X_test[0]
        steps = 6
        future_predictions = predictor.make_predictions(initial_input, steps)
        X_test_rescaled = normalizer.inverse_transform_X(X_test)
        print(f"  Input Data:")
        print(f"    Theme ID: {X_test_rescaled[0, 0]:.2f}")
        print(f"    Category ID: {X_test_rescaled[0, 1]:.2f}")
        print(f"    Start Money: {X_test_rescaled[0, 2]:.2f}")
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

if __name__ == '__main__':
    main()
