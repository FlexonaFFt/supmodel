from sklearn.model_selection import train_test_split
from model import DataPreprocessor, LSTMModel, Predictor #type: ignore
import numpy as np
import pandas as pd
import tensorflow as tf

def main():
    data_path = 'data/maxextend.csv'
    features = ['theme_id', 'category_id', 'comp_idx', 'start_m',
                'investments_m', 'crowdfunding_m', 'team_idx',
                'tech_idx', 'social_idx', 'demand_idx']
    targets = ['social_idx', 'investments_m', 'crowdfunding_m', 'demand_idx',
                'comp_idx']

    preprocessor = DataPreprocessor(data_path, features, targets)
    X_scaled, Y_scaled = preprocessor.preprocess()
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)
    X_train_lstm = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1])) # type: ignore
    X_test_lstm = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1])) # type: ignore

    input_shape = (1, X_train_lstm.shape[2])
    lstm_model = LSTMModel(input_shape)
    lstm_model.train(X_train_lstm, Y_train)

    predictor = Predictor(lstm_model.model, preprocessor.scaler_Y)
    initial_input = X_test[0]
    steps = 6
    future_predictions = predictor.make_predictions(initial_input, steps)

    X_test_rescaled = preprocessor.inverse_transform_X(X_test)
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

if __name__ == '__main__':
    main()
