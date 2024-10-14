from sklearn.model_selection import train_test_split
from model import DataLoader, Normalizer, DataProcessor, DenseModelBuilder # type: ignore
from model import LSTMModelBuilder, Trainer, Predictor # type: ignore
import numpy as np
import pandas as pd
import tensorflow as tf

def variant():
    variant, variable = int(input("[1, 2]: ")), True
    if variant == 1:
        variable = True
    elif variant == 2:
        variable = False
    return variable

def predictfunc(variable):
    file_path = 'data/dataset_.csv'
    features = ['theme_id', 'category_id', 'comp_idx', 'start_m',
                'investments_m', 'crowdfunding_m', 'team_idx',
                'tech_idx', 'social_idx', 'demand_idx']
    targets = ['social_idx', 'investments_m', 'crowdfunding_m', 'demand_idx',
                'comp_idx']

    data_loader = DataLoader(file_path, features, targets)
    X, Y = data_loader.get_features_and_targets()
    normalizer, processor = Normalizer(), DataProcessor()
    X_scaled, Y_scaled = normalizer.fit_transform(X, Y)
    X_train, X_test, Y_train, Y_test = processor.split_data(X_scaled, Y_scaled)
    X_train_lstm = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    if variable:
        var = True
        lstm_builder = LSTMModelBuilder(input_shape=(1, X_train_lstm.shape[2]))
        lstm_model = lstm_builder.build_model()
        trainer = Trainer(lstm_model, X_train_lstm, Y_train)
        trainer.train()

        predictor = Predictor(lstm_model, normalizer.scaler_Y)
        initial_input, steps = X_test[0], 5
        predictions = predictor.make_predictions(initial_input, steps)
        #if predictions.ndim == 3:
            #predictions = predictions.squeeze(axis=1)
    else:
        var = False
        dense_builder = DenseModelBuilder(input_shape=(X_train.shape[1],))
        model = dense_builder.build_model()
        trainer = Trainer(model, X_train, Y_train)
        trainer.train()
        predictions = model.predict(X_test)
        predictions_original = normalizer.inverse_transform_Y(predictions)
        random_index = np.random.randint(0, X_test.shape[0])
        random_input = X_test[random_index].reshape(1, -1)
        single_prediction = model.predict(random_input)
        single_prediction_original = normalizer.inverse_transform_Y(single_prediction)

    return predictions, var

def main():
    variable = variant()
    prediction, var = predictfunc(variable)
    if var:
        for i, pred in enumerate(prediction):
            print(f"Prediction {i+1}:")
            print(f"  Social Index: {pred[0, 0]:.2f}")
            print(f"  Future Investments: {pred[0, 1]:.2f}")
            print(f"  Future Crowdfunding: {pred[0, 2]:.2f}")
            print(f"  Future Demand: {pred[0, 3]:.2f}")
            print(f"  Competition Index: {pred[0, 4]:.2f}")
            print()
    else:
        for i, pred in enumerate(prediction):
            print(f"Prediction {i+1}:")
            print(f"  Social Index: {pred[0]:.2f}")
            print(f"  Future Investments: {pred[1]:.2f}")
            print(f"  Future Crowdfunding: {pred[2]:.2f}")
            print(f"  Future Demand: {pred[3]:.2f}")
            print(f"  Competition Index: {pred[4]:.2f}")
            print()

if __name__ == '__main__':
    main()
