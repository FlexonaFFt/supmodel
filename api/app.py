from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import ModelManager, Normalizer, DataLoader, Predictor, DataProcessor # type: ignore
import tensorflow.keras.losses as losses # type: ignore
import numpy as np

app = FastAPI()
LSTM_MODEL_PATH = 'models/lstm_model.h5'
DENSE_MODEL_PATH = 'models/dense_model.h5'
DATA_FILE = 'data/dataset_.csv'
FEATURES = ['theme_id', 'category_id', 'comp_idx',
    'start_m', 'investments_m', 'crowdfunding_m',
    'team_idx', 'tech_idx', 'social_idx', 'demand_idx']
TARGETS = ['social_idx', 'investments_m', 'crowdfunding_m',
    'demand_idx', 'comp_idx']

data_loader, normalizer, processor = DataLoader(DATA_FILE, FEATURES, TARGETS), Normalizer(), DataProcessor()
x_train, y_train = data_loader.get_features_and_targets()
x_scaled, y_scaled = normalizer.fit_transform(x_train, y_train)
lstm_model = ModelManager.load_model(LSTM_MODEL_PATH, custom_objects={'mse': losses.MeanSquaredError})
dense_model = ModelManager.load_model(DENSE_MODEL_PATH, custom_objects={'mse': losses.MeanSquaredError})
lstm_predictor = Predictor(lstm_model, normalizer.scaler_Y)

class PredictionRequest(BaseModel):
    data: list[float]

class TimeSeriesPredictionRequest(BaseModel):
    data: list[float]
    steps: int

# Маршрут для предсказания с LSTM модели
@app.post("/predict/lstm")
async def predict_lstm(request: PredictionRequest):
    new_data = np.array([request.data])
    try:
        new_data_scaled = normalizer.scaler_X.transform(new_data)
        new_data_lstm = new_data_scaled.reshape((new_data_scaled.shape[0], new_data_scaled.shape[1], 1))
        lstm_prediction = lstm_model.predict(new_data_lstm)
        lstm_prediction_inverse = normalizer.inverse_transform_Y(lstm_prediction)

        return {
            'prediction': lstm_prediction_inverse.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Маршрут для предсказания с Dense модели
@app.post("/predict/dense")
async def predict_dense(request: PredictionRequest):
    new_data = np.array([request.data])
    try:
        new_data_scaled = normalizer.scaler_X.transform(new_data)
        dense_prediction = dense_model.predict(new_data_scaled)
        dense_prediction_inverse = normalizer.inverse_transform_Y(dense_prediction)

        return {
            'prediction': dense_prediction_inverse.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/timeseries")
async def predict_timeseries(request: TimeSeriesPredictionRequest):
    # Преобразуем входные данные в массив (размер (1, 10))
    initial_input = np.array(request.data).reshape((1, len(request.data)))
    print(f"Initial input shape: {initial_input.shape}")  # Выводим форму данных

    try:
        # Масштабируем данные (размер должен остаться (1, 10))
        new_data_scaled = normalizer.scaler_X.transform(initial_input)
        print(f"Scaled input shape: {new_data_scaled.shape}")  # Выводим форму масштабированных данных

        # Преобразуем данные в трёхмерный массив для подачи в LSTM
        initial_input_scaled = new_data_scaled.reshape((1, 1, len(request.data)))  # (1, 1, 10)
        print(f"Reshaped input for LSTM: {initial_input_scaled.shape}")  # Выводим форму данных для LSTM

        steps = request.steps
        # Генерируем предсказания
        future_predictions = lstm_predictor.make_predictions4(initial_input_scaled, steps)

        return {
            'predictions': future_predictions.tolist()
        }
    except Exception as e:
        print(f"Error occurred: {str(e)}")  # Выводим текст ошибки
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
