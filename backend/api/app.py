from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Union
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

class FullFormRequest(BaseModel):
    startup_name: str
    team_name: str
    theme_id: int
    category_id: int
    description: str
    start_m: int
    investments_m: int
    crowdfunding_m: int
    team_mapping: str
    team_size: int
    team_index: int
    tech_level: str
    tech_investment: int
    competition_level: str
    competitor_count: int
    social_impact: str
    demand_level: str
    audience_reach: int
    market_size: int

# Функции для расчета индексов
def calculate_team_idx(team_desc: str, experience_years: int, team_size: int) -> float:
    team_mapping = {"новички": 2, "средний опыт": 5, "эксперты": 8}
    base_score = team_mapping.get(team_desc, 0)
    raw_score = (0.6 * experience_years + 0.4 * team_size) + base_score
    return max(1.0, min(9.99, round(raw_score / 3, 1)))

def calculate_tech_idx(tech_level: str, tech_investment: int) -> float:
    tech_mapping = {"низкий": 2, "средний": 5, "высокий": 8}
    base_score = tech_mapping.get(tech_level, 0)
    raw_score = (0.7 * (tech_investment / 1_000_000) + 0.3 * base_score)
    return max(1.0, min(9.99, round(raw_score, 1)))

def calculate_comp_idx(comp_level: str, competitors: int) -> float:
    comp_mapping = {"низкая конкуренция": 8, "средняя конкуренция": 5, "высокая конкуренция": 2}
    base_score = comp_mapping.get(comp_level, 0)
    raw_score = base_score - min(competitors / 10, base_score - 1)
    return max(1.0, min(9.99, round(raw_score, 1)))

def calculate_social_idx(social_impact_desc: str) -> float:
    social_mapping = {"низкое влияние": 3.0, "среднее влияние": 6.0, "высокое влияние": 9.0}
    return social_mapping.get(social_impact_desc, 1.0)

def calculate_demand_idx(demand_level: str, audience_reach: int, market_size: int) -> float:
    demand_mapping = {"низкий спрос": 2, "средний спрос": 5, "высокий спрос": 8}
    base_score = demand_mapping.get(demand_level, 0)
    scaled_audience = audience_reach / 10_000_000
    scaled_market = market_size / 100_000_000
    raw_score = base_score + scaled_audience + scaled_market
    return max(1.0, min(9.99, round(raw_score, 1)))

def calculate_indices(form_data):
    # Пример вычислений индексов
    team_idx = calculate_team_idx(form_data.team_mapping, form_data.team_size, form_data.team_index)
    tech_idx = calculate_tech_idx(form_data.tech_level, form_data.tech_investment)
    comp_idx = calculate_comp_idx(form_data.competition_level, form_data.competitor_count)
    social_idx = calculate_social_idx(form_data.social_impact)
    demand_idx = calculate_demand_idx(form_data.demand_level, form_data.audience_reach, form_data.market_size)

    return [team_idx, tech_idx, comp_idx, social_idx, demand_idx]

# Настройки CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Здесь можно указать список разрешённых доменов
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Маршрут для предобработки и предсказания
@app.post("/predict/full_form")
async def predict_full_form(request: FullFormRequest):
    try:
        indices = calculate_indices(request)
        # Собираем все данные для предсказания, включая вычисленные индексы
        data = [
            request.theme_id, request.category_id, indices[2], # comp_idx
            request.start_m, request.investments_m, request.crowdfunding_m,
            indices[0], indices[1], indices[3], indices[4]  # team_idx, tech_idx, social_idx, demand_idx
        ]
        new_data = np.array([data])
        new_data_scaled = normalizer.scaler_X.transform(new_data)

        # Выбираем нужную модель для предсказания (например, LSTM)
        new_data_lstm = new_data_scaled.reshape((new_data_scaled.shape[0], new_data_scaled.shape[1], 1))
        prediction = lstm_model.predict(new_data_lstm)
        prediction_inverse = normalizer.inverse_transform_Y(prediction)

        return {
            'prediction': prediction_inverse.tolist(),
            'calculate_indices': indices
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

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
    new_data = np.array([request.data])
    try:
        # Масштабируем данные
        new_data_scaled = normalizer.scaler_X.transform(new_data)
        # Преобразуем данные в трехмерный массив для LSTM
        new_data_lstm = new_data_scaled.reshape((new_data_scaled.shape[0], new_data_scaled.shape[1], 1))

        # Генерация предсказаний
        predictions = []
        pred = lstm_model.predict(new_data_lstm)
        predictions.append(normalizer.inverse_transform_Y(pred).flatten())

        for step in range(1, request.steps):
            current_input = np.concatenate([new_data_scaled.flatten()[:5], pred.flatten()]).reshape((1, 10, 1))
            pred = lstm_model.predict(current_input)
            predictions.append(normalizer.inverse_transform_Y(pred).flatten())

        return {
            'predictions': np.array(predictions).tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
