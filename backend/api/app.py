from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Union
from pydantic import BaseModel
from model import ModelManager, Normalizer, DataLoader, Predictor, DataProcessor # type: ignore
import tensorflow.keras.losses as losses # type: ignore
import numpy as np
import random
import httpx

app = FastAPI()
LSTM_MODEL_PATH = 'models/lstm_model.h5'
DENSE_MODEL_PATH = 'models/dense_model.h5'
SYNTH_LSTM_MODEL_PATH = 'models/synth_lstm_model.h5'
DATA_FILE = 'data/dataset_.csv'
FEATURES = ['theme_id', 'category_id', 'comp_idx',
    'start_m', 'investments_m', 'crowdfunding_m',
    'team_idx', 'tech_idx', 'social_idx', 'demand_idx']
TARGETS = ['social_idx', 'investments_m', 'crowdfunding_m',
    'demand_idx', 'comp_idx']

API_BASE_URL = "http://localhost:8000/"
DJANGO_API_BASE_URL = "http://localhost:8000/api"
USER_INPUT_DATA_URL = f"{DJANGO_API_BASE_URL}/user-input-data/"
PROJECTS_URL = f"{DJANGO_API_BASE_URL}/projects/"
MODEL_PREDICTIONS_URL = f"{DJANGO_API_BASE_URL}/model-predictions/"

data_loader, normalizer, processor = DataLoader(DATA_FILE, FEATURES, TARGETS), Normalizer(), DataProcessor()
x_train, y_train = data_loader.get_features_and_targets()
x_scaled, y_scaled = normalizer.fit_transform(x_train, y_train)
synth_lstm_model = ModelManager.load_model(SYNTH_LSTM_MODEL_PATH, custom_objects={'mse': losses.MeanSquaredError})
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
    team_desc = team_desc.lower()
    team_mapping = {"новички": 2, "средний опыт": 5, "эксперты": 8}
    base_score = team_mapping.get(team_desc, 0)
    raw_score = (0.6 * experience_years + 0.4 * team_size) + base_score
    return max(1.0, min(9.99, round(raw_score / 3, 1)))

def calculate_tech_idx(tech_level: str, tech_investment: int) -> float:
    tech_level = tech_level.lower()
    tech_mapping = {"низкий": 2, "средний": 5, "высокий": 8}
    base_score = tech_mapping.get(tech_level, 0)
    raw_score = (0.7 * (tech_investment / 10) + 0.3 * base_score)
    return max(1.0, min(9.99, round(raw_score, 1)))

def calculate_comp_idx(comp_level: str, competitors: int) -> float:
    comp_level = comp_level.lower()
    comp_mapping = {"низкая конкуренция": 8, "средняя конкуренция": 5, "высокая конкуренция": 2}
    base_score = comp_mapping.get(comp_level, 0)
    raw_score = base_score - min(competitors / 10, base_score - 1)
    return max(1.0, min(9.99, round(raw_score, 1)))

def calculate_social_idx(social_impact: str) -> float:
    social_impact = social_impact.lower()
    social_mapping = {"низкое влияние": 3.0, "среднее влияние": 6.0, "высокое влияние": 9.0}
    return social_mapping.get(social_impact, 1.0)

def calculate_demand_idx(demand_level: str, audience_reach: int, market_size: int) -> float:
    demand_level = demand_level.lower()
    demand_mapping = {"низкий спрос": 2, "средний спрос": 5, "высокий спрос": 8}
    base_score = demand_mapping.get(demand_level, 0)
    scaled_audience = audience_reach / 10_000_000
    scaled_market = market_size / 100_000_000
    raw_score = base_score + scaled_audience + scaled_market
    return max(1.0, min(9.99, round(raw_score, 1)))

def calculate_indices(form_data):
    team_idx = calculate_team_idx(form_data.team_mapping, form_data.team_index, form_data.team_size)
    tech_idx = calculate_tech_idx(form_data.tech_level, form_data.tech_investment)
    comp_idx = calculate_comp_idx(form_data.competition_level, form_data.competitor_count)
    social_idx = calculate_social_idx(form_data.social_impact)
    demand_idx = calculate_demand_idx(form_data.demand_level, form_data.audience_reach, form_data.market_size)
    print([team_idx, tech_idx, comp_idx, social_idx, demand_idx])
    return [team_idx, tech_idx, comp_idx, social_idx, demand_idx]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/full_form")
async def predict_full_form(request: FullFormRequest):
    try:
        indices = calculate_indices(request)
        user_input_data = {
            "startup_name": request.startup_name, "team_name": request.team_name, "theme_id": request.theme_id,
            "category_id": request.category_id, "description": request.description, "start_m": request.start_m,
            "investments_m": request.investments_m, "crowdfunding_m": request.crowdfunding_m, "team_mapping": request.team_mapping,
            "team_size": request.team_size, "team_index": indices[0], "tech_level": request.tech_level,
            "tech_investment": request.tech_investment, "competition_level": request.competition_level,
            "competitor_count": request.competitor_count, "social_impact": request.social_impact,
            "demand_level": request.demand_level, "audience_reach": request.audience_reach,
            "market_size": request.market_size,
        }

        async with httpx.AsyncClient(proxies=None) as client:
            response = await client.post(USER_INPUT_DATA_URL, json=user_input_data)
            response.raise_for_status()
            user_input_id = response.json().get("id")

        project_data = {
            "project_name": request.startup_name, "description": request.description, "user_input_data": user_input_id,
            "project_number": request.project_number if hasattr(request, "project_number") else random.randint(600000, 699999), # type: ignore
            "is_public": request.is_public if hasattr(request, "is_public") else True, # type: ignore
        }

        async with httpx.AsyncClient(proxies=None) as client:
            response = await client.post(PROJECTS_URL, json=project_data)
            response.raise_for_status()
            project_id = response.json().get("id")

        new_data = np.array([[
            request.theme_id, request.category_id, indices[2],
            request.start_m, request.investments_m, request.crowdfunding_m,
            indices[0], indices[1], indices[3], indices[4]
        ]])

        new_data_scaled = normalizer.scaler_X.transform(new_data)
        new_data_lstm = new_data_scaled.reshape((new_data_scaled.shape[0], new_data_scaled.shape[1], 1))
        prediction = lstm_model.predict(new_data_lstm)
        prediction_inverse = normalizer.inverse_transform_Y(prediction)

        prediction_data = {
            "project_id": project_id,
            "model_name": "LSTM",
            "predicted_social_idx": round(float(prediction_inverse[0][0]), 2),
            "predicted_investments_m": round(float(prediction_inverse[0][1]), 2),
            "predicted_crowdfunding_m": round(float(prediction_inverse[0][2]), 2),
            "predicted_demand_idx": round(float(prediction_inverse[0][3]), 2),
            "predicted_comp_idx": round(float(prediction_inverse[0][4]), 2)
        }

        prediction_data["project"] = prediction_data.pop("project_id")
        async with httpx.AsyncClient(proxies=None) as client:
            response = await client.post(MODEL_PREDICTIONS_URL, json=prediction_data)
            response.raise_for_status()

        return {
            "prediction": prediction_inverse.tolist(),
            "data": new_data.tolist(),
            "calculated_indices": indices
        }

    except Exception as e:
        print("Error encountered:", str(e))
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/all_full_form")
async def predict_all_full_form(request: FullFormRequest):
    try:
        indices = calculate_indices(request)
        user_input_data = {
            "startup_name": request.startup_name, "team_name": request.team_name, "theme_id": request.theme_id,
            "category_id": request.category_id, "description": request.description, "start_m": request.start_m,
            "investments_m": request.investments_m, "crowdfunding_m": request.crowdfunding_m, "team_mapping": request.team_mapping,
            "team_size": request.team_size, "team_index": indices[0], "tech_level": request.tech_level,
            "tech_investment": request.tech_investment, "competition_level": request.competition_level,
            "competitor_count": request.competitor_count, "social_impact": request.social_impact,
            "demand_level": request.demand_level, "audience_reach": request.audience_reach,
            "market_size": request.market_size,
        }

        new_data = np.array([[
            request.theme_id, request.category_id, indices[2],
            request.start_m, request.investments_m, request.crowdfunding_m,
            indices[0], indices[1], indices[3], indices[4]
        ]])

        # LSTMPrediction
        print("LSTMPrediction")
        new_data_scaled = normalizer.scaler_X.transform(new_data)
        new_data_lstm = new_data_scaled.reshape((new_data_scaled.shape[0], new_data_scaled.shape[1], 1))
        prediction = lstm_model.predict(new_data_lstm)
        prediction_inverse = normalizer.inverse_transform_Y(prediction)

        # LSTMTimePrediction
        print("LSTMTimePrediction")
        new_data_scaled_two = normalizer.scaler_X.transform(new_data)
        new_data_lstm_two = new_data_scaled_two.reshape((new_data_scaled_two.shape[0], new_data_scaled_two.shape[1], 1))

        predictions_two = []
        pred = synth_lstm_model.predict(new_data_lstm_two)
        predictions_two.append(normalizer.inverse_transform_Y(pred).flatten())

        for step in range(1, 5):
            current_input = np.concatenate([new_data_scaled_two.flatten()[:5], pred.flatten()]).reshape((1, 10, 1))
            pred = synth_lstm_model.predict(current_input)
            predictions_two.append(normalizer.inverse_transform_Y(pred).flatten())

        # SyntheticPrediction
        print("SyntheticPrediction")
        new_data_scaled_three = normalizer.scaler_X.transform(new_data)
        new_data_lstm_three = new_data_scaled_three.reshape((new_data_scaled_three.shape[0], new_data_scaled_three.shape[1], 1))
        lstm_prediction_three = synth_lstm_model.predict(new_data_lstm_three)
        lstm_prediction_inverse_three = normalizer.inverse_transform_Y(lstm_prediction_three)

        # SyntheticTimePrediction
        print("SyntheticTimePrediction")
        new_data_scaled_four = normalizer.scaler_X.transform(new_data)
        new_data_lstm_four = new_data_scaled_four.reshape((new_data_scaled_four.shape[0], new_data_scaled_four.shape[1], 1))

        predictions_four = []
        pred_four = synth_lstm_model.predict(new_data_lstm_four)
        predictions_four.append(normalizer.inverse_transform_Y(pred_four).flatten())

        for step in range(1, 5):
            current_input_four = np.concatenate([new_data_scaled_four.flatten()[:5], pred_four.flatten()]).reshape((1, 10, 1))
            pred_four = synth_lstm_model.predict(current_input_four)
            predictions_four.append(normalizer.inverse_transform_Y(pred_four).flatten())

        return {
            "data": new_data.tolist(),
            "LSTMPrediction": prediction_inverse.tolist(),
            "LSTMTimePrediction": np.array(predictions_two).tolist(),
            "SyntheticPredictions": lstm_prediction_inverse_three.tolist(),
            "SyntheticTimePrediction": np.array(predictions_four).tolist(),
        }

    except Exception as e:
        print("Error encountered:", str(e))
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/synthlstm/timeseries")
async def predict_synth_lstm_timeseries(request: FullFormRequest):
    indices = calculate_indices(request)
    new_data = np.array([[
        request.theme_id, request.category_id, indices[2],
        request.start_m, request.investments_m, request.crowdfunding_m,
        indices[0], indices[1], indices[3], indices[4]
    ]])
    try:
        new_data_scaled = normalizer.scaler_X.transform(new_data)
        new_data_lstm = new_data_scaled.reshape((new_data_scaled.shape[0], new_data_scaled.shape[1], 1))

        predictions = []
        pred = synth_lstm_model.predict(new_data_lstm)
        predictions.append(normalizer.inverse_transform_Y(pred).flatten())

        for step in range(1, 5):
            current_input = np.concatenate([new_data_scaled.flatten()[:5], pred.flatten()]).reshape((1, 10, 1))
            pred = synth_lstm_model.predict(current_input)
            predictions.append(normalizer.inverse_transform_Y(pred).flatten())

        return {
            'predictions': np.array(predictions).tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/lstm/timeseries")
async def predict_lstm_timeseries(request: FullFormRequest):
    indices = calculate_indices(request)
    new_data = np.array([[
        request.theme_id, request.category_id, indices[2],
        request.start_m, request.investments_m, request.crowdfunding_m,
        indices[0], indices[1], indices[3], indices[4]
    ]])
    try:
        new_data_scaled = normalizer.scaler_X.transform(new_data)
        new_data_lstm = new_data_scaled.reshape((new_data_scaled.shape[0], new_data_scaled.shape[1], 1))

        predictions = []
        pred = lstm_model.predict(new_data_lstm)
        predictions.append(normalizer.inverse_transform_Y(pred).flatten())

        for step in range(1, 5):
            current_input = np.concatenate([new_data_scaled.flatten()[:5], pred.flatten()]).reshape((1, 10, 1))
            pred = lstm_model.predict(current_input)
            predictions.append(normalizer.inverse_transform_Y(pred).flatten())

        return {
            'predictions': np.array(predictions).tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/synthlstm/prediction")
async def predict_synthlstm(request: PredictionRequest):
    new_data = np.array([request.data])
    try:
        new_data_scaled = normalizer.scaler_X.transform(new_data)
        new_data_lstm = new_data_scaled.reshape((new_data_scaled.shape[0], new_data_scaled.shape[1], 1))
        lstm_prediction = synth_lstm_model.predict(new_data_lstm)
        lstm_prediction_inverse = normalizer.inverse_transform_Y(lstm_prediction)

        return {
            'prediction': lstm_prediction_inverse.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/lstm/prediction")
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

@app.post("/predict/dense/prediction")
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

@app.post("/predict/test/timeseries")
async def predict_timeseries(request: TimeSeriesPredictionRequest):
    new_data = np.array([request.data])
    try:
        new_data_scaled = normalizer.scaler_X.transform(new_data)
        new_data_lstm = new_data_scaled.reshape((new_data_scaled.shape[0], new_data_scaled.shape[1], 1))

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
    uvicorn.run(app, host='127.0.0.1', port=8001)
