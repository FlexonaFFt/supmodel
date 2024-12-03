import numpy as np
from .model_utils import ModelManager, Normalizer, DataLoader, Predictor, DataProcessor
import tensorflow.keras.losses as losses # type: ignore

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
FULL_PROJECTS_DATA_URL = f"{DJANGO_API_BASE_URL}/project-data/"
INDECES_URL = f"{DJANGO_API_BASE_URL}/indeces/"
LSTM_PREDICTIONS_URL = f"{DJANGO_API_BASE_URL}/lstm-predictions/"
LSTM_TIME_PREDICTIONS_URL = f"{DJANGO_API_BASE_URL}/lstm-time-predictions/"
SYNTHETIC_PREDICTIONS_URL = f"{DJANGO_API_BASE_URL}/synthetic-predictions/"
SYNTHETIC_TIME_PREDICTIONS_URL = f"{DJANGO_API_BASE_URL}/synthetic-time-predictions/"

data_loader, normalizer, processor = DataLoader(DATA_FILE, FEATURES, TARGETS), Normalizer(), DataProcessor()
x_train, y_train = data_loader.get_features_and_targets()
x_scaled, y_scaled = normalizer.fit_transform(x_train, y_train)
synth_lstm_model = ModelManager.load_model(SYNTH_LSTM_MODEL_PATH, custom_objects={'mse': losses.MeanSquaredError})
lstm_model = ModelManager.load_model(LSTM_MODEL_PATH, custom_objects={'mse': losses.MeanSquaredError})
dense_model = ModelManager.load_model(DENSE_MODEL_PATH, custom_objects={'mse': losses.MeanSquaredError})
lstm_predictor = Predictor(lstm_model, normalizer.scaler_Y)

def calculate_team_idx(team_desc: str, experience_years: int, team_size: int) -> float:
    team_desc = team_desc.lower()
    team_mapping = {"новички": 2, "средний опыт": 5, "эксперты": 8}
    base_score = team_mapping.get(team_desc, 0)
    raw_score = (0.6 * experience_years + 0.4 * team_size) + base_score
    return max(1.0, min(9.9, round(raw_score / 3, 1)))

def calculate_tech_idx(tech_level: str, tech_investment: int) -> float:
    tech_level = tech_level.lower()
    tech_mapping = {"низкий": 2, "средний": 5, "высокий": 8}
    base_score = tech_mapping.get(tech_level, 0)
    raw_score = (0.7 * (tech_investment / 10) + 0.3 * base_score)
    return max(1.0, min(9.9, round(raw_score, 1)))

def calculate_comp_idx(comp_level: str, competitors: int) -> float:
    comp_level = comp_level.lower()
    comp_mapping = {"низкая конкуренция": 8, "средняя конкуренция": 5, "высокая конкуренция": 2}
    base_score = comp_mapping.get(comp_level, 0)
    raw_score = base_score - min(competitors / 10, base_score - 1)
    return max(1.0, min(9.9, round(raw_score, 1)))

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
    return max(1.0, min(9.9, round(raw_score, 1)))

def calculate_indices(form_data):
    team_idx = calculate_team_idx(form_data["team_mapping"], form_data["team_index"], form_data["team_size"])
    tech_idx = calculate_tech_idx(form_data["tech_level"], form_data["tech_investment"])
    comp_idx = calculate_comp_idx(form_data["competition_level"], form_data["competitor_count"])
    social_idx = calculate_social_idx(form_data["social_impact"])
    demand_idx = calculate_demand_idx(form_data["demand_level"], form_data["audience_reach"], form_data["market_size"])
    return [team_idx, tech_idx, comp_idx, social_idx, demand_idx]
