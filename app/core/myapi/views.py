# myapi/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from app.api.app import API_BASE_URL, FULL_PROJECTS_DATA_URL, PROJECTS_URL, USER_INPUT_DATA_URL
from .serializers import FullFormRequestSerializer
import numpy as np
import random
import httpx
from model import ModelManager, Normalizer, DataLoader, Predictor, DataProcessor # type: ignore
import tensorflow.keras.losses as losses # type: ignore
import os

# Импортируйте ваши функции и модели
from .utils import DJANGO_API_BASE_URL, LSTM_PREDICTIONS_URL, LSTM_TIME_PREDICTIONS_URL, calculate_indices, LSTM_MODEL_PATH, DENSE_MODEL_PATH, SYNTH_LSTM_MODEL_PATH, DATA_FILE, FEATURES, TARGETS, API_BASE_URL, DJANGO_API_BASE_URL, USER_INPUT_DATA_URL, PROJECTS_URL, FULL_PROJECTS_DATA_URL, INDECES_URL , LSTM_PREDICTIONS_URL, LSTM_PREDICTIONS_URL, LSTM_TIME_PREDICTIONS_URL, SYNTHETIC_PREDICTIONS_URL , SYNTHETIC_TIME_PREDICTIONS_URL

class PredictAllFullFormView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = FullFormRequestSerializer(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            try:
                indices = calculate_indices(data)
                user_input_data = {
                    "startup_name": data["startup_name"], "team_name": data["team_name"], "theme_id": data["theme_id"],
                    "category_id": data["category_id"], "description": data["description"], "start_m": data["start_m"],
                    "investments_m": data["investments_m"], "crowdfunding_m": data["crowdfunding_m"], "team_mapping": data["team_mapping"],
                    "team_size": data["team_size"], "team_index": data["team_index"], "tech_level": data["tech_level"],
                    "tech_investment": data["tech_investment"], "competition_level": data["competition_level"],
                    "competitor_count": data["competitor_count"], "social_impact": data["social_impact"],
                    "demand_level": data["demand_level"], "audience_reach": data["audience_reach"],
                    "market_size": data["market_size"],
                }

                with httpx.Client() as client:
                    response = client.post(USER_INPUT_DATA_URL, json=user_input_data)
                    if response.status_code != 201:
                        return Response({"detail": response.text}, status=response.status_code)
                    user_input_id = response.json().get("id")

                new_data = np.array([[
                    data["theme_id"], data["category_id"], indices[0],
                    data["start_m"], data["investments_m"],
                    data["crowdfunding_m"],
                    indices[0], indices[1], indices[2], indices[4]
                ]])

                with httpx.Client() as client:
                    project_response = client.post(PROJECTS_URL, json={
                        "project_name": data["startup_name"],
                        "description": data["description"],
                        "user_input": user_input_id,
                        "project_number": data.get("project_number", random.randint(600000, 699999)),
                        "is_public": True
                    })
                    if project_response.status_code != 201:
                        return Response({"detail": project_response.text}, status=project_response.status_code)
                    project_id = project_response.json()["id"]

                with httpx.Client() as client:
                    indeces_response = client.post(INDECES_URL, json={
                        "project": project_id,
                        "competition_idx": indices[0],
                        "team_idx": indices[1],
                        "tech_idx": indices[2],
                        "social_idx": indices[3],
                        "demand_idx": indices[4],
                    })
                    if indeces_response.status_code != 201:
                        return Response({"detail": indeces_response.text}, status=indeces_response.status_code)

                # LSTMPrediction
                new_data_scaled = normalizer.scaler_X.transform(new_data)
                new_data_lstm = new_data_scaled.reshape((new_data_scaled.shape[0], new_data_scaled.shape[1], 1))
                prediction = lstm_model.predict(new_data_lstm)
                prediction_inverse = normalizer.inverse_transform_Y(prediction)

                lstm_prediction_data = {
                    "project": project_id,
                    "predicted_social_idx": float(prediction_inverse[0][0]),
                    "predicted_investments_m": float(prediction_inverse[0][1]),
                    "predicted_crowdfunding_m": float(prediction_inverse[0][2]),
                    "predicted_demand_idx": float(prediction_inverse[0][3]),
                    "predicted_comp_idx": float(prediction_inverse[0][4])
                }

                with httpx.Client() as client:
                    lstm_prediction_response = client.post(LSTM_PREDICTIONS_URL, json=lstm_prediction_data)
                    if lstm_prediction_response.status_code != 201:
                        return Response({"detail": lstm_prediction_response.text}, status=lstm_prediction_response.status_code)

                # LSTMTimePrediction
                new_data_scaled_two = normalizer.scaler_X.transform(new_data)
                new_data_lstm_two = new_data_scaled_two.reshape((new_data_scaled_two.shape[0], new_data_scaled_two.shape[1], 1))

                predictions_two = []
                pred = synth_lstm_model.predict(new_data_lstm_two)
                predictions_two.append(normalizer.inverse_transform_Y(pred).flatten().tolist())

                for step in range(1, 5):
                    current_input = np.concatenate([new_data_scaled_two.flatten()[-5:], pred.flatten()]).reshape(1, 10, 1)
                    pred = synth_lstm_model.predict(current_input)
                    predictions_two.append(normalizer.inverse_transform_Y(pred).flatten().tolist())

                for pred in predictions_two:
                    lstm_time_prediction_data = {
                        "project": project_id,
                        "predicted_social_idx": float(pred[0]),
                        "predicted_investments_m": float(pred[1]),
                        "predicted_crowdfunding_m": float(pred[2]),
                        "predicted_demand_idx": float(pred[3]),
                        "predicted_comp_idx": float(pred[4])
                    }
                    with httpx.Client() as client:
                        lstm_time_prediction_response = client.post(LSTM_TIME_PREDICTIONS_URL, json=lstm_time_prediction_data)
                        if lstm_time_prediction_response.status_code != 201:
                            return Response({"detail": lstm_time_prediction_response.text}, status=lstm_time_prediction_response.status_code)

                # SyntheticPrediction
                new_data_scaled_three = normalizer.scaler_X.transform(new_data)
                new_data_lstm_three = new_data_scaled_three.reshape((new_data_scaled_three.shape[0], new_data_scaled_three.shape[1], 1))

                lstm_prediction_three = synth_lstm_model.predict(new_data_lstm_three)
                lstm_prediction_inverse_three = normalizer.inverse_transform_Y(lstm_prediction_three)

                synthetic_prediction_data = {
                    "project": project_id,
                    "predicted_social_idx": float(lstm_prediction_inverse_three[0][0]),
                    "predicted_investments_m": float(lstm_prediction_inverse_three[0][1]),
                    "predicted_crowdfunding_m": float(lstm_prediction_inverse_three[0][2]),
                    "predicted_demand_idx": float(lstm_prediction_inverse_three[0][3]),
                    "predicted_comp_idx": float(lstm_prediction_inverse_three[0][4])
                }

                with httpx.Client() as client:
                    synthetic_prediction_response = client.post(SYNTHETIC_PREDICTIONS_URL, json=synthetic_prediction_data)
                    if synthetic_prediction_response.status_code != 201:
                        return Response({"detail": synthetic_prediction_response.text}, status=synthetic_prediction_response.status_code)

                # SyntheticTimePrediction
                new_data_scaled_four = normalizer.scaler_X.transform(new_data)
                new_data_lstm_four = new_data_scaled_four.reshape((new_data_scaled_four.shape[0], new_data_scaled_four.shape[1], 1))

                predictions_four = []
                pred_four = synth_lstm_model.predict(new_data_lstm_four)
                predictions_four.append(normalizer.inverse_transform_Y(pred_four).flatten())

                for step in range(1, 5):
                    current_input_four = np.concatenate([new_data_scaled_four.flatten()[:5], pred_four.flatten()]).reshape((1, 10, 1))
                    pred_four = synth_lstm_model.predict(current_input_four)
                    predictions_four.append(normalizer.inverse_transform_Y(pred_four).flatten())

                for pred in predictions_four:
                    synthetic_time_prediction_data = {
                        "project": project_id,
                        "predicted_social_idx": float(pred[0]),
                        "predicted_investments_m": float(pred[1]),
                        "predicted_crowdfunding_m": float(pred[2]),
                        "predicted_demand_idx": float(pred[3]),
                        "predicted_comp_idx": float(pred[4])
                    }
                    with httpx.Client() as client:
                        synthetic_time_prediction_response = client.post(SYNTHETIC_TIME_PREDICTIONS_URL, json=synthetic_time_prediction_data)
                        if synthetic_time_prediction_response.status_code != 201:
                            return Response({"detail": synthetic_time_prediction_response.text}, status=synthetic_time_prediction_response.status_code)

                return Response({
                    "project_id": project_id,
                    "data": new_data.tolist(),
                    "indeces": np.array(indices).tolist(),
                    "LSTMPrediction": prediction_inverse.tolist(),
                    "LSTMTimePrediction": np.array(predictions_two).tolist(),
                    "SyntheticPredictions": lstm_prediction_inverse_three.tolist(),
                    "SyntheticTimePrediction": np.array(predictions_four).tolist(),
                }, status=status.HTTP_200_OK)

            except httpx.HTTPStatusError as http_err:
                return Response({"detail": str(http_err)}, status=http_err.response.status_code)

            except Exception as e:
                return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class GetProjectNumberView(APIView):
    def get(self, request, project_id, *args, **kwargs):
        with httpx.Client() as client:
            project_number_response = client.get(f"{DJANGO_API_BASE_URL}/projects/{project_id}/")
            if project_number_response.status_code != 200:
                return Response({"detail": project_number_response.text}, status=project_number_response.status_code)

            project_data = project_number_response.json()
            project_number = project_data.get("project_number")

            if project_number is None:
                return Response({"detail": "project_number is None in the response"}, status=status.HTTP_404_NOT_FOUND)

            return Response({"project_number": project_number}, status=status.HTTP_200_OK)
