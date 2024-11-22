from rest_framework import serializers
from .models import (
    UserInputData,
    Project,
    LSTMPrediction,
    LSTMTimePrediction,
    SyntheticPrediction,
    SyntheticTimePrediction,
)

class UserInputDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserInputData
        fields = '__all__'

class ProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = Project
        fields = '__all__'

class LSTMPredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = LSTMPrediction
        fields = '__all__'

class LSTMTimePredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = LSTMTimePrediction
        fields = '__all__'

class SyntheticPredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = SyntheticPrediction
        fields = '__all__'

class SyntheticTimePredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = SyntheticTimePrediction
        fields = '__all__'

class FullProjectSerializer(serializers.ModelSerializer):
    user_input = UserInputDataSerializer()
    lstm_predictions = LSTMPredictionSerializer(many=True)
    lstm_time_predictions = LSTMTimePredictionSerializer(many=True)
    synthetic_predictions = SyntheticPredictionSerializer(many=True)
    synthetic_time_predictions = SyntheticTimePredictionSerializer(many=True)

    class Meta:
        model = Project
        fields = '__all__'
