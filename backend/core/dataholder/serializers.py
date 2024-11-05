from rest_framework import serializers
from .models import UserInputData, Projects, ModelPredictions

class UserInputDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserInputData
        fields = '__all__'

class ProjectsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Projects
        fields = '__all__'

class ModelPredictionsSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelPredictions
        fields = '__all__'
