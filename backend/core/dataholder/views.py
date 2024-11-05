from django.shortcuts import render
from rest_framework import viewsets
from .models import UserInputData, Projects, ModelPredictions
from .serializers import UserInputDataSerializer, ProjectsSerializer, ModelPredictionsSerializer

class UserInputDataViewSet(viewsets.ModelViewSet):
    queryset = UserInputData.objects.all() # type: ignore
    serializer_class = UserInputDataSerializer

class ProjectsViewSet(viewsets.ModelViewSet):
    queryset = Projects.objects.all() # type: ignore
    serializer_class = ProjectsSerializer

class ModelPredictionsViewSet(viewsets.ModelViewSet):
    queryset = ModelPredictions.objects.all() # type: ignore
    serializer_class = ModelPredictionsSerializer
