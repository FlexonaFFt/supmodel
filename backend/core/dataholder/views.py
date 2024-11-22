from django.shortcuts import render, get_object_or_404
from rest_framework import viewsets
from django.http import JsonResponse
from .models import (
    UserInputData,
    Project,
    LSTMPrediction,
    LSTMTimePrediction,
    SyntheticPrediction,
    SyntheticTimePrediction,
)
from .serializers import (
    UserInputDataSerializer,
    ProjectSerializer,
    LSTMPredictionSerializer,
    LSTMTimePredictionSerializer,
    SyntheticPredictionSerializer,
    SyntheticTimePredictionSerializer
)

class UserInputDataViewSet(viewsets.ModelViewSet):
    queryset = UserInputData.objects.all() # type: ignore
    serializer_class = UserInputDataSerializer

class ProjectsViewSet(viewsets.ModelViewSet):
    queryset = Project.objects.all() # type: ignore
    serializer_class = ProjectSerializer

class LSTMPredictionsViewSet(viewsets.ModelViewSet):
    queryset = LSTMPrediction.objects.all() # type: ignore
    serializer_class = LSTMPredictionSerializer

class LSTMTimePredictionsViewSet(viewsets.ModelViewSet):
    queryset = LSTMTimePrediction.objects.all() # type: ignore
    serializer_class = LSTMTimePredictionSerializer

class SyntheticPredictionsViewSet(viewsets.ModelViewSet):
    queryset = SyntheticPrediction.objects.all() # type: ignore
    serializer_class = SyntheticPredictionSerializer

class SyntheticTimePredictionsViewSet(viewsets.ModelViewSet):
    queryset = SyntheticTimePrediction.objects.all() # type: ignore
    serializer_class = SyntheticTimePredictionSerializer

def project_detail(request, project_number):
    project = get_object_or_404(Project, project_number=project_number)
    return render(request, 'project.html', {'project': project})

def get_project_data(request, project_number):
    project = get_object_or_404(Project, project_number=project_number)
    predictions = LSTMPrediction.objects.filter(project=project) # type: ignore

    # Пример данных для графиков
    investments_data = [pred.predicted_investments_m for pred in predictions]
    crowdfunding_data = [pred.predicted_crowdfunding_m for pred in predictions]

    data = {
        'investments_data': investments_data,
        'crowdfunding_data': crowdfunding_data,
    }
    return JsonResponse(data)
