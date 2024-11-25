from django.shortcuts import render, get_object_or_404
from rest_framework.decorators import api_view
from rest_framework.views import Response
from rest_framework import status
from rest_framework import viewsets
from django.http import JsonResponse
from .models import (
    UserInputData,
    Project,
    LSTMPrediction,
    LSTMTimePrediction,
    SyntheticPrediction,
    SyntheticTimePrediction,
    Indeces,
)
from .serializers import (
    UserInputDataSerializer,
    ProjectSerializer,
    LSTMPredictionSerializer,
    LSTMTimePredictionSerializer,
    SyntheticPredictionSerializer,
    SyntheticTimePredictionSerializer,
    FullProjectSerializer,
    IndecesSerializer,
)

class UserInputDataViewSet(viewsets.ModelViewSet):
    queryset = UserInputData.objects.all() # type: ignore
    serializer_class = UserInputDataSerializer

class IndecesViewSet(viewsets.ModelViewSet):
    queryset = Indeces.objects.all() # type: ignore
    serializer_class = IndecesSerializer

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
    context = {
        'project': project,
    }
    return render(request, 'project.html', {'project': project})

@api_view(['GET'])
def get_project_data(request, project_number):
    try:
        project = Project.objects.get(project_number=project_number) # type: ignore
    except Project.DoesNotExist: # type: ignore
        return Response(status=status.HTTP_404_NOT_FOUND)

    serializer = FullProjectSerializer(project)
    return Response(serializer.data)
