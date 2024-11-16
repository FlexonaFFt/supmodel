from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    UserInputDataViewSet,
    ProjectsViewSet,
    LSTMPredictionsViewSet,
    LSTMTimePredictionsViewSet,
    SyntheticPredictionsViewSet,
    SyntheticTimePredictionsViewSet,
)

router = DefaultRouter()
router.register(r'user-input-data', UserInputDataViewSet, basename='user-input-data')
router.register(r'projects', ProjectsViewSet, basename='projects')
router.register(r'lstm-predictions', LSTMPredictionsViewSet, basename='lstm-predictions')
router.register(r'lstm-time-predictions', LSTMTimePredictionsViewSet, basename='lstm-time-predictions')
router.register(r'synthetic-predictions', SyntheticPredictionsViewSet, basename='synthetic-predictions')
router.register(r'synthetic-time-predictions', SyntheticTimePredictionsViewSet, basename='synthetic-time-predictions')

urlpatterns = [
    path('', include(router.urls)),
]
