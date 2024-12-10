from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views
from .views import (
    UserInputDataViewSet,
    ProjectsViewSet,
    LSTMPredictionsViewSet,
    LSTMTimePredictionsViewSet,
    SyntheticPredictionsViewSet,
    SyntheticTimePredictionsViewSet,
    IndecesViewSet,
)

router = DefaultRouter()
router.register(r'user-input-data', UserInputDataViewSet, basename='user-input-data')
router.register(r'indeces', IndecesViewSet, basename='indeces')
router.register(r'projects', ProjectsViewSet, basename='projects')
router.register(r'lstm-predictions', LSTMPredictionsViewSet, basename='lstm-predictions')
router.register(r'lstm-time-predictions', LSTMTimePredictionsViewSet, basename='lstm-time-predictions')
router.register(r'synthetic-predictions', SyntheticPredictionsViewSet, basename='synthetic-predictions')
router.register(r'synthetic-time-predictions', SyntheticTimePredictionsViewSet, basename='synthetic-time-predictions')

urlpatterns = [
    path('', views.standalone, name='standalone'),
    path('api/', include(router.urls)),
    path('checkout', views.checkout, name='checkout'),
    path('loading', views.loading, name='loading'),
    path('projectlist', views.projectlist, name='projectlist'),
    path('project/<int:project_number>/', views.project_detail, name='project_detail'),
    path('api/project-data/<int:project_number>/', views.get_project_data, name='get_project_data'),
]
