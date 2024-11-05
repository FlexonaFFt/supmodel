from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import UserInputDataViewSet, ProjectsViewSet, ModelPredictionsViewSet

router = DefaultRouter()
router.register(r'user-input-data', UserInputDataViewSet)
router.register(r'projects', ProjectsViewSet)
router.register(r'model-predictions', ModelPredictionsViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
