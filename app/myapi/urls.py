from django.urls import path
from .views import PredictAllFullFormView, GetProjectNumberView


urlpatterns = [
    path('predict/all_full_form/', PredictAllFullFormView.as_view(), name='predict_all_full_form'),
    path('project/<int:project_id>/', GetProjectNumberView.as_view(), name='get_project_number'),
]
