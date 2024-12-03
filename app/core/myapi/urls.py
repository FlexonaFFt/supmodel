from django.urls import path
from .views import PredictAllFullFormView

urlpatterns = [
    path('predict/all_full_form/', PredictAllFullFormView.as_view(), name='predict_all_full_form'),
]
