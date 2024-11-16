from django.contrib import admin
from .models import (
    UserInputData,
    Project,
    LSTMPrediction,
    LSTMTimePrediction,
    SyntheticPrediction,
    SyntheticTimePrediction,
)

admin.site.register(UserInputData)
admin.site.register(Project)
admin.site.register(LSTMPrediction)
admin.site.register(LSTMTimePrediction)
admin.site.register(SyntheticPrediction)
admin.site.register(SyntheticTimePrediction)
