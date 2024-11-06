from django.contrib import admin
from .models import UserInputData, Projects, ModelPredictions

admin.site.register(UserInputData)
admin.site.register(Projects)
admin.site.register(ModelPredictions)
