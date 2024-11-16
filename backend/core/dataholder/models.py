from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

class UserInputData(models.Model):
    startup_name = models.TextField()
    team_name = models.TextField()
    theme_id = models.IntegerField()
    category_id = models.IntegerField()
    description = models.TextField()
    start_m = models.IntegerField()
    investments_m = models.IntegerField()
    crowdfunding_m = models.IntegerField()
    team_mapping = models.TextField()
    team_size = models.IntegerField()
    team_index = models.IntegerField()
    tech_level = models.TextField()
    tech_investment = models.IntegerField()
    competition_level = models.TextField()
    competitor_count = models.IntegerField()
    social_impact = models.TextField()
    demand_level = models.TextField()
    audience_reach = models.IntegerField()
    market_size = models.IntegerField()


class Project(models.Model):
    project_name = models.TextField()
    description = models.TextField()
    user_input = models.ForeignKey(UserInputData, on_delete=models.CASCADE)
    project_number = models.IntegerField(
        null=True,
        blank=True,
        validators=[
            MinValueValidator(600000),
            MaxValueValidator(699999)
        ]

    )
    is_public = models.BooleanField(default=True) # type: ignore


class LSTMPrediction(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    predicted_social_idx = models.FloatField()
    predicted_investments_m = models.FloatField()
    predicted_crowdfunding_m = models.FloatField()
    predicted_demand_idx = models.FloatField()
    predicted_comp_idx = models.FloatField()
    prediction_date = models.DateTimeField(auto_now_add=True)


class LSTMTimePrediction(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    predicted_social_idx = models.FloatField()
    predicted_investments_m = models.FloatField()
    predicted_crowdfunding_m = models.FloatField()
    predicted_demand_idx = models.FloatField()
    predicted_comp_idx = models.FloatField()
    prediction_date = models.DateTimeField(auto_now_add=True)


class SyntheticPrediction(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    predicted_social_idx = models.FloatField()
    predicted_investments_m = models.FloatField()
    predicted_crowdfunding_m = models.FloatField()
    predicted_demand_idx = models.FloatField()
    predicted_comp_idx = models.FloatField()
    prediction_date = models.DateTimeField(auto_now_add=True)


class SyntheticTimePrediction(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    predicted_social_idx = models.FloatField()
    predicted_investments_m = models.FloatField()
    predicted_crowdfunding_m = models.FloatField()
    predicted_demand_idx = models.FloatField()
    predicted_comp_idx = models.FloatField()
    prediction_date = models.DateTimeField(auto_now_add=True)
