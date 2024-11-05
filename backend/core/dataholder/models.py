from django.db import models

class UserInputData(models.Model):
    startup_name = models.TextField(blank=True, null=True)
    team_name = models.TextField(blank=True, null=True)
    theme_id = models.IntegerField(blank=True, null=True)
    category_id = models.IntegerField(blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    start_m = models.IntegerField(blank=True, null=True)
    investments_m = models.IntegerField(blank=True, null=True)
    crowdfunding_m = models.IntegerField(blank=True, null=True)
    team_mapping = models.TextField(blank=True, null=True)
    team_size = models.IntegerField(blank=True, null=True)
    team_index = models.FloatField(blank=True, null=True)
    tech_level = models.TextField(blank=True, null=True)
    tech_investment = models.IntegerField(blank=True, null=True)
    competition_level = models.TextField(blank=True, null=True)
    competitor_count = models.IntegerField(blank=True, null=True)
    social_impact = models.TextField(blank=True, null=True)
    demand_level = models.TextField(blank=True, null=True)
    audience_reach = models.IntegerField(blank=True, null=True)
    market_size = models.IntegerField(blank=True, null=True)

class Projects(models.Model):
    project_name = models.TextField(blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    user_input_data = models.ForeignKey(UserInputData, related_name='projects', on_delete=models.CASCADE)
    project_number = models.IntegerField()
    is_public = models.BooleanField(default=True) # type: ignore

    class Meta:
        constraints = [
            models.CheckConstraint(check=models.Q(project_number__gte=100000) & models.Q(project_number__lte=999999), name='project_number_range')
        ]

class ModelPredictions(models.Model):
    project = models.ForeignKey(Projects, related_name='predictions', on_delete=models.CASCADE)
    model_name = models.TextField(blank=True, null=True)
    predicted_social_idx = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    predicted_investments_m = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    predicted_crowdfunding_m = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    predicted_demand_idx = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    predicted_comp_idx = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    prediction_date = models.DateTimeField(auto_now_add=True)
