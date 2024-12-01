# Generated by Django 5.0.6 on 2024-11-05 14:15

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Projects',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('project_name', models.TextField(blank=True, null=True)),
                ('description', models.TextField(blank=True, null=True)),
                ('project_number', models.IntegerField()),
                ('is_public', models.BooleanField(default=True)),
            ],
        ),
        migrations.CreateModel(
            name='UserInputData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('startup_name', models.TextField(blank=True, null=True)),
                ('team_name', models.TextField(blank=True, null=True)),
                ('theme_id', models.IntegerField(blank=True, null=True)),
                ('category_id', models.IntegerField(blank=True, null=True)),
                ('description', models.TextField(blank=True, null=True)),
                ('start_m', models.IntegerField(blank=True, null=True)),
                ('investments_m', models.IntegerField(blank=True, null=True)),
                ('crowdfunding_m', models.IntegerField(blank=True, null=True)),
                ('team_mapping', models.TextField(blank=True, null=True)),
                ('team_size', models.IntegerField(blank=True, null=True)),
                ('team_index', models.IntegerField(blank=True, null=True)),
                ('tech_level', models.TextField(blank=True, null=True)),
                ('tech_investment', models.IntegerField(blank=True, null=True)),
                ('competition_level', models.TextField(blank=True, null=True)),
                ('competitor_count', models.IntegerField(blank=True, null=True)),
                ('social_impact', models.TextField(blank=True, null=True)),
                ('demand_level', models.TextField(blank=True, null=True)),
                ('audience_reach', models.IntegerField(blank=True, null=True)),
                ('market_size', models.IntegerField(blank=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='ModelPredictions',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('model_name', models.TextField(blank=True, null=True)),
                ('predicted_social_idx', models.DecimalField(blank=True, decimal_places=2, max_digits=10, null=True)),
                ('predicted_investments_m', models.DecimalField(blank=True, decimal_places=2, max_digits=10, null=True)),
                ('predicted_crowdfunding_m', models.DecimalField(blank=True, decimal_places=2, max_digits=10, null=True)),
                ('predicted_demand_idx', models.DecimalField(blank=True, decimal_places=2, max_digits=10, null=True)),
                ('predicted_comp_idx', models.DecimalField(blank=True, decimal_places=2, max_digits=10, null=True)),
                ('prediction_date', models.DateTimeField(auto_now_add=True)),
                ('project', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='predictions', to='dataholder.projects')),
            ],
        ),
        migrations.AddField(
            model_name='projects',
            name='user_input_data',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='projects', to='dataholder.userinputdata'),
        ),
        migrations.AddConstraint(
            model_name='projects',
            constraint=models.CheckConstraint(check=models.Q(('project_number__gte', 100000), ('project_number__lte', 999999)), name='project_number_range'),
        ),
    ]