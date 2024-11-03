-- Создание таблицы user_input_data
CREATE TABLE user_input_data (
    id bigint primary key generated always as identity,
    startup_name text,
    team_name text,
    theme_id int,
    category_id int,
    description text,
    start_m int,
    investments_m int,
    crowdfunding_m int,
    team_mapping text,
    team_size int,
    team_index int,
    tech_level text,
    tech_investment int,
    competition_level text,
    competitor_count int,
    social_impact text,
    demand_level text,
    audience_reach int,
    market_size int
);

-- Создание таблицы projects
CREATE TABLE projects (
    id bigint primary key generated always as identity,
    project_name text,
    description text,
    user_input_id bigint references user_input_data(id),
    project_number int CHECK (project_number >= 100000 AND project_number <= 999999),
    is_public boolean DEFAULT true
);

-- Создание таблицы model_predictions
CREATE TABLE model_predictions (
    id bigint primary key generated always as identity,
    project_id bigint references projects(id),
    model_name text,
    predicted_social_idx numeric,
    predicted_investments_m numeric,
    predicted_crowdfunding_m numeric,
    predicted_demand_idx numeric,
    predicted_comp_idx numeric,
    prediction_date timestamp with time zone default now()
);
