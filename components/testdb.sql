create table user_input_data (
  id bigint primary key generated always as identity,
  theme_id text,
  category_id text,
  comp_idx numeric,
  start_m numeric,
  investments_m numeric,
  crowdfunding_m numeric,
  team_idx numeric,
  tech_idx numeric,
  social_idx numeric,
  demand_idx numeric
);

create table model_predictions (
  id bigint primary key generated always as identity,
  user_input_id bigint references user_input_data (id),
  model_name text,
  predicted_social_idx numeric,
  predicted_investments_m numeric,
  predicted_crowdfunding_m numeric,
  predicted_demand_idx numeric,
  predicted_comp_idx numeric,
  prediction_date timestamp with time zone default now()
);

alter table model_predictions
drop constraint model_predictions_user_input_id_fkey;

drop table user_input_data;

create table user_input_data (
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

alter table model_predictions
add constraint model_predictions_user_input_id_fkey foreign key (user_input_id) references user_input_data (id);

alter table user_input_data
add column is_public boolean default true,
add column project_number int check (
  project_number >= 100000
  and project_number <= 999999
);

create table projects (
  id bigint primary key generated always as identity,
  project_name text,
  description text,
  user_input_id bigint references user_input_data (id)
);

alter table model_predictions
alter column model_name type text;

alter table user_input_data
drop project_number,
drop is_public;

alter table projects
add column project_number int check (
  project_number >= 100000
  and project_number <= 999999
),
add column is_public boolean default true;

alter table model_predictions
drop constraint model_predictions_user_input_id_fkey;

alter table model_predictions
add column project_id bigint references projects (id);

alter table model_predictions
drop user_input_id;
