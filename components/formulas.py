def calculate_team_idx(team_desc, experience_years, team_size):
    team_mapping = {"новички": 2, "средний опыт": 5, "эксперты": 8}
    base_score = team_mapping[team_desc]
    return round((0.6 * experience_years + 0.4 * team_size) / 10 + base_score / 10, 1)

def calculate_tech_idx(tech_level, tech_investment):
    tech_mapping = {"низкий": 2, "средний": 5, "высокий": 8}
    base_score = tech_mapping[tech_level]
    return round((0.5 * tech_investment / 1000000 + 0.5 * base_score) / 10, 1)

def calculate_comp_idx(comp_level, competitors):
    comp_mapping = {"низкая конкуренция": 8, "средняя конкуренция": 5, "высокая конкуренция": 2}
    base_score = comp_mapping[comp_level]
    return round(base_score - (competitors / 100), 1)

def calculate_social_idx(social_impact_desc):
    social_mapping = {"низкое влияние": 3, "среднее влияние": 6, "высокое влияние": 9}
    return social_mapping[social_impact_desc]

def calculate_demand_idx(demand_level, audience_reach, market_size):
    demand_mapping = {"низкий спрос": 2, "средний спрос": 5, "высокий спрос": 8}
    base_score = demand_mapping[demand_level]
    return round((base_score + (audience_reach + market_size) / (1000000 + 100000000)) * 10, 1)

# Пример применения для одной строки данных
team_idx = calculate_team_idx('средний опыт', 10, 30)
tech_idx = calculate_tech_idx('высокий', 800000)
comp_idx = calculate_comp_idx('средняя конкуренция', 50)
social_idx = calculate_social_idx('высокое влияние')
demand_idx = calculate_demand_idx('средний спрос', 500000, 30000000)
