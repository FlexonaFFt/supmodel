def calculate_team_idx(team_desc: str, experience_years: int, team_size: int) -> float:
    team_desc = team_desc.lower()
    team_mapping = {"новички": 2, "средний опыт": 5, "эксперты": 8}
    base_score = team_mapping.get(team_desc, 0)
    raw_score = (0.6 * experience_years + 0.4 * team_size) + base_score
    return max(1.0, min(9.9, round(raw_score / 3, 1)))

def calculate_tech_idx(tech_level: str, tech_investment: int) -> float:
    tech_level = tech_level.lower()
    tech_mapping = {"низкий": 2, "средний": 5, "высокий": 8}
    base_score = tech_mapping.get(tech_level, 0)
    raw_score = (0.7 * (tech_investment / 10) + 0.3 * base_score)
    return max(1.0, min(9.9, round(raw_score, 1)))

def calculate_comp_idx(comp_level: str, competitors: int) -> float:
    comp_level = comp_level.lower()
    comp_mapping = {"низкая конкуренция": 8, "средняя конкуренция": 5, "высокая конкуренция": 2}
    base_score = comp_mapping.get(comp_level, 0)
    raw_score = base_score - min(competitors / 10, base_score - 1)
    return max(1.0, min(9.9, round(raw_score, 1)))

def calculate_social_idx(social_impact: str) -> float:
    social_impact = social_impact.lower()
    social_mapping = {"низкое влияние": 3.0, "среднее влияние": 6.0, "высокое влияние": 9.0}
    return social_mapping.get(social_impact, 1.0)

def calculate_demand_idx(demand_level: str, audience_reach: int, market_size: int) -> float:
    demand_level = demand_level.lower()
    demand_mapping = {"низкий спрос": 2, "средний спрос": 5, "высокий спрос": 8}
    base_score = demand_mapping.get(demand_level, 0)
    scaled_audience = audience_reach / 10_000_000
    scaled_market = market_size / 100_000_000
    raw_score = base_score + scaled_audience + scaled_market
    return max(1.0, min(9.9, round(raw_score, 1)))

def calculate_tech_idx2(tech_level: str, tech_investment: int) -> float:
    tech_level = tech_level.lower()
    tech_mapping = {"низкий": 2, "средний": 5, "высокий": 8}
    base_score = tech_mapping.get(tech_level, 0)
    min_investment = 200
    max_investment = 50000
    normalized_investment = (tech_investment - min_investment) / (max_investment - min_investment)
    scaled_investment = normalized_investment * 10
    raw_score = 0.7 * scaled_investment + 0.3 * base_score
    return max(1.0, min(9.9, round(raw_score, 1)))

def main():
    team_idx = calculate_team_idx("эксперты", 8, 4)
    tech_idx = calculate_tech_idx2("низкий", 3500)
    comp_idx = calculate_comp_idx("средняя конкуренция", 9)
    social_idx = calculate_social_idx("среднее влияние")
    demand_idx = calculate_demand_idx("средний спрос", 200000, 500000)
    return [team_idx, tech_idx, comp_idx, social_idx, demand_idx]

if __name__ == '__main__':
    print(main())
