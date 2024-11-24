document.addEventListener("DOMContentLoaded", function () {
  const projectNumber = document
    .getElementById("myChart")
    .getAttribute("data-project-number");

  fetch(`/api/project-data/${projectNumber}/`)
    .then((response) => response.json())
    .then((data) => {
      const userInput = data.user_input;

      // Функции для расчета индексов
      function calculateTeamIdx(teamDesc, experienceYears, teamSize) {
        const teamMapping = { новички: 2, "средний опыт": 5, эксперты: 8 };
        const baseScore = teamMapping[teamDesc.toLowerCase()] || 0;
        const rawScore = 0.6 * experienceYears + 0.4 * teamSize + baseScore;
        return Math.max(
          1.0,
          Math.min(9.99, Math.round((rawScore / 3) * 10) / 10),
        );
      }

      function calculateTechIdx(techLevel, techInvestment) {
        const techMapping = { низкий: 2, средний: 5, высокий: 8 };
        const baseScore = techMapping[techLevel.toLowerCase()] || 0;
        const rawScore = 0.7 * (techInvestment / 10) + 0.3 * baseScore;
        return Math.max(1.0, Math.min(9.99, Math.round(rawScore * 10) / 10));
      }

      function calculateCompIdx(compLevel, competitors) {
        const compMapping = {
          "низкая конкуренция": 8,
          "средняя конкуренция": 5,
          "высокая конкуренция": 2,
        };
        const baseScore = compMapping[compLevel.toLowerCase()] || 0;
        const rawScore = baseScore - Math.min(competitors / 10, baseScore - 1);
        return Math.max(1.0, Math.min(9.99, Math.round(rawScore * 10) / 10));
      }

      function calculateSocialIdx(socialImpact) {
        const socialMapping = {
          "низкое влияние": 3.0,
          "среднее влияние": 6.0,
          "высокое влияние": 9.0,
        };
        return socialMapping[socialImpact.toLowerCase()] || 1.0;
      }

      function calculateDemandIdx(demandLevel, audienceReach, marketSize) {
        const demandMapping = {
          "низкий спрос": 2,
          "средний спрос": 5,
          "высокий спрос": 8,
        };
        const baseScore = demandMapping[demandLevel.toLowerCase()] || 0;
        const scaledAudience = audienceReach / 10000000;
        const scaledMarket = marketSize / 100000000;
        const rawScore = baseScore + scaledAudience + scaledMarket;
        return Math.max(1.0, Math.min(9.99, Math.round(rawScore * 10) / 10));
      }

      // Расчет индексов
      const teamIdx = calculateTeamIdx(
        userInput.team_index,
        userInput.team_mapping,
        userInput.team_size,
      );
      const techIdx = calculateTechIdx(
        userInput.tech_level,
        userInput.tech_investment,
      );
      const compIdx = calculateCompIdx(
        userInput.competition_level,
        userInput.competitor_count,
      );
      const socialIdx = calculateSocialIdx(userInput.social_impact);
      const demandIdx = calculateDemandIdx(
        userInput.demand_level,
        userInput.audience_reach,
        userInput.market_size,
      );

      const tempValues = [teamIdx, techIdx, compIdx, socialIdx, demandIdx];
      console.log(tempValues);

      const values = tempValues;

      const options = [
        {
          range: [1, 2],
          text: "Проект демонстрирует очень низкие показатели по основным параметрам. Требуется значительная доработка, чтобы повысить шансы на успех.",
        },
        {
          range: [2.1, 4],
          text: "Характеристики проекта находятся на уровне ниже среднего. Хотя есть потенциал, он ограничен, и потребуется много усилий для улучшения.",
        },
        {
          range: [4.1, 7],
          text: "Проект имеет средние показатели. Это неплохая отправная точка, но для достижения высоких результатов нужно продолжать работать над ключевыми аспектами.",
        },
        {
          range: [7.1, 10],
          text: "Проект демонстрирует отличные показатели! Это сильная позиция, которая обещает высокие шансы на успех и дальнейшее развитие.",
        },
      ];

      function getTextForAverage(value) {
        const option = options.find(
          ({ range }) => value >= range[0] && value <= range[1],
        );
        return option ? option.text : "Текст не найден.";
      }

      const average =
        values.reduce((sum, value) => sum + value, 0) / values.length;
      const conclusionText = getTextForAverage(average);
      document.querySelector("#dynamic-conclusion p").textContent =
        conclusionText;
    })
    .catch((error) => console.error("Error fetching data:", error));
});
