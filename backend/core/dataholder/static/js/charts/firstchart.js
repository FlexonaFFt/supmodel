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

      const values = [demandIdx, compIdx, teamIdx, techIdx, socialIdx];
      const ctx = document.getElementById("myChart").getContext("2d");
      const myChart = new Chart(ctx, {
        type: "bar",
        data: {
          labels: ["demand", "comp_idx", "team_idx", "tech_idx", "social_idx"],
          datasets: [
            {
              label: "Основные характеристики",
              data: values,
              backgroundColor: "rgba(12, 110, 253, 0.2)",
              borderColor: "rgba(12, 110, 253, 1)",
              borderWidth: 1,
            },
          ],
        },
        options: {
          animation: {
            duration: 2000, // Длительность анимации в миллисекундах
            easing: "easeOutBounce", // Эффект для анимации (например, 'easeOutBounce')
          },
          scales: {
            responsive: true,
            maintainAspectRatio: false,
            aspectRatio: 1,
            y: {
              beginAtZero: true,
              ticks: {
                stepSize: 1,
              },
            },
          },
        },
      });
    })
    .catch((error) => console.error("Error fetching data:", error));
});
