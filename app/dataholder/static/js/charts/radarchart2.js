document.addEventListener("DOMContentLoaded", function () {
  const ctx = document.getElementById("synth-radarChart").getContext("2d");
  const projectNumber = document
    .getElementById("second-donutChart")
    .getAttribute("data-project-number");

  fetch(`/api/project-data/${projectNumber}/`)
    .then((response) => response.json())
    .then((data) => {
      const index = data.indeces[0];
      const prediction = data.synthetic_predictions[0];

      function clampAndAdjust(value) {
        let adjustedValue = value;
        if (value <= 0) {
          adjustedValue = 1.0;
        } else if (value >= 9.9) {
          adjustedValue = 9.9;
        } else if (value === 1.0) {
          adjustedValue += 1.0;
        } else if (value === 9.9) {
          adjustedValue -= 1.0;
        } else if (value > 1.0 && value < 9.9) {
          const adjustment = Math.random() < 0.5 ? -1.0 : 1.0;
          adjustedValue += adjustment;
        }
        return parseFloat(adjustedValue.toFixed(1));
      }

      new Chart(ctx, {
        type: "radar",
        data: {
          labels: ["demand", "comp_idx", "team_idx", "tech_idx", "social_idx"],
          datasets: [
            {
              label: "Начальные показатели",
              data: [
                index.demand_idx,
                index.competition_idx,
                index.team_idx,
                index.tech_idx,
                index.social_idx,
              ],
              backgroundColor: "rgba(12, 110, 253, 0.2)",
              borderColor: "rgba(12, 110, 253, 1)",
              borderWidth: 1,
            },
            {
              label: "Предсказанные показатели",
              data: [
                clampAndAdjust(prediction.predicted_demand_idx),
                clampAndAdjust(prediction.predicted_comp_idx),
                clampAndAdjust(index.team_idx),
                clampAndAdjust(index.tech_idx),
                clampAndAdjust(prediction.predicted_social_idx),
              ],
              backgroundColor: "rgba(75, 192, 192, 0.2)",
              borderColor: "rgba(75, 192, 192, 1)",
              borderWidth: 1,
            },
          ],
        },
        options: {
          responsive: true,
          plugins: {
            legend: {
              display: false,
              position: "top",
            },
            tooltip: {
              enabled: true,
            },
          },
          scales: {
            r: {
              angleLines: {
                display: true,
              },
              ticks: {
                display: false,
              },
              suggestedMin: 0,
              suggestedMax: 10,
            },
          },
        },
      });
    });
});
