document.addEventListener("DOMContentLoaded", function () {
  const ctx = document.getElementById("radarChart").getContext("2d");
  const projectNumber = document
    .getElementById("myChart")
    .getAttribute("data-project-number");

  fetch(`/api/project-data/${projectNumber}/`)
    .then((response) => response.json())
    .then((data) => {
      const index = data.indeces[0];
      const prediction = data.lstm_predictions[0];
      function randomAdjust(value) {
        let adjustedValue = value;
        if (value === 1.0) {
          adjustedValue += 1.0;
        } else if (value === 9.99) {
          adjustedValue -= 1.0;
        } else if (value > 1.0 && value < 9.99) {
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
                parseFloat(prediction.predicted_demand_idx.toFixed(1)),
                parseFloat(prediction.predicted_comp_idx.toFixed(1)),
                randomAdjust(index.team_idx),
                randomAdjust(index.tech_idx),
                parseFloat(prediction.predicted_social_idx.toFixed(1)),
              ],
              backgroundColor: "rgba(54, 162, 235, 0.2)",
              borderColor: "rgba(54, 162, 235, 1)",
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
