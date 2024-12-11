document.addEventListener("DOMContentLoaded", function () {
  const ctx = document.getElementById("modalChart2").getContext("2d");
  const projectNumber = document
    .getElementById("myChart")
    .getAttribute("data-project-number");

  fetch(`/api/project-data/${projectNumber}/`)
    .then((response) => response.json())
    .then((data) => {
      const userInput = data.indeces[0];
      const predictions = data.synthetic_time_predictions;
      const predictedSocialIndxs = predictions.map((pred) =>
        Math.round(pred.predicted_social_idx),
      );
      const predictedDemandIndxs = predictions.map((pred) =>
        Math.round(pred.predicted_demand_idx),
      );
      const predictedCompIndxs = predictions.map((pred) =>
        Math.round(pred.predicted_comp_idx),
      );

      function clamp(value, max) {
        return value > max ? max : value;
      }

      const myChart = new Chart(ctx, {
        type: "line",
        data: {
          labels: ["Start", "H1", "H2", "H3", "H4", "H5"],
          datasets: [
            {
              label: "Социальный индекс",
              data: [
                clamp(Math.round(userInput.social_idx), 10),
                ...predictedSocialIndxs.map((value) => clamp(value, 10)),
              ],
              borderColor: "rgba(255, 99, 132, 1)",
              backgroundColor: "rgba(255, 99, 132, 0.2)",
              borderWidth: 2,
              tension: 0,
              fill: false,
            },
            {
              label: "Индекс спроса",
              data: [
                clamp(Math.round(userInput.demand_idx), 10),
                ...predictedDemandIndxs.map((value) => clamp(value, 10)),
              ],
              borderColor: "rgba(54, 162, 235, 1)",
              backgroundColor: "rgba(54, 162, 235, 0.2)",
              borderWidth: 2,
              tension: 0,
              fill: false,
            },
            {
              label: "Индекс конкурентоспособности",
              data: [
                clamp(Math.round(userInput.competition_idx), 10),
                ...predictedCompIndxs.map((value) => clamp(value, 10)),
              ],
              borderColor: "rgba(75, 192, 192, 1)",
              backgroundColor: "rgba(75, 192, 192, 0.2)",
              borderWidth: 2,
              tension: 0,
              fill: false,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: {
              title: {
                display: true,
                text: "Полугодия",
              },
            },
            y: {
              beginAtZero: true,
              title: {
                display: true,
                text: "Значения",
              },
              max: 12,
            },
          },
          plugins: {
            legend: {
              position: "top",
            },
          },
        },
      });
    });
});
