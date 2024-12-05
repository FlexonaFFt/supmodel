document.addEventListener("DOMContentLoaded", function () {
  const projectNumber = document
    .getElementById("myChart")
    .getAttribute("data-project-number");
  const ctx = document.getElementById("second-donutChart").getContext("2d");

  fetch(`/api/project-data/${projectNumber}/`)
    .then((response) => response.json())
    .then((data) => {
      const userInput = data.user_input;
      const prediction = data.synthetic_predictions[0];
      new Chart(ctx, {
        type: "doughnut",
        data: {
          labels: ["Стартовый капитал", "Инвестиции", "Краудфандинг"],
          datasets: [
            {
              data: [
                Math.round(userInput.start_m),
                Math.round(userInput.investments_m),
                Math.round(userInput.crowdfunding_m),
              ],
              backgroundColor: [
                "rgba(12, 110, 253, 0.2)",
                "rgba(12, 110, 253, 0.4)",
                "rgba(12, 110, 253, 0.6)",
              ],
              borderColor: [
                "rgba(12, 110, 253, 1)",
                "rgba(12, 110, 253, 1)",
                "rgba(12, 110, 253, 1)",
              ],
              borderWidth: 1,
            },
            {
              data: [
                Math.round(prediction.predicted_investments_m),
                Math.round(prediction.predicted_crowdfunding_m),
              ],
              backgroundColor: [
                "rgba(75, 192, 192, 0.2)",
                "rgba(75, 192, 192, 0.4)",
              ],
              borderColor: ["rgba(75, 192, 192, 1)", "rgba(75, 192, 192, 1)"],
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
          cutout: "50%",
        },
      });
    })
    .catch((error) => console.error("Error fetching data:", error));
});
