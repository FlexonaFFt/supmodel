document.addEventListener("DOMContentLoaded", function () {
  const ctx = document.getElementById("horizontalChart").getContext("2d");
  const projectNumber = document
    .getElementById("myChart")
    .getAttribute("data-project-number");

  fetch(`/api/project-data/${projectNumber}/`)
    .then((response) => response.json())
    .then((data) => {
      const userInput = data.user_input;
      const predictions = data.lstm_time_predictions;

      const initialInvestments = Math.round(userInput.investments_m);
      const initialCrowdfunding = Math.round(userInput.crowdfunding_m);

      console.log(`Initial Investments: ${initialInvestments}`);
      console.log(`Initial Crowdfunding: ${initialCrowdfunding}`);

      const predictedInvestments = predictions.map((pred) => {
        const investment = Math.round(pred.predicted_investments_m);
        const adaptedInvestment =
          investment > initialInvestments * 4
            ? Math.round(investment / 4)
            : investment;
        console.log(
          `Predicted Investment: ${investment}, Adapted: ${adaptedInvestment}`,
        );
        return adaptedInvestment;
      });

      const predictedCrowdfunding = predictions.map((pred) => {
        const crowdfunding = Math.round(pred.predicted_crowdfunding_m);
        const adaptedCrowdfunding =
          crowdfunding > initialCrowdfunding * 4
            ? Math.round(crowdfunding / 4)
            : crowdfunding;
        console.log(
          `Predicted Crowdfunding: ${crowdfunding}, Adapted: ${adaptedCrowdfunding}`,
        );
        return adaptedCrowdfunding;
      });

      const myChart = new Chart(ctx, {
        type: "bar",
        data: {
          labels: ["Start", ...predictions.map((_, index) => `H${index + 1}`)],
          datasets: [
            {
              label: "Инвестиции",
              data: [initialInvestments, ...predictedInvestments],
              backgroundColor: "rgba(255, 99, 132, 0.6)",
              borderColor: "rgba(255, 99, 132, 1)",
              borderWidth: 1,
            },
            {
              label: "Краудфандинг",
              data: [initialCrowdfunding, ...predictedCrowdfunding],
              backgroundColor: "rgba(54, 162, 235, 0.6)",
              borderColor: "rgba(54, 162, 235, 1)",
              borderWidth: 1,
            },
          ],
        },
        options: {
          responsive: true,
          indexAxis: "y",
          maintainAspectRatio: false,
          scales: {
            x: {
              beginAtZero: true,
              title: {
                display: false,
                text: "Значения",
              },
            },
            y: {
              title: {
                display: false,
                text: "Месяцы",
              },
            },
          },
          plugins: {
            legend: {
              display: false,
              position: "top",
            },
          },
        },
      });
    });
});
