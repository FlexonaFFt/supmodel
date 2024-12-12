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

      const adaptValue = (value, initialValue) => {
        const ratio = value / initialValue;
        if (ratio > 4) {
          return Math.round(value / 4);
        } else if (ratio > 3) {
          return Math.round(value / 2.5);
        } else if (ratio > 2) {
          return value;
        } else {
          return value;
        }
      };

      const predictedInvestments = predictions.map((pred) => {
        const investment = Math.round(pred.predicted_investments_m);
        const adaptedInvestment = adaptValue(investment, initialInvestments);
        return adaptedInvestment;
      });

      const predictedCrowdfunding = predictions.map((pred) => {
        const crowdfunding = Math.round(pred.predicted_crowdfunding_m);
        const adaptedCrowdfunding = adaptValue(
          crowdfunding,
          initialCrowdfunding,
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
