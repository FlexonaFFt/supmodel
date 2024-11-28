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

      const predictedInvestments = predictions.map((pred) =>
        Math.round(pred.predicted_investments_m),
      );
      const predictedCrowdfunding = predictions.map((pred) =>
        Math.round(pred.predicted_crowdfunding_m),
      );

      const myChart = new Chart(ctx, {
        type: "bar",
        data: {
          labels: ["Start", ...predictions.map((_, index) => `H${index + 1}`)],
          datasets: [
            {
              label: "Инвестиции",
              data: [
                Math.round(userInput.investments_m),
                ...predictedInvestments,
              ],
              backgroundColor: "rgba(255, 99, 132, 0.6)",
              borderColor: "rgba(255, 99, 132, 1)",
              borderWidth: 1,
            },
            {
              label: "Краудфандинг",
              data: [
                Math.round(userInput.crowdfunding_m),
                ...predictedCrowdfunding,
              ],
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
/*
// Получите элемент canvas
const ctx = document.getElementById("donutChart2").getContext("2d");

// Создайте график
const myChart = new Chart(ctx, {
  type: "doughnut",
  data: {
    labels: ["investments", "crowdfunding"],
    datasets: [
      {
        label: "Предсказанные характеристики за 5 промежуток",
        data: [10000, 28900],
        backgroundColor: ["rgba(189, 151, 243, 0.2)", "rgba(189, 151, 243, 1)"],
        borderColor: ["rgba(12, 110, 253, 1)", "rgba(12, 110, 253, 1)"],
        borderWidth: 0,
      },
      {
        label: "Предсказанные характеристики за 4 промежуток",
        data: [8000, 43000],
        backgroundColor: ["rgba(12, 110, 253, 0.2)", "rgba(12, 110, 253, 1)"],
        borderColor: ["rgba(12, 110, 253, 1)", "rgba(12, 110, 253, 1)"],
        borderWidth: 0,
      },
      {
        label: "Предсказанные характеристики за 3 промежуток",
        data: [7600, 34500],
        backgroundColor: ["rgba(54, 162, 235, 0.2)", "rgba(54, 162, 235, 1)"],
        borderColor: ["rgba(12, 110, 253, 1)", "rgba(12, 110, 253, 1)"],
        borderWidth: 0,
      },
      {
        label: "Предсказанные характеристики за 2 промежуток",
        data: [14387, 12345],
        backgroundColor: ["rgba(255, 99, 132, 0.2)", "rgba(255, 99, 132, 1)"],
        borderColor: ["rgba(12, 110, 253, 1)", "rgba(12, 110, 253, 1)"],
        borderWidth: 0,
      },
      {
        label: "Предсказанные характеристики за 1 промежуток",
        data: [16770, 28900],
        backgroundColor: ["rgba(75, 192, 192, 0.2)", "rgba(75, 192, 192, 1)"],
        borderColor: ["rgba(12, 110, 253, 1)", "rgba(12, 110, 253, 1)"],
        borderWidth: 0,
      },
      {
        label: "Базовые характеристики",
        data: [22000, 43210],
        backgroundColor: ["rgba(12, 110, 253, 0.2)", "rgba(12, 110, 253, 1)"],
        borderColor: ["rgba(12, 110, 253, 1)", "rgba(12, 110, 253, 1)"],
        borderWidth: 0,
      },
    ],
  },
  options: {
    responsive: true,
    plugins: {
      tooltip: {
        enabled: true,
      },
      legend: {
        display: false,
      },
    },
    animation: {
      animateRotate: true,
      animateScale: true,
      duration: 2000,
    },
    cutout: "20%",
  },
  }); */
