document.addEventListener("DOMContentLoaded", function () {
  const ctx = document.getElementById("donutChart").getContext("2d");

  new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Category 1", "Category 2", "Category 3"],
      datasets: [
        {
          data: [30, 50, 20],
          backgroundColor: ["#FF6384", "#36A2EB", "#FFCE56"],
          borderColor: ["#FFFFFF", "#FFFFFF", "#FFFFFF"],
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          display: false, // Убирает легенду, если false
          position: "top",
        },
        tooltip: {
          enabled: true, // Включает/выключает всплывающие подсказки
        },
      },
      cutout: "70%", // Делает внутренний вырез для donut-графика
    },
  });
});
