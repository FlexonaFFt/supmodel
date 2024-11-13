document.addEventListener("DOMContentLoaded", function () {
  const ctx = document.getElementById("donutChart").getContext("2d");

  new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["start_m", "investments", "crowdfunding"],
      datasets: [
        {
          data: [25000, 35670, 12300],
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
