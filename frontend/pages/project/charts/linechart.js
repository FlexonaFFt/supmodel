// Получите элемент canvas
const ctx = document.getElementById("lineChart").getContext("2d");

// Создайте график
const myChart = new Chart(ctx, {
  type: "line",
  data: {
    labels: ["Start", "H1", "H2", "H3", "H4", "H5"],
    datasets: [
      {
        label: "Инвестиции",
        data: [13400, 21300, 16500, 17290, 24340, 23200],
        backgroundColor: "rgba(75, 192, 192, 0.2)",
        borderColor: "rgba(75, 192, 192, 1)",
        borderWidth: 2,
        tension: 0.4,
        fill: true,
      },
      {
        label: "Краудфандинг",
        data: [8600, 10200, 7800, 19000, 32060, 21000],
        backgroundColor: "rgba(255, 99, 132, 0.2)",
        borderColor: "rgba(255, 99, 132, 1)",
        borderWidth: 2,
        tension: 0.4,
        fill: true,
      },
      {
        label: "Общая сумма",
        data: [
          13400 + 8600,
          21300 + 10200,
          16500 + 7800,
          17290 + 19000,
          24340 + 32060,
          23200 + 21000,
        ],
        backgroundColor: "rgba(54, 162, 235, 0.2)",
        borderColor: "rgba(54, 162, 235, 1)",
        borderWidth: 2,
        tension: 0.4,
        fill: true,
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
          text: "Сумма (в рублях)",
        },
        ticks: {
          stepSize: 2000,
        },
      },
    },
    plugins: {
      tooltip: {
        mode: "index",
        intersect: false,
      },
      legend: {
        position: "top",
        labels: {
          font: {
            size: 14,
          },
        },
      },
    },
  },
});
