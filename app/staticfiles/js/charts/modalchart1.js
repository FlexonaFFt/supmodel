const ctx = document.getElementById("modalChart1").getContext("2d");

const myChart = new Chart(ctx, {
  type: "line",
  data: {
    labels: ["Start", "H1", "H2", "H3", "H4", "H5"],
    datasets: [
      {
        label: "Социальный индекс",
        data: [6.7, 6.2, 5.3, 5, 5.4, 5.2, 5.4],
        borderColor: "rgba(255, 99, 132, 1)",
        backgroundColor: "rgba(255, 99, 132, 0.2)",
        borderWidth: 2,
        tension: 0,
        fill: false,
      },
      {
        label: "Индекс спроса",
        data: [5.4, 5.4, 5.3, 5.6, 6, 5.8, 5.7],
        borderColor: "rgba(54, 162, 235, 1)",
        backgroundColor: "rgba(54, 162, 235, 0.2)",
        borderWidth: 2,
        tension: 0,
        fill: false,
      },
      {
        label: "Индекс конкурентоспособности",
        data: [7.8, 6.9, 6.8, 6.3, 6.1, 5.9, 5.8],
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
      },
    },
    plugins: {
      legend: {
        position: "top",
      },
    },
  },
});
