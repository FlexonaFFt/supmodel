// Получите элемент canvas
const ctx = document.getElementById("myChart").getContext("2d");

// Создайте график
const myChart = new Chart(ctx, {
  type: "bar",
  data: {
    labels: ["Demand", "February", "March", "April", "May"],
    datasets: [
      {
        label: "Sample Data",
        data: [7.5, 4.3, 3.0, 5.65, 2.35],
        backgroundColor: "rgba(12, 110, 253, 0.2)",
        borderColor: "rgba(12, 110, 253, 1)",
        borderWidth: 1,
      },
    ],
  },
  options: {
    scales: {
      responsive: true,
      maintainAspectRatio: false,
      aspectRatio: 1,
      y: {
        beginAtZero: true,
      },
    },
  },
});
