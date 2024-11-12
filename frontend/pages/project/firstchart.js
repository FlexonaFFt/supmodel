// Получите элемент canvas
const ctx = document.getElementById("myChart").getContext("2d");

// Создайте график
const myChart = new Chart(ctx, {
  type: "bar", // Тип графика (bar, line, pie и др.)
  data: {
    labels: ["January", "February", "March", "April", "May", "June", "July"],
    datasets: [
      {
        label: "Sample Data",
        data: [12, 19, 3, 5, 2, 3, 7],
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
      y: {
        beginAtZero: true,
      },
    },
  },
});
