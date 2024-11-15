// Получите элемент canvas
const ctx = document.getElementById("myChart").getContext("2d");

// Создайте график
const myChart = new Chart(ctx, {
  type: "bar",
  data: {
    labels: ["demand", "comp_idx", "team_idx", "tech_idx", "social_idx"],
    datasets: [
      {
        label: "Основные характеристики",
        data: [7.5, 4.3, 3.0, 5.65, 2.35],
        backgroundColor: "rgba(12, 110, 253, 0.2)",
        borderColor: "rgba(12, 110, 253, 1)",
        borderWidth: 1,
      },
    ],
  },
  options: {
    animation: {
      duration: 2000, // Длительность анимации в миллисекундах
      easing: "easeOutBounce", // Эффект для анимации (например, 'easeOutBounce')
    },
    scales: {
      responsive: true,
      maintainAspectRatio: false,
      aspectRatio: 1,
      y: {
        beginAtZero: true,
        ticks: {
          stepSize: 1,
        },
      },
    },
  },
});
