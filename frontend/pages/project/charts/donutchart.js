document.addEventListener("DOMContentLoaded", function () {
  const ctx = document.getElementById("donutChart").getContext("2d");

  new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Стартовый капитал", "Инвестиции", "Краудфандинг"],
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
      animation: {
        duration: 2000,
        easing: "easeOutBounce",
      },
      plugins: {
        legend: {
          display: false,
          position: "top",
        },
        tooltip: {
          enabled: true,
        },
      },
      cutout: "70%",
    },
  });
});
