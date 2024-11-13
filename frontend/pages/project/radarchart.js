document.addEventListener("DOMContentLoaded", function () {
  const ctx = document.getElementById("radarChart").getContext("2d");

  new Chart(ctx, {
    type: "radar",
    data: {
      labels: [
        "Strength",
        "Speed",
        "Endurance",
        "Agility",
        "Intelligence",
        "Flexibility",
      ],
      datasets: [
        {
          label: "Athlete A",
          data: [65, 75, 70, 80, 60, 75],
          backgroundColor: "rgba(255, 99, 132, 0.2)",
          borderColor: "rgba(255, 99, 132, 1)",
          borderWidth: 1,
        },
        {
          label: "Athlete B",
          data: [54, 65, 60, 70, 78, 82],
          backgroundColor: "rgba(54, 162, 235, 0.2)",
          borderColor: "rgba(54, 162, 235, 1)",
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          display: false,
          position: "top",
        },
        tooltip: {
          enabled: true,
        },
      },
      scales: {
        r: {
          angleLines: {
            display: true,
          },
          ticks: {
            display: false,
          },
          suggestedMin: 0,
          suggestedMax: 100,
        },
      },
    },
  });
});
