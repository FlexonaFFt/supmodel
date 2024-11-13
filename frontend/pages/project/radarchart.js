document.addEventListener("DOMContentLoaded", function () {
  const ctx = document.getElementById("radarChart").getContext("2d");

  new Chart(ctx, {
    type: "radar",
    data: {
      labels: ["demand", "comp_idx", "team_idx", "tech_idx", "social_idx"],
      datasets: [
        {
          label: "Athlete A",
          data: [35, 42, 56, 34, 20],
          backgroundColor: "rgba(12, 110, 253, 0.2)",
          borderColor: "rgba(12, 110, 253, 1)",
          borderWidth: 1,
        },
        {
          label: "Athlete B",
          data: [74, 100, 34, 88, 42],
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
