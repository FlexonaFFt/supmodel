document.addEventListener("DOMContentLoaded", function () {
  const projectNumber = document
    .getElementById("myChart")
    .getAttribute("data-project-number");

  fetch(`/api/project-data/${projectNumber}/`)
    .then((response) => response.json())
    .then((data) => {
      const idxs = data.indeces;
      const ctx = document.getElementById("myChart").getContext("2d");
      const myChart = new Chart(ctx, {
        type: "bar",
        data: {
          labels: ["demand", "comp_idx", "team_idx", "tech_idx", "social_idx"],
          datasets: [
            {
              label: "Основные характеристики",
              data: [
                idxs.demand_idx,
                idxs.competition_idx,
                idxs.team_idx,
                idxs.tech_idx,
                idxs.social_idx,
              ],
              backgroundColor: "rgba(12, 110, 253, 0.2)",
              borderColor: "rgba(12, 110, 253, 1)",
              borderWidth: 1,
            },
          ],
        },
        options: {
          animation: {
            duration: 2000,
            easing: "easeOutBounce",
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
    })
    .catch((error) => console.error("Error fetching data:", error));
});
