document.addEventListener("DOMContentLoaded", function () {
  const projectNumber = document
    .getElementById("myChart")
    .getAttribute("data-project-number");

  fetch(`/api/project-data/${projectNumber}/`)
    .then((response) => response.json())
    .then((data) => {
      // Получаем первый элемент массива indeces
      const index = data.indeces[0];

      const ctx = document.getElementById("myChart").getContext("2d");
      const myChart = new Chart(ctx, {
        type: "bar",
        data: {
          labels: ["demand", "comp_idx", "team_idx", "tech_idx", "social_idx"],
          datasets: [
            {
              label: "Основные характеристики",
              data: [
                index.demand_idx,
                index.competition_idx,
                index.team_idx,
                index.tech_idx,
                index.social_idx,
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
            y: {
              beginAtZero: true,
              ticks: {
                stepSize: 1,
              },
            },
          },
          responsive: true,
          maintainAspectRatio: false,
        },
      });
    })
    .catch((error) => console.error("Error fetching data:", error));
});
