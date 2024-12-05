document.addEventListener("DOMContentLoaded", function () {
  const projectNumber = document
    .getElementById("myChart")
    .getAttribute("data-project-number");
  const ctx = document.getElementById("donutChart").getContext("2d");

  fetch(`/api/project-data/${projectNumber}/`)
    .then((response) => response.json())
    .then((data) => {
      const userInput = data.user_input;
      new Chart(ctx, {
        type: "doughnut",
        data: {
          labels: ["Стартовый капитал", "Инвестиции", "Краудфандинг"],
          datasets: [
            {
              data: [
                userInput.start_m,
                userInput.investments_m,
                userInput.crowdfunding_m,
              ],
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
    })
    .catch((error) => console.error("Error fetching data:", error));
});
