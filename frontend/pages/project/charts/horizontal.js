const ctx = document.getElementById("horizontalChart").getContext("2d");

const myChart = new Chart(ctx, {
  type: "bar",
  data: {
    labels: ["Start", "H1", "H2", "H3", "H4", "H5"],
    datasets: [
      {
        label: "Инвестиции",
        data: [12340, 9900, 8650, 15120, 10200, 6000],
        backgroundColor: "rgba(255, 99, 132, 0.6)",
        borderColor: "rgba(255, 99, 132, 1)",
        borderWidth: 1,
      },
      {
        label: "Краудфандинг",
        data: [9200, 14345, 11100, 18920, 12350, 8060],
        backgroundColor: "rgba(54, 162, 235, 0.6)",
        borderColor: "rgba(54, 162, 235, 1)",
        borderWidth: 1,
      },
    ],
  },
  options: {
    responsive: true,
    indexAxis: "y",
    maintainAspectRatio: false,
    scales: {
      x: {
        beginAtZero: true,
        title: {
          display: false,
          text: "Значения",
        },
      },
      y: {
        title: {
          display: false,
          text: "Месяцы",
        },
      },
    },
    plugins: {
      legend: {
        display: false,
        position: "top",
      },
    },
  },
});

/*
// Получите элемент canvas
const ctx = document.getElementById("donutChart2").getContext("2d");

// Создайте график
const myChart = new Chart(ctx, {
  type: "doughnut",
  data: {
    labels: ["investments", "crowdfunding"],
    datasets: [
      {
        label: "Предсказанные характеристики за 5 промежуток",
        data: [10000, 28900],
        backgroundColor: ["rgba(189, 151, 243, 0.2)", "rgba(189, 151, 243, 1)"],
        borderColor: ["rgba(12, 110, 253, 1)", "rgba(12, 110, 253, 1)"],
        borderWidth: 0,
      },
      {
        label: "Предсказанные характеристики за 4 промежуток",
        data: [8000, 43000],
        backgroundColor: ["rgba(12, 110, 253, 0.2)", "rgba(12, 110, 253, 1)"],
        borderColor: ["rgba(12, 110, 253, 1)", "rgba(12, 110, 253, 1)"],
        borderWidth: 0,
      },
      {
        label: "Предсказанные характеристики за 3 промежуток",
        data: [7600, 34500],
        backgroundColor: ["rgba(54, 162, 235, 0.2)", "rgba(54, 162, 235, 1)"],
        borderColor: ["rgba(12, 110, 253, 1)", "rgba(12, 110, 253, 1)"],
        borderWidth: 0,
      },
      {
        label: "Предсказанные характеристики за 2 промежуток",
        data: [14387, 12345],
        backgroundColor: ["rgba(255, 99, 132, 0.2)", "rgba(255, 99, 132, 1)"],
        borderColor: ["rgba(12, 110, 253, 1)", "rgba(12, 110, 253, 1)"],
        borderWidth: 0,
      },
      {
        label: "Предсказанные характеристики за 1 промежуток",
        data: [16770, 28900],
        backgroundColor: ["rgba(75, 192, 192, 0.2)", "rgba(75, 192, 192, 1)"],
        borderColor: ["rgba(12, 110, 253, 1)", "rgba(12, 110, 253, 1)"],
        borderWidth: 0,
      },
      {
        label: "Базовые характеристики",
        data: [22000, 43210],
        backgroundColor: ["rgba(12, 110, 253, 0.2)", "rgba(12, 110, 253, 1)"],
        borderColor: ["rgba(12, 110, 253, 1)", "rgba(12, 110, 253, 1)"],
        borderWidth: 0,
      },
    ],
  },
  options: {
    responsive: true,
    plugins: {
      tooltip: {
        enabled: true,
      },
      legend: {
        display: false,
      },
    },
    animation: {
      animateRotate: true,
      animateScale: true,
      duration: 2000,
    },
    cutout: "20%",
  },
  }); */
